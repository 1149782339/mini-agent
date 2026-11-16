"""Read 工具：移植自 Claude-Code FileReadTool（文本路径）。

保留的机制：
- cat -n 行号视图（行号 + 制表符，前缀不是文件内容）
- offset(1 起)/limit 分页；整读字节帽 256KB，超出报错并提示分段（TS 侧
  结论：报错比截断省 token——错误约 100 字节 vs 25K token 截断内容）
- token 输出帽（粗估，超帽报错提示 offset/limit）
- file_unchanged 去重：同范围同 mtime 直接回 stub（TS 侧数据：约 18%
  的 Read 是重复读）
- 读状态登记到 FileStateCache，供 Edit/Write 做 read-before-write 闸
- 双路径读取：小文件读全文再切片；大文件流式只保留范围内行
- 二进制 NUL 嗅探、设备文件黑名单（/dev/zero 等防挂起）、UNC 零 I/O

图片（jpg/png/gif/webp/bmp）走同一工具：魔数识别后带回图片块，不另设工具。
PDF 和 ipynb 仍不读。
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from ..tokens import estimate_text_tokens
from .base import Tool, ToolContext, ToolError, ToolOutput, ValidationResult, semantic_int
from .image import process_image, sniff_image_mime
from .filestate import (
    MAX_OUTPUT_SIZE,
    FileState,
    FileToolContext,
    detect_encoding,
    find_similar_file,
    is_unc,
    read_disk_text,
    resolve_anywhere,
)

FILE_UNCHANGED_STUB = (
    "File unchanged since last read. The content from the earlier Read "
    "tool_result in this conversation is still current — refer to that "
    "instead of re-reading."
)

MAX_LINES_TO_READ = 2000  # TS 侧 MAX_LINES_TO_READ（默认读取行数）
MAX_BODY_CHARS = 50000    # Pi truncate.ts：单次输出字节帽（先到先赢）
MAX_LINE_CHARS = 2000     # 单行截断（防 minified 行刷屏，Grep 侧还有 --max-columns 500）
_STREAM_THRESHOLD = 10 * 1024 * 1024  # TS 侧 readFileInRange 快路径阈值

# TS 侧 isBlockedDevicePath：会阻塞或产生无限输出的设备文件（路径黑名单，无 I/O）
_BLOCKED_DEVICES = {
    "/dev/zero", "/dev/urandom", "/dev/random", "/dev/full", "/dev/stdin",
}


class ReadTool(Tool):
    name = "Read"
    description = (
        "读取文本文件，返回 cat -n 格式内容（行号 + 制表符 + 行内容，"
        "行号前缀不是文件内容，不要把它复制进后续编辑）。"
        "也读取图片（jpg、png、gif、webp、bmp），图片作为附件发给支持视觉的模型。\n"
        "用法：\n"
        "- file_path 必填，相对路径基于工作目录，绝对路径任意位置均可。\n"
        "- 默认从头读最多 2000 行；长文件用 offset/limit 分段读。\n"
        "- 整读有 256KB 字节帽，超限会报错，按提示用 offset/limit 分段。\n"
        "- 同一范围重复读取会收到 file_unchanged 提示，直接引用之前的结果即可。\n"
        "- 只能读文件；看目录请用 Ls。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要读取的文件路径（相对路径基于工作目录，绝对路径任意位置均可）",
            },
            "offset": {
                "type": "integer",
                "description": "起始行号（从 1 开始）。仅当文件太大一次读不完时才需要",
            },
            "limit": {
                "type": "integer",
                "description": "最多读取行数。仅当文件太大一次读不完时才需要",
            },
        },
        "required": ["file_path"],
        "additionalProperties": False,
    }

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    # ---------- 第 1 段：纯校验（零 I/O） ----------
    def validate_input(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        raw = args.get("file_path")
        if not isinstance(raw, str) or not raw.strip():
            return ValidationResult.fail("file_path 必须是非空字符串", error_code=1)
        if is_unc(raw.strip()):
            # TS 侧语义：UNC 路径 defer 到授权阶段；mini-agent 没有授权层，直接拒绝
            return ValidationResult.fail(
                f"不支持 UNC 路径 {raw!r}（防止 SMB/NTLM 凭据泄漏）", error_code=2
            )
        try:
            resolve_anywhere(ctx.workdir, raw.strip())
        except ToolError as exc:
            return ValidationResult.fail(str(exc), error_code=exc.error_code)
        name = Path(raw.strip()).name
        if name in _BLOCKED_DEVICES:
            return ValidationResult.fail(
                f"不能读取设备文件 {raw!r}：会阻塞或产生无限输出", error_code=9
            )
        return ValidationResult.pass_()

    # ---------- 第 3 段：执行 ----------
    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        if not isinstance(ctx, FileToolContext):
            raise ToolError("Read 需要 FileToolContext", 0)
        fctx = ctx
        target = resolve_anywhere(fctx.workdir, args["file_path"].strip())

        offset = semantic_int(args.get("offset"), 1) or 1
        offset = max(1, offset)
        limit = semantic_int(args.get("limit"), None)
        if limit is not None:
            limit = max(1, min(limit, 50000))

        if target.is_dir():
            raise ToolError(f"这是目录不是文件：{target}。看目录请用 Ls", 4)
        if not target.exists():
            hint = find_similar_file(target)
            msg = f"文件不存在：{target}"
            if hint is not None:
                msg += f"。你是否想找 {hint}？"
            raise ToolError(msg, 3)

        mime = sniff_image_mime(target)
        if mime:
            return self._read_image(target, mime)

        # 去重：仅对"模型亲眼读过"的条目（offset 非 None）且范围完全一致生效；
        # Edit/Write 回写的条目 offset=None，不参与去重（TS 侧 543-551 注释）
        existing = fctx.state.get(target)
        if (
            existing is not None
            and not existing.is_partial_view
            and existing.offset is not None
            and existing.offset == offset
            and existing.limit == limit
        ):
            try:
                if fctx.stat_mtime_ns(target) == existing.mtime_ns:
                    return ToolOutput(FILE_UNCHANGED_STUB, {"type": "file_unchanged"})
            except OSError:
                pass  # stat 失败则走完整读取

        # 读取：快路径拿全文，流式路径拿片段 + 总行数
        full_text: str | None = None
        size = target.stat().st_size
        if limit is None and size > fctx.max_file_bytes:
            raise ToolError(
                f"文件过大（{size} 字节，整读上限 {MAX_OUTPUT_SIZE // 1024} KB）：{target}。"
                f"请用 offset/limit 分段读取。", 6
            )
        if limit is None or size <= _STREAM_THRESHOLD:
            full_text, encoding, eol = read_disk_text(target)
            all_lines = full_text.split("\n")
            if all_lines and all_lines[-1] == "":
                all_lines.pop()
            total_lines = len(all_lines)
            end = total_lines if limit is None else min(offset + limit - 1, total_lines)
            chunk = all_lines[offset - 1 : end] if offset <= total_lines else []
        else:
            with open(target, "rb") as fh:
                encoding, _ = detect_encoding(fh.read(4))
            chunk, total_lines = _stream_lines(target, encoding, offset, limit)
            eol = "\r\n" if any("\r" in ln for ln in chunk[:50]) else "\n"

        if full_text is not None:
            head = full_text[:8192]
        else:
            head = "\n".join(chunk[:200])
        if "\x00" in head:
            raise ToolError(
                f"无法读取二进制文件 {target}。它是二进制文件，请用合适的工具分析。", 5
            )

        # 单行截断（Pi：超长行不能整行塞给模型）
        chunk = [
            (ln[:MAX_LINE_CHARS] + " …[本行过长已截断]") if len(ln) > MAX_LINE_CHARS else ln
            for ln in chunk
        ]

        body = "\n".join(f"{i + offset:>6}\t{line}" for i, line in enumerate(chunk))
        # 输出字节帽（Pi truncateHead：保头、绝不在行中间切断）
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS]
            if "\n" in body:
                body = body[: body.rfind("\n")]
            body += (
                f"\n[单次输出超过 {MAX_BODY_CHARS} 字符已截断，"
                f"用 offset/limit 继续读]"
            )
        if estimate_text_tokens(body) > fctx.max_read_tokens:
            raise ToolError(
                f"读取结果超出 token 上限（约 {fctx.max_read_tokens}）。"
                f"请用 offset/limit 缩小范围，或先用 Grep 定位。", 8
            )

        if total_lines == 0:
            content = "<system-reminder>Warning: 文件存在但内容为空。</system-reminder>"
        elif not chunk:
            content = (
                f"<system-reminder>Warning: 文件存在但比 offset ({offset}) 短，"
                f"共 {total_lines} 行。</system-reminder>"
            )
        else:
            shown_end = offset + len(chunk) - 1
            content = body + f"\n[共 {total_lines} 行，已显示 {offset}-{shown_end}]"
            if shown_end < total_lines:
                content += f"\n（还有 {total_lines - shown_end} 行，用 offset={shown_end + 1} 继续读）"

        # 登记读状态：Edit/Write 的前置条件 + 下次去重基准。
        # 快路径 content 是全文；流式路径只有片段（check_fresh 只对全文读做内容兜底）。
        fctx.state.set(
            target,
            FileState(
                content=full_text if full_text is not None else "\n".join(chunk),
                mtime_ns=fctx.stat_mtime_ns(target),
                offset=offset,
                limit=limit,
                encoding=encoding,
                eol=eol,
            ),
        )
        return ToolOutput(content, {"type": "text", "path": str(target)})

    def _read_image(self, target: Path, mime: str) -> ToolOutput:
        """图片不进读状态缓存，也不走行号视图。结果是说明文字加图片块。"""
        try:
            raw = target.read_bytes()
        except OSError as exc:
            raise ToolError(f"无法读取 {target}：{exc}", 5) from None
        processed = process_image(raw, mime)
        if not processed.ok:
            note = f"Read image file [{mime}]\n{processed.message}"
            return ToolOutput(note, {"type": "image", "path": str(target), "ok": False})
        note = f"Read image file [{processed.mime_type}]"
        if processed.hints:
            note += "\n" + "\n".join(processed.hints)
        return ToolOutput(
            note,
            {"type": "image", "path": str(target), "mimeType": processed.mime_type},
            images=[{"data": processed.data, "mimeType": processed.mime_type}],
        )


def _stream_lines(target: Path, encoding: str, offset: int,
                  limit: int | None) -> tuple[list[str], int]:
    """流式路径（TS 侧 readFileInRange）：只保留范围内行，范围外只计数。

    返回 (片段行列表, 总行数)。UTF-16 逐行解码交给 io 的增量解码器。
    """
    chunk: list[str] = []
    total = 0
    stop = offset + limit if limit is not None else None
    with open(target, "rb") as fh:
        wrapper = io.TextIOWrapper(fh, encoding=encoding, errors="replace")
        for idx, line in enumerate(wrapper, start=1):
            total = idx  # 扫到文件尾拿真实总行数（TS 侧同样读穿整个文件）
            if idx >= offset and (stop is None or idx < stop):
                chunk.append(line.rstrip("\r\n"))
    return chunk, total
