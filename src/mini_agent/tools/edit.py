"""Edit 工具：移植自 Claude-Code FileEditTool（精确字符串替换）。

保留的机制：
- read-before-write：必须先用 Read 读过该文件（FileStateCache 前置闸），
  且磁盘 mtime 未变；mtime 变化时对全文读回退比对内容（Windows 云同步/
  杀软会造成 mtime 假阳性），不一致则要求重新 Read
- 匹配在"CRLF 归一化"后的磁盘内容上做；写盘按原编码/行尾还原
- old_string 不唯一时报错并提示加上下文或 replace_all
- 弯引号容错（TS 侧 preserveQuoteStyle）：old_string 里的弯引号 ' '' "
  匹配失败后按直引号重试，命中后对 new_string 反向施加弯引号
- new_string 去行尾空白；.md/.mdx 例外（行尾两空格是 Markdown 硬换行）
- 文件大小上限 1 GiB；UNC 零 I/O；old/new 相同早退

错误信息即自愈指引（TS 侧设计）：每个失败都告诉模型下一步该做什么。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    Tool, ToolContext, ToolError, ToolOutput, ValidationResult, semantic_bool,
)
from .filestate import (
    MAX_FILE_BYTES,
    FileState,
    FileToolContext,
    find_similar_file,
    is_unc,
    read_disk_text,
    resolve_in_workdir,
    write_disk_text,
)

# TS 侧 normalizeFileEditInput：Markdown 行尾两空格是硬换行，不能剥
_TRAILING_WS_EXEMPT = {".md", ".mdx"}

# 弯引号 → 直引号（TS 侧 curly quote normalization）
_QUOTE_TRANSLATION = {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'}
_OPENING_CONTEXT = set(" \t\n([{<—–-'\"‘’“”")


def _straighten(s: str) -> str:
    return s.translate(_QUOTE_TRANSLATION)


def _find_actual(text: str, old: str) -> tuple[int, str] | None:
    """先精确匹配；失败则弯引号归一化后重试（TS 侧 findActualString）。

    弯引号与直引号都是单字符，归一化后索引 1:1 对齐，切出原文子串返回。
    """
    idx = text.find(old)
    if idx >= 0:
        return idx, old
    flat_text, flat_old = _straighten(text), _straighten(old)
    idx = flat_text.find(flat_old)
    if idx < 0:
        return None
    return idx, text[idx : idx + len(old)]


def _curly_heuristic(s: str, i: int) -> bool:
    """引号是否为"开"：前是空白/开括号/破折号/行首视为开（TS 侧启发式）。"""
    return i == 0 or s[i - 1] in _OPENING_CONTEXT


def _restore_quotes(new: str, actual_old: str) -> str:
    """对 new_string 反向施加弯引号（TS 侧 preserveQuoteStyle）。

    先按位置对应：old 该位置是弯引号且 new 同位置是直引号 → 换成同款弯引号；
    剩余/新增的直引号按开闭启发式配对；字母间的撇号（don't）特判为右单引号。
    """
    chars = list(new)
    for i, ch in enumerate(actual_old):
        if i >= len(chars):
            break
        if ch in "‘’“”" and chars[i] in "'\"":
            chars[i] = ch
    for i, ch in enumerate(chars):
        if ch == "'":
            if 0 < i < len(chars) - 1 and chars[i - 1].isalpha() and chars[i + 1].isalpha():
                chars[i] = "’"  # 缩写撇号特判
            else:
                chars[i] = "‘" if _curly_heuristic(chars, i) else "’"
        elif ch == '"':
            chars[i] = "“" if _curly_heuristic(chars, i) else "”"
    return "".join(chars)


def _strip_trailing_ws(s: str, path: Path) -> str:
    if path.suffix.lower() in _TRAILING_WS_EXEMPT:
        return s
    return "\n".join(line.rstrip() for line in s.split("\n"))


class EditTool(Tool):
    name = "Edit"
    description = (
        "在文件中做精确字符串替换。\n"
        "用法：\n"
        "- 编辑前必须先用 Read 读过该文件，否则工具会报错。\n"
        "- 从 Read 输出复制 old_string 时，只取行号前缀（行号 + 制表符）之后的内容，"
        "缩进必须与原文件完全一致。\n"
        "- 优先编辑已有文件；除非明确要求，不要新建文件。\n"
        "- old_string 在文件中必须唯一，否则编辑失败：请提供更长的上下文使其唯一，"
        "或用 replace_all 替换所有出现。\n"
        "- replace_all 适合跨文件改名、批量替换同一字符串。\n"
        "- 行尾/编码保真：CRLF 文件写回仍是 CRLF，UTF-16 写回仍是 UTF-16。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要修改的文件路径（相对路径基于工作目录；绝对路径必须在工作目录内）",
            },
            "old_string": {"type": "string", "description": "要被替换的原文"},
            "new_string": {
                "type": "string",
                "description": "替换后的文本（必须与 old_string 不同）",
            },
            "replace_all": {
                "type": "boolean",
                "description": "替换文件中所有 old_string 出现处（默认 false）",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
        "additionalProperties": False,
    }

    def is_concurrency_safe(self) -> bool:
        return False  # 写文件是屏障：同一轮里必须按 tool_calls 顺序执行

    # ---------- 第 1 段：校验 + 预匹配（错误在写入前全部暴露） ----------
    def validate_input(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        parsed = self._parse_args(args, ctx)
        if isinstance(parsed, ValidationResult):
            return parsed
        fctx, target, old, new, replace_all = parsed

        if old == new:
            return ValidationResult.fail(
                "old_string 与 new_string 相同，没有可做的修改", error_code=6
            )
        if old == "":
            return ValidationResult.fail(
                f"old_string 为空。要在 {target.name} 追加内容请提供一个唯一的锚点，"
                "或确认语义后使用 Write 完整重写", error_code=7
            )
        if not target.exists():
            hint = find_similar_file(target)
            msg = f"文件不存在：{target}（Edit 不能新建文件，请用 Write）"
            if hint is not None:
                msg += f"。你是否想找 {hint}？"
            return ValidationResult.fail(msg, error_code=3)
        if target.is_dir():
            return ValidationResult.fail(f"这是目录不是文件：{target}", error_code=4)
        if target.suffix.lower() == ".ipynb":
            return ValidationResult.fail(
                "不支持直接编辑 .ipynb；请用 run_command 配合 python/jq 处理", 8
            )
        try:
            if target.stat().st_size > MAX_FILE_BYTES:
                return ValidationResult.fail(
                    f"文件过大（上限 {MAX_FILE_BYTES >> 30} GiB），请分段处理", 9
                )
        except OSError:
            return ValidationResult.fail(f"无法 stat {target}", 9)

        # read-before-write + staleness 闸（mtime 假阳性有内容兜底）
        try:
            fctx.check_fresh(target)
        except ToolError as exc:
            return ValidationResult.fail(str(exc), error_code=exc.error_code)

        text, _, _ = read_disk_text(target)
        found = _find_actual(text, old)
        if found is None:
            return ValidationResult.fail(
                f"未找到 old_string（{len(old)} 字符）。它可能已被修改——"
                f"请重新 Read {target.name}，并逐字复制（注意缩进/制表符）。",
                error_code=10,
            )
        _, actual = found
        hits = text.count(actual)
        if hits > 1 and not replace_all:
            return ValidationResult.fail(
                f"old_string 命中 {hits} 处，不唯一。请提供更多上下文使其唯一，"
                "或设置 replace_all=true 替换全部。", 11
            )
        return ValidationResult.pass_()

    # ---------- 第 3 段：执行（校验后同步完成，无 await 间隙） ----------
    def call(self, args: dict[str, Any], ctx: ToolContext):
        parsed = self._parse_args(args, ctx)
        if isinstance(parsed, ValidationResult):  # dispatch 已拦，防御
            raise ToolError(parsed.message, parsed.error_code)
        fctx, target, old, new, replace_all = parsed

        # 同一路径互斥：check_fresh 到写盘必须在同一把锁里
        with fctx.exclusive_path(target):
            fctx.check_fresh(target)

            text, encoding, eol = read_disk_text(target)
            found = _find_actual(text, old)
            if found is None:
                raise ToolError("文件在校验后被修改，old_string 不再匹配，请重新 Read", 12)
            idx, actual = found

            new_clean = _strip_trailing_ws(new, target)
            if actual != old:  # 弯引号路径：还原 new_string 的引号风格
                new_clean = _restore_quotes(new_clean, actual)

            if replace_all:
                hits = text.count(actual)
                new_text = text.replace(actual, new_clean)
            else:
                hits = 1
                new_text = text[:idx] + new_clean + text[idx + len(actual) :]

            write_disk_text(target, new_text, encoding, eol)
            mtime = fctx.stat_mtime_ns(target)
            fctx.state.set(
                target,
                FileState(content=new_text, mtime_ns=mtime, offset=None, limit=None,
                          encoding=encoding, eol=eol),
            )

        old_lines = len(actual.split("\n"))
        new_lines = len(new_clean.split("\n"))
        where = f"{hits} 处" if replace_all else "1 处"
        return ToolOutput(
            f"已编辑 {target}：替换 {where}（-{old_lines}/+{new_lines} 行，"
            f"行尾 {'CRLF' if eol == '\r\n' else 'LF'} 已保真）",
            {"type": "edit", "path": str(target), "hits": hits},
        )

    # ---------- 内部 ----------
    def _parse_args(self, args: dict[str, Any], ctx: ToolContext):
        """解析参数 + 解析路径。返回 (fctx, target, old, new, replace_all)
        或 ValidationResult（失败）。"""
        if not isinstance(ctx, FileToolContext):
            return ValidationResult.fail("Edit 需要 FileToolContext", 0)
        raw = args.get("file_path")
        if not isinstance(raw, str) or not raw.strip():
            return ValidationResult.fail("file_path 必须是非空字符串", 1)
        if is_unc(raw.strip()):
            return ValidationResult.fail(
                f"不支持 UNC 路径 {raw!r}（防止 SMB/NTLM 凭据泄漏）", 2
            )
        old = args.get("old_string")
        new = args.get("new_string")
        if not isinstance(old, str):
            return ValidationResult.fail("old_string 必须是字符串", 4)
        if not isinstance(new, str):
            return ValidationResult.fail("new_string 必须是字符串", 5)
        try:
            target = resolve_in_workdir(ctx.workdir, raw.strip())
        except ToolError as exc:
            return ValidationResult.fail(str(exc), error_code=exc.error_code)
        return ctx, target, old, new, semantic_bool(args.get("replace_all"), False)
