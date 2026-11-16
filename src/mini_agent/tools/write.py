"""Write 工具：移植自 Claude-Code FileWriteTool。

与 Edit 的分工（TS 侧 prompt）：修改已有文件优先用 Edit（只发 diff），
Write 仅用于新建或完整重写；绝不主动建 *.md/README。

保留的机制：
- 覆盖保护（mtime 乐观锁）：文件已存在时必须先 Read 过且未过期，
  否则报错——校验阶段一次、写入临界区内一次（TS 侧重读检查）
- 父目录自动创建；写后回写读状态，后续 Edit 才能工作
- 内容原样写入（不做行尾改写；TS 侧教训：保留旧行尾曾把 bash 脚本
  写坏），文件编码统一 UTF-8
"""
from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolError, ToolOutput, ValidationResult
from .filestate import (
    MAX_FILE_BYTES,
    FileState,
    FileToolContext,
    is_unc,
    resolve_in_workdir,
)


class WriteTool(Tool):
    name = "Write"
    description = (
        "把完整内容写入文件（新建或整文件覆盖）。\n"
        "用法：\n"
        "- 已存在的文件会整体覆盖；覆盖前必须先用 Read 读过该文件，否则工具会失败。\n"
        "- 修改已有文件优先用 Edit——它只发送差异；Write 只用于新建或完整重写。\n"
        "- 父目录不存在会自动创建。\n"
        "- 绝不主动创建文档文件（*.md / README），除非用户明确要求。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要写入的文件路径（相对路径基于工作目录；绝对路径必须在工作目录内）",
            },
            "content": {"type": "string", "description": "完整文件内容"},
        },
        "required": ["file_path", "content"],
        "additionalProperties": False,
    }

    def is_concurrency_safe(self) -> bool:
        return False  # 写文件是屏障：同一轮里必须按 tool_calls 顺序执行

    def validate_input(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        if not isinstance(ctx, FileToolContext):
            return ValidationResult.fail("Write 需要 FileToolContext", 0)
        raw = args.get("file_path")
        if not isinstance(raw, str) or not raw.strip():
            return ValidationResult.fail("file_path 必须是非空字符串", 1)
        if is_unc(raw.strip()):
            return ValidationResult.fail(
                f"不支持 UNC 路径 {raw!r}（防止 SMB/NTLM 凭据泄漏）", 2
            )
        content = args.get("content")
        if not isinstance(content, str):
            return ValidationResult.fail("content 必须是字符串", 4)
        try:
            resolve_in_workdir(ctx.workdir, raw.strip())
        except ToolError as exc:
            return ValidationResult.fail(str(exc), error_code=exc.error_code)
        return ValidationResult.pass_()

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        if not isinstance(ctx, FileToolContext):
            raise ToolError("Write 需要 FileToolContext", 0)
        fctx = ctx
        target = resolve_in_workdir(fctx.workdir, args["file_path"].strip())
        content: str = args["content"]

        with fctx.exclusive_path(target):
            if target.is_dir():
                raise ToolError(f"这是目录不是文件：{target}", 4)
            existed = target.exists()

            if existed:
                if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                    raise ToolError(
                        f"内容过大（上限 {MAX_FILE_BYTES >> 30} GiB），请分段写入", 9
                    )
                # 覆盖保护：必须先 Read 且未过期（mtime 假阳性有内容兜底）
                fctx.check_fresh(target)

            # 写盘：内容原样（UTF-8）；父目录自动创建（TS 侧 Write 行为）
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))

            # 回写读状态：后续 Edit/再次 Write 以它为新鲜基准
            fctx.state.set(
                target,
                FileState(
                    content=content.replace("\r\n", "\n"),
                    mtime_ns=fctx.stat_mtime_ns(target),
                    offset=None,
                    limit=None,
                    encoding="utf-8",
                    eol="\n",
                ),
            )

        lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        if existed:
            return ToolOutput(
                f"已更新 {target}（{lines} 行，{len(content)} 字符）",
                {"type": "update", "path": str(target)},
            )
        return ToolOutput(
            f"已新建 {target}（{lines} 行，{len(content)} 字符）",
            {"type": "create", "path": str(target)},
        )
