"""Glob 工具：移植自 Claude-Code GlobTool。

底层是 `rg --files --glob pattern --sort=modified`（不是 Python glob），
免费获得 .gitignore 语义和仓库级性能。保留的机制：

- 结果按修改时间排序（最新在前），截断 100 条并给出 truncated 提示
- 输出路径相对工作目录，省 token
- 超时与"无匹配"区分；超时有部分结果则丢弃残缺尾行后返回
- ENOENT 时 "Did you mean X?" 建议
- 忽略语义：默认 rg 原生（跳过隐藏文件 + 尊重 .gitignore）；
  MINI_AGENT_RG_ALL=1 可切换为 --hidden --no-ignore（TS 侧默认）
"""
from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolError, ToolOutput, ValidationResult
from .filestate import FileToolContext, is_unc, resolve_anywhere
from .rg import run_rg, rg_all_flag

LIMIT = 100          # TS 侧 GlobTool 默认截断条数
TIMEOUT = 20         # TS 侧 CLAUDE_CODE_GLOB_TIMEOUT_SECONDS 默认 20s


class GlobTool(Tool):
    name = "Glob"
    description = (
        "按文件名模式快速匹配文件，任意规模的代码库都适用。\n"
        "- 支持 **/*.py、src/**/*.ts 这类 glob 模式。\n"
        "- 返回匹配路径，按修改时间排序（最新在前），最多 100 条。\n"
        "- 用于按名字找文件；找内容请用 Grep。\n"
        "- 搜索范围用 path 参数指定，pattern 只写相对模式。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "glob 模式，如 **/*.py、src/**/test_*.py",
            },
            "path": {
                "type": "string",
                "description": "搜索的目录（相对路径基于工作目录，绝对路径任意位置均可），默认工作目录",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def validate_input(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return ValidationResult.fail("pattern 必须是非空字符串", 1)
        if Path(pattern).is_absolute() or ":" in pattern.split("/")[0]:
            return ValidationResult.fail(
                "pattern 请用相对模式（如 **/*.py）；搜索范围用 path 参数指定", 2
            )
        raw = args.get("path")
        if raw is not None:
            if not isinstance(raw, str) or not raw.strip():
                return ValidationResult.fail("path 必须是非空字符串", 3)
            if is_unc(raw.strip()):
                return ValidationResult.fail(
                    f"不支持 UNC 路径 {raw!r}（防止 SMB/NTLM 凭据泄漏）", 4
                )
            try:
                resolve_anywhere(ctx.workdir, raw.strip())
            except ToolError as exc:
                return ValidationResult.fail(str(exc), error_code=exc.error_code)
        return ValidationResult.pass_()

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        if not isinstance(ctx, FileToolContext):
            raise ToolError("Glob 需要 FileToolContext", 0)
        pattern = args["pattern"].strip()
        raw_path = args.get("path")
        target = resolve_anywhere(
            ctx.workdir, raw_path.strip() if isinstance(raw_path, str) else "."
        )
        if not target.is_dir():
            raise ToolError(
                f"不是目录：{target}（Glob 只搜索目录；找单个文件用 Read）", 5
            )

        out, timed_out = run_rg(
            ["--files", "--glob", pattern, "--sortr=modified", *rg_all_flag()],
            cwd=target,
            timeout=TIMEOUT,
        )
        files = [ln for ln in out.split("\n") if ln.strip()]
        truncated = len(files) > LIMIT
        files = files[:LIMIT]

        prefix = os.path.relpath(target, ctx.workdir)
        if prefix != ".":
            files = [f"{prefix}/{f}" for f in files]

        if not files:
            body = "No files found"
            if timed_out:
                body += "（搜索超时，结果可能不完整；请缩小 path 或精确化 pattern）"
            return ToolOutput(body, {"type": "glob", "matches": 0})

        body = "\n".join(files)
        notes = []
        if truncated:
            notes.append(f"（匹配超过 {LIMIT} 个，已按修改时间只显示最新 {LIMIT} 个）")
        if timed_out:
            notes.append("（搜索超时，结果不完整）")
        if notes:
            body += "\n" + " ".join(notes)
        return ToolOutput(body, {"type": "glob", "matches": len(files), "truncated": truncated})


def suggest_dir(workdir: Path, missing: Path) -> str | None:
    """ENOENT 时的 "Did you mean …?"：在父目录里找名字最接近的目录。"""
    try:
        candidates = [p.name for p in missing.parent.iterdir() if p.is_dir()]
    except OSError:
        return None
    if not candidates:
        return None
    match = difflib.get_close_matches(missing.name, candidates, n=1, cutoff=0.6)
    return match[0] if match else None
