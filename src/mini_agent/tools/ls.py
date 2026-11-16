"""Ls 工具：列一层目录。

Glob 按名字找文件，Grep 按内容找。看一个目录里有什么用这个，不必再跑 ls。
只读，路径规则与 Read/Glob/Grep 相同：相对路径基于工作目录，绝对路径任意位置，
UNC 在任何文件系统调用之前拒绝。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolError, ToolOutput, ValidationResult, semantic_bool
from .filestate import FileToolContext, is_unc, resolve_anywhere
from .glob_tool import suggest_dir

LIMIT = 200


class LsTool(Tool):
    name = "Ls"
    description = (
        "列出一个目录里的直接内容（文件和子目录），只列一层。\n"
        "- 要按名字找文件用 Glob，按内容找用 Grep，读文件用 Read。\n"
        "- 目录名以 / 结尾，符号链接以 @ 结尾。目录在前，其余按名字排序。\n"
        "- 默认跳过以 . 开头的隐藏项；all=true 时包含它们。\n"
        "- 最多 200 项。path 相对工作目录，绝对路径任意位置均可，省略时列工作目录。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要列出的目录（相对路径基于工作目录，绝对路径任意位置均可），默认工作目录",
            },
            "all": {
                "type": "boolean",
                "description": "包含以 . 开头的隐藏项（默认 false）",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def validate_input(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        raw = args.get("path")
        if raw is not None:
            if not isinstance(raw, str) or not raw.strip():
                return ValidationResult.fail("path 必须是非空字符串", 1)
            if is_unc(raw.strip()):
                return ValidationResult.fail(
                    f"不支持 UNC 路径 {raw!r}（防止 SMB/NTLM 凭据泄漏）", 2
                )
            try:
                resolve_anywhere(ctx.workdir, raw.strip())
            except ToolError as exc:
                return ValidationResult.fail(str(exc), error_code=exc.error_code)
        return ValidationResult.pass_()

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        if not isinstance(ctx, FileToolContext):
            raise ToolError("Ls 需要 FileToolContext", 0)
        raw = args.get("path")
        target = resolve_anywhere(
            ctx.workdir, raw.strip() if isinstance(raw, str) else "."
        )
        show_all = semantic_bool(args.get("all"), False)

        if not target.exists():
            hint = suggest_dir(ctx.workdir, target)
            msg = f"目录不存在：{target}"
            if hint is not None:
                msg += f"。你是否想找 {hint}？"
            raise ToolError(msg, 3)
        if not target.is_dir():
            raise ToolError(f"这是文件不是目录：{target}。读文件内容请用 Read", 4)

        try:
            entries = list(target.iterdir())
        except OSError as exc:
            raise ToolError(f"无法列出 {target}：{exc}", 5) from None

        skipped = 0
        visible: list[Path] = []
        for entry in entries:
            if not show_all and entry.name.startswith("."):
                skipped += 1
                continue
            visible.append(entry)

        def sort_key(entry: Path) -> tuple[int, str]:
            return (0 if _is_dir(entry) else 1, entry.name.casefold())

        visible.sort(key=sort_key)
        truncated = len(visible) > LIMIT
        shown = visible[:LIMIT]
        lines = [_label(entry) for entry in shown]

        header = _display_root(target, ctx.workdir)
        if lines:
            body = header + "\n" + "\n".join(lines)
        else:
            body = header + "\n（空目录）"
        notes = []
        if skipped:
            notes.append(f"（已跳过 {skipped} 个隐藏项，all=true 可显示）")
        if truncated:
            notes.append(f"（超过 {LIMIT} 项，只显示前 {LIMIT} 项；请进入子目录再列）")
        if notes:
            body += "\n" + " ".join(notes)
        return ToolOutput(
            body,
            {
                "type": "ls",
                "path": str(target),
                "entries": len(shown),
                "truncated": truncated,
                "skipped_hidden": skipped,
            },
        )


def _is_dir(entry: Path) -> bool:
    try:
        return entry.is_dir() and not entry.is_symlink()
    except OSError:
        return False


def _label(entry: Path) -> str:
    try:
        if entry.is_symlink():
            return entry.name + "@"
        if entry.is_dir():
            return entry.name + "/"
    except OSError:
        pass
    return entry.name


def _display_root(target: Path, workdir: Path) -> str:
    """工作目录内用相对路径，外面用绝对路径。根用 `.`。"""
    try:
        rel = target.resolve().relative_to(workdir.resolve())
    except ValueError:
        return str(target)
    text = rel.as_posix()
    return "." if text == "." else text
