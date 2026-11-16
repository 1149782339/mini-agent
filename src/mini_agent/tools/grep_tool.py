"""Grep 工具：ripgrep 的 content / files_with_matches / count。

- output_mode: content（默认带行号）/ files_with_matches / count
- context 参数（-B/-A/-C/context）仅 content 模式生效
- head_limit 默认 250 + offset 分页；仅实际截断时才提示翻页
- files_with_matches 按 mtime 降序（最新文件优先），stat 失败容忍
- --max-columns 500 防 minified/base64 行刷屏
- pattern 以 - 开头自动改用 -e 传递（防被当成命令行 flag）
- glob 参数支持空格/逗号分隔多个模式，保留 {a,b} 花括号
- multiline 显式开启（-U --multiline-dotall）
- 超时与"无匹配"区分；超时有部分结果则丢弃残缺尾行
- 输出路径相对工作目录
- 没有按路径的排除规则，默认靠 rg 原生忽略语义（隐藏文件/.gitignore），
  MINI_AGENT_RG_ALL=1 可反转
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .base import (
    Tool, ToolContext, ToolError, ToolOutput, ValidationResult,
    semantic_bool, semantic_int,
)
from .filestate import FileToolContext, is_unc, resolve_anywhere
from .glob_tool import suggest_dir
from .rg import run_rg, rg_all_flag

DEFAULT_HEAD_LIMIT = 250
MAX_COLUMNS = 500         # --max-columns


def _split_globs(raw: str) -> list[str]:
    """按空格/逗号分隔多个 glob，但不拆开 {a,b} 花括号组。"""
    parts: list[str] = []
    depth = 0
    current = ""
    for ch in raw:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if ch in " ," and depth == 0:
            if current.strip():
                parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


class GrepTool(Tool):
    name = "Grep"
    description = (
        "基于 ripgrep 的强大内容搜索。\n"
        "用法：\n"
        "- 搜索一律用本工具，不要在 run_command 里跑 grep/rg。\n"
        "- 完整正则：log.*Error、function\\s+\\w+。注意 rg 语法——字面花括号要转义"
        "（搜 Go 的 interface{} 要写 interface\\{\\}）。\n"
        "- glob 过滤文件（如 *.py、**/*.ts、*.{ts,tsx}），type 指定文件类型（py/js/rust）。\n"
        "- output_mode：content 显示匹配行（默认，带行号）；files_with_matches 只列文件；"
        "count 只计匹配数。\n"
        "- head_limit 默认 250，结果更多时用 offset 翻页。\n"
        "- 跨行匹配（如 struct \\{[\\s\\S]*?field）必须设 multiline: true。\n"
        "- 注意：CRLF 行尾的文件里 $ 匹配不到 \\r 之前（rg 行为），"
        "锚定行尾请写 \\r?$。\n"
        "- 开放式的多轮搜索请配合 run_command 缩小范围后再搜。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "ripgrep 正则表达式（在文件内容中搜索）",
            },
            "path": {
                "type": "string",
                "description": "搜索的文件或目录（相对路径基于工作目录，绝对路径任意位置均可），默认工作目录",
            },
            "glob": {
                "type": "string",
                "description": "按 glob 过滤文件（如 *.py、**/*.ts、*.{ts,tsx}），多个用空格或逗号分隔",
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": "content=匹配行（默认，带行号）；files_with_matches=只列文件；count=匹配计数",
            },
            "-B": {"type": "integer", "description": "匹配行前显示 N 行（content 模式）"},
            "-A": {"type": "integer", "description": "匹配行后显示 N 行（content 模式）"},
            "-C": {"type": "integer", "description": "前后各显示 N 行（content 模式）"},
            "context": {"type": "integer", "description": "-C 的别名"},
            "-n": {
                "type": "boolean",
                "description": "显示行号（content 模式，默认 true）",
            },
            "-i": {"type": "boolean", "description": "忽略大小写"},
            "type": {"type": "string", "description": "文件类型过滤（rg --type），如 py、js"},
            "head_limit": {
                "type": "integer",
                "description": "最多输出 N 条（所有模式通用），默认 250；0 表示不限",
            },
            "offset": {
                "type": "integer",
                "description": "跳过前 N 条再应用 head_limit（等价 offset+head_limit 翻页），默认 0",
            },
            "multiline": {
                "type": "boolean",
                "description": "跨行匹配：. 匹配换行，模式可跨行（rg -U --multiline-dotall），默认 false",
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
        if not isinstance(pattern, str) or not pattern:
            return ValidationResult.fail("pattern 必须是非空字符串", 1)
        raw = args.get("path")
        if raw is not None:
            if not isinstance(raw, str) or not raw.strip():
                return ValidationResult.fail("path 必须是非空字符串", 2)
            if is_unc(raw.strip()):
                return ValidationResult.fail(
                    f"不支持 UNC 路径 {raw!r}（防止 SMB/NTLM 凭据泄漏）", 3
                )
            try:
                resolve_anywhere(ctx.workdir, raw.strip())
            except ToolError as exc:
                return ValidationResult.fail(str(exc), error_code=exc.error_code)
        return ValidationResult.pass_()

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        if not isinstance(ctx, FileToolContext):
            raise ToolError("Grep 需要 FileToolContext", 0)
        pattern: str = args["pattern"]
        raw_path = args.get("path")
        target = resolve_anywhere(
            ctx.workdir, raw_path.strip() if isinstance(raw_path, str) else "."
        )
        if not target.exists():
            hint = suggest_dir(ctx.workdir, target)
            msg = f"路径不存在：{target}"
            if hint is not None:
                msg += f"。你是否想找 {hint}？"
            raise ToolError(msg, 4)

        mode = args.get("output_mode") or "files_with_matches"
        if mode not in {"content", "files_with_matches", "count"}:
            raise ToolError(
                f"未知 output_mode {mode!r}，可用：content / files_with_matches / count", 5
            )
        show_lines = semantic_bool(args.get("-n"), True)
        ignore_case = semantic_bool(args.get("-i"), False)
        multiline = semantic_bool(args.get("multiline"), False)
        head_limit = semantic_int(args.get("head_limit"), DEFAULT_HEAD_LIMIT)
        head_limit = DEFAULT_HEAD_LIMIT if head_limit is None else head_limit
        offset = semantic_int(args.get("offset"), 0) or 0
        offset = max(0, offset)

        rg_args: list[str] = [
            "--max-columns", str(MAX_COLUMNS),
            "--path-separator=/",  # Windows 默认输出 .\path，统一成 ./ 便于剥离
            *rg_all_flag(),
        ]
        if mode == "files_with_matches":
            rg_args.append("-l")
        elif mode == "count":
            rg_args.append("-c")
        elif show_lines:
            rg_args.append("-n")
        if ignore_case:
            rg_args.append("-i")
        if multiline:
            rg_args += ["-U", "--multiline-dotall"]
        file_type = args.get("type")
        if isinstance(file_type, str) and file_type.strip():
            rg_args += ["--type", file_type.strip()]
        glob_raw = args.get("glob")
        if isinstance(glob_raw, str) and glob_raw.strip():
            for g in _split_globs(glob_raw):
                rg_args += ["--glob", g]
        if mode == "content":
            context = semantic_int(args.get("context"), None)
            c = context if context is not None else semantic_int(args.get("-C"), None)
            b = semantic_int(args.get("-B"), None)
            a = semantic_int(args.get("-A"), None)
            if c is not None:
                rg_args += ["-C", str(max(0, c))]
            else:
                if b is not None:
                    rg_args += ["-B", str(max(0, b))]
                if a is not None:
                    rg_args += ["-A", str(max(0, a))]
        if pattern.startswith("-"):
            rg_args += ["-e", pattern]  # 防被当成命令行 flag
        else:
            rg_args.append(pattern)

        # 目录搜索必须显式给路径参数：rg 在"零路径"时做 stdin 启发式
        # （Windows 下子进程 stdin 是管道 → rg 去读 stdin 而不是搜索目录，
        # 表现为莫名其妙的 No matches）。传 "." 会引入 "./" 前缀，随后剥离。
        if target.is_dir():
            search_dir, subject = target, "."
        else:
            search_dir, subject = target.parent, target.name
        rg_args.append(subject)

        out, timed_out = run_rg(rg_args, cwd=search_dir)
        lines = [ln for ln in out.split("\n") if ln.strip()]
        if not lines:
            body = "No matches found"
            if timed_out:
                body += "（搜索超时，结果可能不完整；请缩小 path 或更精确的 pattern）"
            return ToolOutput(body, {"type": "grep", "matches": 0})

        # files_with_matches：按 mtime 降序（最新优先），stat 失败记 0 容忍竞态删除
        if mode == "files_with_matches":
            def _mtime(line: str) -> int:
                try:
                    return (search_dir / line.split(":")[0]).stat().st_mtime_ns
                except OSError:
                    return 0
            lines.sort(key=_mtime, reverse=True)

        # 剥离 "./" 前缀。content 模式的字段分隔符：匹配行是 "path:行号:..."，
        # 上下文行（-A/-B/-C，rg --field-context-separator 语义）是 "path-行号-..."。
        # 只剥这两种形态；files/count 模式整行都是路径，直接剥。
        stripped = []
        for ln in lines:
            if ln.startswith("./"):
                rest = ln[2:]
                if mode != "content" or re.match(r"^(.+?:\d+:|.+?-\d+-)", rest):
                    ln = rest
            stripped.append(ln)
        lines = stripped

        # 分页：先 offset 跳过，再 head_limit 截断；仅实际截断时提示翻页
        if offset:
            lines = lines[offset:]
        truncated = False
        if head_limit > 0 and len(lines) > head_limit:
            truncated = True
            lines = lines[:head_limit]

        prefix = os.path.relpath(search_dir, ctx.workdir)
        if prefix != ".":
            lines = [f"{prefix}/{ln}" for ln in lines]

        body = "\n".join(lines)
        notes = []
        if truncated:
            notes.append(
                f"[仅显示 {len(lines)} 条结果（head_limit={head_limit}）；"
                f"用 offset 参数翻页，或缩小搜索范围]"
            )
        if timed_out:
            notes.append("（搜索超时，已丢弃不完整的最后一行，结果为部分匹配）")
        if notes:
            body += "\n" + " ".join(notes)
        return ToolOutput(
            body, {"type": "grep", "matches": len(lines), "truncated": truncated}
        )
