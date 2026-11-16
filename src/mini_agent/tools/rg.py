"""ripgrep 子进程封装：Glob/Grep 共同的底层。

- 退出码 1 = "无匹配"，转成空结果而不是错误——"无匹配"和"搜索出错/超时"
  必须区分，否则模型会把超时误判成没有匹配
- 超时若有部分输出：丢弃最后一行（可能不完整）后返回部分结果
- 只用命令名 rg，不解析绝对路径。list 形式传参不经 shell。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .base import ToolError, decode_output

RIPGREP_TIMEOUT = 30        # 秒
MAX_RG_STDOUT = 20 * 1024 * 1024  # stdout 上限


def rg_all_flag() -> list[str]:
    """MINI_AGENT_RG_ALL=1 时搜索包含隐藏文件、忽略 .gitignore。

    默认相反：rg 原生的"跳过隐藏文件 + 尊重 .gitignore"语义，这样 .git/
    node_modules/ 不会被搜出来。没有单独的路径排除规则，靠 rg 默认值兜底。
    """
    import os
    if os.environ.get("MINI_AGENT_RG_ALL", "").strip().lower() in {"1", "true", "yes"}:
        return ["--hidden", "--no-ignore"]
    return []


def run_rg(args: list[str], cwd: Path,
           timeout: int = RIPGREP_TIMEOUT) -> tuple[str, bool]:
    """运行 rg，返回 (解码后的 stdout, 是否超时截断)。

    退出码 1 → 无匹配 → ("", False)。其余非零退出 → ToolError。
    """
    try:
        proc = subprocess.run(
            ["rg", *args],
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise ToolError("系统没有 ripgrep（rg），请先安装并加入 PATH", 20) from None
    except subprocess.TimeoutExpired as exc:
        partial = decode_output(exc.stdout or b"")
        lines = partial.split("\n")
        if lines and lines[-1].strip() == "":
            lines.pop()
        elif lines:
            lines.pop()  # 最后一行可能不完整，丢弃
        return "\n".join(lines), True

    if proc.returncode == 1:
        return "", False
    if proc.returncode != 0:
        err = decode_output(proc.stderr).strip()
        raise ToolError(f"ripgrep 失败（exit {proc.returncode}）：{err[:300]}", 21)
    out = decode_output(proc.stdout)
    if len(out) > MAX_RG_STDOUT:
        out = out[:MAX_RG_STDOUT]
    return out, False
