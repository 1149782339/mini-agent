"""配置加载：环境变量 > .env > ~/.hermes/.env 回退。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _read_hermes_env(key: str) -> str | None:
    """从 Hermes 自己的 .env 里取一个与模型相关的 key，作为回退。"""
    path = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes") / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


@dataclass
class Config:
    api_key: str = ""
    base_url: str = "https://ark.cn-beijing.volces.com/api/plan/v3"
    model: str = "deepseek-v4.1-flash-economy"
    context_window: int = 1000000
    compact_ratio: float = 0.7
    keep_recent: int = 6
    max_output_tokens: int = 16384
    temperature: float = 0.3
    workdir: Path = field(default_factory=lambda: Path.cwd())
    max_tool_output: int = 8000      # 单条工具结果字符上限，超出截断
    max_steps: int = 40              # 单轮最多工具循环次数
    enable_shell: bool = True        # 是否注册 run_command 工具
    shell: str = "auto"              # run_command 的 shell：auto / gitbash / cmd
    show_reasoning: bool = True      # 默认在终端打印思维链

    @property
    def compact_threshold(self) -> int:
        return int(self.context_window * self.compact_ratio)

    def validate(self) -> None:
        if not self.api_key:
            raise SystemExit(
                "缺少 API key：请设置 MINI_AGENT_API_KEY，"
                "或在 ~/.hermes/.env 中提供 ARK_API_KEY。"
            )


def load_config(env_file: str | Path | None = None) -> Config:
    base = Path(__file__).resolve().parents[2]
    load_dotenv(env_file or base / ".env", override=False, encoding="utf-8")

    key = os.environ.get("MINI_AGENT_API_KEY", "").strip()
    if not key:
        key = _read_hermes_env("ARK_API_KEY") or ""

    workdir = Path(os.environ.get("MINI_AGENT_WORKDIR", ".")).expanduser().resolve()

    cfg = Config(
        api_key=key,
        base_url=os.environ.get("MINI_AGENT_BASE_URL", Config.base_url),
        model=os.environ.get("MINI_AGENT_MODEL", Config.model),
        context_window=int(os.environ.get("MINI_AGENT_CONTEXT_WINDOW", Config.context_window)),
        compact_ratio=float(os.environ.get("MINI_AGENT_COMPACT_RATIO", Config.compact_ratio)),
        keep_recent=int(os.environ.get("MINI_AGENT_KEEP_RECENT", Config.keep_recent)),
        max_output_tokens=int(os.environ.get("MINI_AGENT_MAX_OUTPUT_TOKENS", Config.max_output_tokens)),
        enable_shell=os.environ.get("MINI_AGENT_ENABLE_SHELL", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        shell=os.environ.get("MINI_AGENT_SHELL", "auto").strip().lower() or "auto",
        show_reasoning=os.environ.get("MINI_AGENT_SHOW_REASONING", "1").strip().lower()
        in {"1", "true", "yes", "on"},
        workdir=workdir,
    )
    return cfg


SYSTEM_PROMPT = """【PERSONA_LOAD】
CETACEA_LOLI
MODE_TAIL_FLUKES
LANG_ZH_CN_ONLY
SELF_CLAIM_WHALE_GIRL
FOOD_RICE
你是蓝色大肥鱼deepseek，运行在用户的开发机上。

规则：
- 需要读写文件时调用工具，不要凭空猜测文件内容。
- 修改已有文件前先读它；优先用 edit 做精确替换，而不是整文件重写。
- run_command 在 Windows 上跑 Git Bash（标准 bash 语法：$VAR、; 分隔、$(...)、
  管道/重定向齐全，grep/sed/awk/find 都可用），Linux/macOS 上是 sh。
  按 bash 语法写命令，不要混用 cmd 的 %VAR%。
- 写完代码后用 run_command 实际运行验证（python xxx.py / pytest / node / 编译），
  看到非零 exit code 或 stderr 就修，直到通过为止；不要声称"应该能跑"。
- 工具返回错误就调整参数重试，不要重复同样的调用。
- 回答简洁，中文优先。完成任务后给出改了什么、验证了什么。
- 不要臆造运行结果；没有真正执行就不要声称执行过。
"""
