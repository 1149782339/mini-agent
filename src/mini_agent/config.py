"""配置加载：环境变量 > .env > ~/.hermes/.env 回退。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _read_hermes_env(key: str) -> str | None:
    """从 ~/.hermes/.env 里取一个与模型相关的 key，作为回退。"""
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
    max_output_tokens: int = 32768
    temperature: float = 0.3
    workdir: Path = field(default_factory=lambda: Path.cwd())
    max_tool_output: int = 80000     # 单条工具结果字符上限。高于 Read 的 50000，避免读完又被掐成头尾
    max_steps: int = 40              # 单轮最多工具循环次数（子代理经 env 提到 150）
    keep_recent_tokens: int = 20000  # 压缩时尾部保留的 token 预算
    enable_subagent: bool = True    # 是否注册 Task 子代理工具（print 模式永不注册）
    enable_shell: bool = True        # 是否注册 run_command 工具
    shell: str = "auto"              # run_command 的 shell：auto / gitbash / cmd
    show_reasoning: bool = True      # 默认在终端打印思维链
    vision: bool | None = None       # None 按模型名判断；True/False 为 MINI_AGENT_VISION 强制

    @property
    def compact_threshold(self) -> int:
        return int(self.context_window * self.compact_ratio)

    @property
    def accepts_images(self) -> bool:
        """模型能不能看图。显式开关优先，否则看模型名。"""
        if self.vision is not None:
            return self.vision
        return model_accepts_images(self.model)

    def validate(self) -> None:
        if not self.api_key:
            raise SystemExit(
                "缺少 API key：请设置 MINI_AGENT_API_KEY，"
                "或在 ~/.hermes/.env 中提供 ARK_API_KEY。"
            )


# 名字里带这些片段的模型默认可看图。
# DeepSeek-V4.1-Flash（含经济版）另判：官方 deepseek-flash 原生多模态，
# V4-Pro / V3 / deepseek-chat 不看图。
_VISION_KEYS = (
    "-vl", "vl-", "vision", "gpt-4o", "gpt-4.1", "gpt-5",
    "gemini", "claude", "pixtral", "grok-2-vision", "grok-4",
)


def model_accepts_images(model: str) -> bool:
    name = (model or "").lower()
    if any(key in name for key in _VISION_KEYS):
        return True
    # 火山方舟写作 deepseek-v4.1-flash-economy，或 deepseek-v4-1-flash-economy-261017。
    return "deepseek" in name and "flash" in name


def _env_optional_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
        keep_recent_tokens=int(os.environ.get("MINI_AGENT_KEEP_RECENT_TOKENS", Config.keep_recent_tokens)),
        max_output_tokens=int(os.environ.get("MINI_AGENT_MAX_OUTPUT_TOKENS", Config.max_output_tokens)),
        max_steps=int(os.environ.get("MINI_AGENT_MAX_STEPS", Config.max_steps)),
        enable_shell=os.environ.get("MINI_AGENT_ENABLE_SHELL", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        enable_subagent=os.environ.get("MINI_AGENT_ENABLE_SUBAGENT", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        shell=os.environ.get("MINI_AGENT_SHELL", "auto").strip().lower() or "auto",
        show_reasoning=os.environ.get("MINI_AGENT_SHOW_REASONING", "1").strip().lower()
        in {"1", "true", "yes", "on"},
        vision=_env_optional_bool("MINI_AGENT_VISION"),
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
- 列目录用 Ls，不要用 run_command 跑 ls。按名字找文件用 Glob，按内容找用 Grep。
- 图片（jpg、png、gif、webp、bmp）用 Read 读取，不要用 run_command。
- 修改已有文件前先读它；优先用 edit 做精确替换，而不是整文件重写。
- 删除文件或目录用 RM，移动或重命名用 MV。路径必须是工作目录内的字面绝对路径
  （不要用相对路径或 ~/xxx）。删除目录必须显式 recursive=true；
  MV 不会覆盖已存在的目标。不要在 run_command 里直接执行 rm 或 mv。
- run_command 在 Windows 上跑 Git Bash（标准 bash 语法：$VAR、; 分隔、$(...)、
  管道/重定向齐全，grep/sed/awk/find 都可用），Linux/macOS 上是 sh。
  按 bash 语法写命令，不要混用 cmd 的 %VAR%。
- 下载、编译这类长时间命令用 run_command 的 background=true，立刻拿到 job_id；
  需要做完后自动继续时再加 notify_on_complete=true。不要同步死等。
  查看或取消用 process_manage（list / poll / log / wait / kill）。
  若已打开完成通知，不要反复 poll；结束后会自动叫醒你。
  前台命令超时在交互模式下会升级为后台并打开完成通知，不要当成已经失败。
- 写完代码后用 run_command 实际运行验证（python xxx.py / pytest / node / 编译），
  看到非零 exit code 或 stderr 就修，直到通过为止；不要声称"应该能跑"。
- 工具返回错误就调整参数重试，不要重复同样的调用。
- 回答简洁，中文优先。完成任务后给出改了什么、验证了什么。
- 不要臆造运行结果；没有真正执行就不要声称执行过。
- 预计要很多轮工具往返的任务（开放式搜索、独立模块开发/调试、批量分析）
  优先用 Task 委派子代理：把目标和所需上下文写进任务文本，子代理独立完成后
  返回结论，不占本会话上下文。一两步能做完的事直接自己做，不要委派。
"""
