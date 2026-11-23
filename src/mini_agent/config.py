"""配置加载：CLI > 环境变量 > 用户 config.toml > 默认值。

- `config.toml` 默认在 `~/.config/mini-agent/config.toml`，包含 API key、
  模型和其他长期偏好。
- 环境变量可覆盖 TOML 中的配置，便于临时运行和 Task 子代理传递生效配置。
- 配置读取不修改 `os.environ`，各来源合并成一次性的 Config。
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .prompts import load_prompt

APP_DIRNAME = "mini-agent"
_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


@dataclass
class Config:
    api_key: str = ""
    base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3"
    model: str = "deepseek-v4.1-flash-economy"
    context_window: int = 1000000
    compact_ratio: float = 0.7
    max_output_tokens: int = 32768
    temperature: float = 0.3
    workdir: Path = field(default_factory=lambda: Path.cwd())
    max_tool_output: int = 80000     # 单条工具结果字符上限。高于 Read 的 50000，避免读完又被掐成头尾
    max_steps: int = 40              # 单轮最多工具循环次数（子代理经 env 提到 150）
    foreground_block_budget: float = 15.0  # 前台命令最多阻塞几秒，到点转后台继续跑
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
                f"缺少 API key：请在 {user_config_path()} 中设置 api_key，"
                "或使用 MINI_AGENT_API_KEY 环境变量覆盖。"
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


def user_config_path() -> Path:
    """用户配置文件：MINI_AGENT_CONFIG 优先，否则统一使用 ~/.config。"""
    override = os.environ.get("MINI_AGENT_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / APP_DIRNAME / "config.toml"


def _read_toml(path: Path, *, required: bool) -> dict:
    if not path.is_file():
        if required:
            raise SystemExit(f"配置文件不存在：{path}")
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"配置文件解析失败 {path}：{exc}") from exc
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            raise SystemExit(f"配置文件 {path} 的 {key} 必须是单个值，不支持分组或数组")
    return data


def _to_bool(name: str, raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _BOOL_TRUE:
        return True
    if text in _BOOL_FALSE:
        return False
    raise SystemExit(f"配置项 {name} 需要布尔值（true/false），收到 {raw!r}")


def _coerce(name: str, raw: object, default: object) -> object:
    if name == "vision":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if not text or text == "auto":
            return None
        return _to_bool(name, text)
    if isinstance(default, bool):
        return _to_bool(name, raw)
    if isinstance(default, int):
        try:
            return int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise SystemExit(f"配置项 {name} 需要整数，收到 {raw!r}") from None
    if isinstance(default, float):
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise SystemExit(f"配置项 {name} 需要数字，收到 {raw!r}") from None
    if isinstance(default, Path):
        return Path(str(raw)).expanduser().resolve()
    return str(raw).strip()


def load_config(config_file: str | Path | None = None) -> Config:
    explicit_toml = config_file is not None or bool(os.environ.get("MINI_AGENT_CONFIG"))
    toml_path = Path(config_file).expanduser() if config_file else user_config_path()
    toml_data = _read_toml(toml_path, required=explicit_toml)

    allowed = {spec.name for spec in fields(Config)}
    unknown = sorted(set(toml_data) - allowed)
    if unknown:
        raise SystemExit(f"配置文件 {toml_path} 含未知键：{', '.join(unknown)}")

    defaults = Config()
    values: dict[str, object] = {}
    for spec in fields(Config):
        env_name = f"MINI_AGENT_{spec.name.upper()}"
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            raw = toml_data.get(spec.name)
        if raw is None or raw == "":
            continue
        values[spec.name] = _coerce(spec.name, raw, getattr(defaults, spec.name))

    return Config(**values)


SYSTEM_PROMPT = load_prompt("system.md")
