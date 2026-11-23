"""配置加载：CLI > 环境变量 > .env > 用户 config.toml > 默认值。

- `.env` 从当前目录向上查找，主要用来放密钥；已存在的环境变量优先于它。
- `config.toml` 默认在 `~/.config/mini-agent/config.toml`（Windows 为
  `%APPDATA%/mini-agent/config.toml`），存放不敏感的长期偏好。
- 密钥只从环境变量或 `.env` 读取，config.toml 里出现 `api_key` 直接报错。
- 配置读取不修改 `os.environ`，各来源合并成一次性的 Config。
"""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import dotenv_values

from .prompts import load_prompt

APP_DIRNAME = "mini-agent"
_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


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
                "缺少 API key：请设置 MINI_AGENT_API_KEY 环境变量，"
                f"或写入当前项目的 .env（如 {Path.cwd() / '.env'}）。"
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
    """用户配置文件：MINI_AGENT_CONFIG 优先，其次 XDG / APPDATA。"""
    override = os.environ.get("MINI_AGENT_CONFIG")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_DIRNAME / "config.toml"


def find_env_file() -> Path | None:
    """`.env`：MINI_AGENT_ENV_FILE 显式指定，否则从当前目录逐级向上找。"""
    override = os.environ.get("MINI_AGENT_ENV_FILE")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise SystemExit(f"MINI_AGENT_ENV_FILE 指向的文件不存在：{path}")
        return path
    for directory in (Path.cwd(), *Path.cwd().parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


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
    if "api_key" in data:
        raise SystemExit(f"配置文件不保存 api_key：{path}，请改用环境变量或 .env")
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
    env_path = find_env_file()
    dotenv_data = dotenv_values(env_path) if env_path else {}
    explicit_toml = config_file is not None or bool(os.environ.get("MINI_AGENT_CONFIG"))
    toml_path = Path(config_file).expanduser() if config_file else user_config_path()
    toml_data = _read_toml(toml_path, required=explicit_toml)

    allowed = {spec.name for spec in fields(Config)} - {"api_key"}
    unknown = sorted(set(toml_data) - allowed)
    if unknown:
        raise SystemExit(f"配置文件 {toml_path} 含未知键：{', '.join(unknown)}")

    defaults = Config()
    values: dict[str, object] = {}
    for spec in fields(Config):
        if spec.name == "api_key":
            continue
        env_name = f"MINI_AGENT_{spec.name.upper()}"
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            raw = dotenv_data.get(env_name)
        if raw is None or raw == "":
            raw = toml_data.get(spec.name)
        if raw is None or raw == "":
            continue
        values[spec.name] = _coerce(spec.name, raw, getattr(defaults, spec.name))

    api_key = (
        os.environ.get("MINI_AGENT_API_KEY")
        or dotenv_data.get("MINI_AGENT_API_KEY")
        or ""
    ).strip()
    return Config(api_key=api_key, **values)


SYSTEM_PROMPT = load_prompt("system.md")
