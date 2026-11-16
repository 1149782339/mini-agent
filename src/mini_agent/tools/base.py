"""工具框架：四段生命周期 + 注册表。

每个工具分四段：

1. validate_input(args, ctx)   纯输入校验，不做 I/O；失败信息直接反馈给模型
2. check_permissions(args, ctx) validate 通过后才调用；暂无审批层，
   默认 allow，留作扩展点
3. call(args, ctx)             真正执行，返回 ToolOutput（结构化输出）
4. render(output, args)        把结构化输出序列化成发给模型的字符串

is_read_only / is_concurrency_safe 默认 False。权限是 allow/deny 字符串，
路径约束在 workdir 上。
"""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..store import project_data_dir


def decode_output(raw: bytes | str | None) -> str:
    """子进程输出统一按 UTF-8 解码（项目约定：文件、数据库、子进程输出全是 UTF-8）。

    不用 subprocess 的 text=True：它按 locale 编码（这台机器 cp936）解码，
    rg / node / python 输出 UTF-8 时会抛 UnicodeDecodeError，stdout 变成 None，
    后续 .strip() 就崩。非法字节一律替换成 �，不抛异常、不猜 GBK。
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="replace")


# ---------- 模型参数容错 ----------
# 模型偶尔把布尔/数字写成字符串："replace_all":"false"。不能用 bool("false")，
# 非空字符串恒为 True。

def semantic_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() == "true":
            return True
        if value.strip().lower() == "false":
            return False
    return default


def semantic_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):  # True/False 不是有意义的整数
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


# ---------- 四段生命周期 ----------

@dataclass
class ValidationResult:
    ok: bool
    message: str = ""
    error_code: int | None = None  # 每个失败分支一个编号，便于定位

    @classmethod
    def pass_(cls) -> "ValidationResult":
        return cls(ok=True)

    @classmethod
    def fail(cls, message: str, error_code: int | None = None) -> "ValidationResult":
        return cls(ok=False, message=message, error_code=error_code)


@dataclass
class ToolOutput:
    """call() 的结构化产物；render() 把它变成发给模型的字符串。

    images 是 Read 读到的图片（base64 + mime）。文本走 content，图片另存，
    发请求时再决定是附上还是换成占位说明。
    """

    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    images: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ExecutedTool:
    """一次派发的结果：给终端和 tool 消息的文本，以及可选的图片块。"""

    text: str
    images: list[dict[str, str]] = field(default_factory=list)


class ToolError(Exception):
    """工具执行期错误（I/O 失败、状态冲突等），消息直接回给模型。"""

    def __init__(self, message: str, error_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code


class ToolContext:
    """工具可见的会话级上下文；文件工具用子类 FileToolContext 扩展。

    event_hook：工具执行期间向终端上报进度/事件的回调
    Callable[[kind: str, payload: dict], None]，由 Agent 注入。
    长耗时工具（如 Task 子代理）用它上报实时进度；没有 hook 时静默，
    测试与无 TTY 场景零影响。
    """

    def __init__(self, workdir: Path):
        self.workdir = workdir.resolve()
        self.event_hook: Callable[[str, dict], None] | None = None


class Tool(ABC):
    """工具基类。name/description/parameters 是 OpenAI function schema 的三要素。"""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    def is_read_only(self) -> bool:
        return False  # fail-closed（TOOL_DEFAULTS）

    def is_concurrency_safe(self) -> bool:
        return False  # fail-closed（TOOL_DEFAULTS）

    def validate_input(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        return ValidationResult.pass_()

    def check_permissions(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """返回 "allow" | "deny"（deny 可附原因，用 ":" 分隔）。暂无审批层。"""
        return "allow"

    @abstractmethod
    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        ...

    def render(self, output: ToolOutput, args: dict[str, Any]) -> str:
        return output.text

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FnTool(Tool):
    """简单函数工具的适配器（run_command 这类无状态工具）。

    fn 返回字符串；validate 只按 schema 的 required 列表查缺参。
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., str],
        read_only: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn
        self._read_only = read_only

    def is_read_only(self) -> bool:
        return self._read_only

    def validate_input(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        required = self.parameters.get("required") or []
        missing = [k for k in required if args.get(k) is None]
        if missing:
            return ValidationResult.fail(
                f"缺少必填参数：{', '.join(missing)}", error_code=1
            )
        return ValidationResult.pass_()

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return ToolOutput(text=self.fn(**args))


@dataclass
class ToolRegistry:
    max_output: int = 80000
    spill_dir: Path = field(default_factory=lambda: project_data_dir() / "spill")
    context: ToolContext | None = None
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def names(self) -> list[str]:
        return list(self.tools)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output:
            return text
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        # 并行派发下同秒可能多个工具同时落盘，文件名带线程 id 防撞
        spill = self.spill_dir / (
            f"{time.strftime('%Y%m%d_%H%M%S')}_{threading.get_ident()}_{len(text)}.txt"
        )
        spill.write_text(text, encoding="utf-8")
        head = text[: self.max_output // 2]
        tail = text[-self.max_output // 4 :]
        omitted = len(text) - len(head) - len(tail)
        return (
            f"{head}\n\n[... 已省略 {omitted} 字符，完整内容见 {spill} ...]\n\n{tail}"
        )

    def dispatch(self, name: str, arguments: str | dict) -> str:
        """执行工具并返回给模型看的文本。测试和只关心文本的调用用这个。"""
        return self.execute(name, arguments).text

    def execute(self, name: str, arguments: str | dict) -> ExecutedTool:
        tool = self.tools.get(name)
        if tool is None:
            return ExecutedTool(f"ERROR: 未知工具 {name!r}，可用：{', '.join(self.names())}")

        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return ExecutedTool(f"ERROR: 参数不是合法 JSON：{exc}")
        else:
            args = arguments or {}

        if not isinstance(args, dict):
            return ExecutedTool("ERROR: 参数必须是 JSON 对象")

        ctx = self.context or ToolContext(Path.cwd())

        # 第 1 段：纯校验（无 I/O 承诺由工具自身遵守）
        checked = tool.validate_input(args, ctx)
        if not checked.ok:
            code = f" (errorCode {checked.error_code})" if checked.error_code else ""
            return ExecutedTool(f"ERROR: {checked.message}{code}")

        # 第 2 段：权限（默认 allow；deny 可直接带原因）
        decision = tool.check_permissions(args, ctx)
        if decision != "allow":
            _, _, reason = decision.partition(":")
            return ExecutedTool(f"ERROR: 权限拒绝：{reason.strip() or name}")

        # 第 3、4 段：执行 + 序列化；异常统一转成 ERROR 字符串，不让主循环崩
        try:
            output = tool.call(args, ctx)
            return ExecutedTool(self._truncate(tool.render(output, args)), list(output.images))
        except ToolError as exc:
            code = f" (errorCode {exc.error_code})" if exc.error_code else ""
            return ExecutedTool(f"ERROR: {exc}{code}")
        except TypeError as exc:
            return ExecutedTool(f"ERROR: 参数不匹配 {name}：{exc}")
        except Exception as exc:
            return ExecutedTool(f"ERROR: {name} 执行失败：{type(exc).__name__}: {exc}")
