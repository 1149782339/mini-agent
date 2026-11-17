"""防空转护栏（瘦身版，思路来自 Hermes 的 tool_guardrails）。

控制器除自身计数外没有副作用，只返回决策；由主循环决定是改写成短引用、
追加提示、返回合成错误，还是结束本轮。每个 Agent.run()（每一轮用户输入）
开始时重置一次。状态只存在于控制器内存，不写入 SQLite 会话原文。

三层里只做前两层：

1. 相同调用盯梢。连续相同的「工具名 + 规范化参数 + 结果」：第 2 次且结果
   超过 512 字符时，工具仍执行，但写入上下文的改成指向首次结果的短引用；
   第 3 次追加提示；硬停开启时第 5 次在写回结果后结束本轮，不再请求模型。
   另外检测周期 2 到 4 的循环（例如 Read A、Grep B 交替），阈值相同。
   轮询类工具（名字在 REPEATABLE_TOOLS，或以 _poll 结尾）豁免提示和硬停，
   短引用照做。process_manage 属于这一类。
2. 失败次数。同一工具同一参数失败：第 2 次警告；硬停开启时第 5 次在
   before_call 拦截，不再真正执行。同一工具换参数反复失败：第 3 次警告，
   硬停开启时第 8 次结束本轮；run_command 走这条不计硬停（测试/编译失败
   是正常诊断），但相同参数的拦截仍适用。Edit / Write / run_command / Task
   / RM / MV 成功视为有进展，清空失败计数。
3. 每轮上限（搜索、子代理等）不做。

硬停策略：交互式 CLI 默认只警告、不硬停；-p / 脚本模式和 Task 子代理默认开。
本护栏与 max_steps、溢出压缩重试并存，不取代它们。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

# 阈值与 Hermes 默认对齐
IDENTICAL_HINT_AFTER = 3
IDENTICAL_HALT_AFTER = 5
IDENTICAL_STUB_MIN_CHARS = 512
EXACT_FAILURE_WARN_AFTER = 2
EXACT_FAILURE_BLOCK_AFTER = 5
SAME_TOOL_FAILURE_WARN_AFTER = 3
SAME_TOOL_FAILURE_HALT_AFTER = 8
MAX_CYCLE_PERIOD = 4
CYCLE_HISTORY = 64
_STUB_ARGS_PREVIEW = 120

READ_ONLY_TOOLS = frozenset({"Read", "Ls", "Glob", "Grep"})
PROGRESS_TOOLS = frozenset({"Edit", "Write", "run_command", "Task", "RM", "MV"})
FAILURE_TOLERANT_TOOLS = frozenset({"run_command"})
# 轮询类工具挂钩：日后可把名字加进集合，或使用 _poll 后缀。
REPEATABLE_TOOLS: frozenset[str] = frozenset({"process_manage"})
REPEATABLE_SUFFIXES: tuple[str, ...] = ("_poll",)

_EXIT_RE = re.compile(r"^exit=(-?\d+)")


def is_repeatable(tool_name: str) -> bool:
    return tool_name in REPEATABLE_TOOLS or tool_name.endswith(REPEATABLE_SUFFIXES)


def canonical_args(args: Mapping[str, Any] | None) -> str:
    return json.dumps(args or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def parse_tool_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return val if isinstance(val, dict) else {"_raw": raw}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def signature_key(tool_name: str, args: Mapping[str, Any] | None) -> str:
    return f"{tool_name}:{_sha256(canonical_args(args))}"


def is_tool_failure(tool_name: str, result: str) -> bool:
    """ERROR: 前缀算失败；run_command 的非零 exit / 超时也算失败。"""
    if (result or "").startswith("ERROR:"):
        return True
    if tool_name == "run_command":
        first = (result or "").splitlines()[0] if result else ""
        m = _EXIT_RE.match(first)
        if m and int(m.group(1)) != 0:
            return True
        if "[TIMEOUT" in first:
            return True
    return False


def _is_guardrail_block(result: str) -> bool:
    return (result or "").startswith("ERROR: 已拦截")


@dataclass(frozen=True)
class GuardDecision:
    action: str = "allow"  # allow | warn | block | halt
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    period: int = 0

    @property
    def allows_execution(self) -> bool:
        return self.action in {"allow", "warn"}

    @property
    def should_halt(self) -> bool:
        return self.action in {"block", "halt"}

    @property
    def synthetic_error(self) -> str:
        return self.message if self.message.startswith("ERROR:") else f"ERROR: {self.message}"


@dataclass(frozen=True)
class CallObservation:
    notice: str | None = None
    stub: str | None = None
    kind: str | None = None
    count: int = 0


class StallGuard:
    """单轮用户输入内的防空转控制器。"""

    def __init__(self, *, hard_stop_enabled: bool = False):
        self.hard_stop_enabled = hard_stop_enabled
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._exact_failures: dict[str, int] = {}
            self._same_tool_failures: dict[str, int] = {}
            self._halt: GuardDecision | None = None
            self._streak_key: str | None = None
            self._streak_count = 0
            self._history: deque[tuple[str, str, bool]] = deque(maxlen=CYCLE_HISTORY)

    def halt_decision(self) -> GuardDecision | None:
        return self._halt

    def should_halt(self) -> bool:
        return self._halt is not None and self._halt.should_halt

    def halt_message(self) -> str:
        d = self._halt
        if d is None:
            return ""
        tool = d.tool_name or "同一个工具"
        n = d.count
        return (
            f"我停了，因为同一个工具（{tool}）重复了 {n} 次还没进展。"
            "上面是最后一次结果。请告诉我怎么继续，或者发「继续」，我会换个路子。"
        )

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> GuardDecision:
        key = signature_key(tool_name, args)
        with self._lock:
            exact = self._exact_failures.get(key, 0)
            # 第 N 次相同失败调用在执行前拦截（N = EXACT_FAILURE_BLOCK_AFTER）
            if self.hard_stop_enabled and exact + 1 >= EXACT_FAILURE_BLOCK_AFTER:
                count = exact + 1
                decision = GuardDecision(
                    "block",
                    "repeated_exact_failure_block",
                    f"ERROR: 已拦截 {tool_name}：相同参数已经失败 {exact} 次，"
                    f"第 {count} 次未执行。请不要原样重试，换策略或说明卡点。",
                    tool_name,
                    count,
                )
                self._halt = decision
                return decision
            return GuardDecision(tool_name=tool_name)

    def after_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str,
        *,
        failed: bool | None = None,
    ) -> GuardDecision:
        key = signature_key(tool_name, args)
        text = result or ""
        if failed is None:
            failed = is_tool_failure(tool_name, text)
        # 自己的拦截结果不再累加失败计数
        if _is_guardrail_block(text):
            return GuardDecision(tool_name=tool_name)

        with self._lock:
            if failed:
                exact = self._exact_failures[key] = self._exact_failures.get(key, 0) + 1
                same = self._same_tool_failures[tool_name] = (
                    self._same_tool_failures.get(tool_name, 0) + 1
                )
                if (
                    self.hard_stop_enabled
                    and tool_name not in FAILURE_TOLERANT_TOOLS
                    and same >= SAME_TOOL_FAILURE_HALT_AFTER
                ):
                    return self._set_halt(
                        "halt",
                        "same_tool_failure_halt",
                        f"[提示：{tool_name} 本轮已经失败 {same} 次。请换一种做法，而不是继续用这个工具硬试。]",
                        tool_name,
                        same,
                    )
                if exact >= EXACT_FAILURE_WARN_AFTER:
                    return GuardDecision(
                        "warn",
                        "repeated_exact_failure_warning",
                        f"[提示：{tool_name} 用相同参数已经失败 {exact} 次。"
                        "这很像在原地打转，请先看错误再换策略，不要原样重试。]",
                        tool_name,
                        exact,
                    )
                if same >= SAME_TOOL_FAILURE_WARN_AFTER:
                    return GuardDecision(
                        "warn",
                        "same_tool_failure_warning",
                        f"[提示：{tool_name} 本轮已经失败 {same} 次（参数不同）。"
                        "请换一种做法，而不是继续用这个工具硬试。]",
                        tool_name,
                        same,
                    )
                return GuardDecision(tool_name=tool_name, count=exact)

            self._exact_failures.pop(key, None)
            self._same_tool_failures.pop(tool_name, None)
            if tool_name in PROGRESS_TOOLS:
                self._exact_failures.clear()
                self._same_tool_failures.clear()
            return GuardDecision(tool_name=tool_name)

    def observe_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str,
        *,
        failed: bool = False,
        tool_call_id: str = "",
    ) -> CallObservation:
        del tool_call_id  # 预留：短引用将来可指向首次 tool_call_id
        key = signature_key(tool_name, args)
        text = result if isinstance(result, str) else ""
        result_hash = _sha256(text) if isinstance(result, str) else ""
        repeatable = is_repeatable(tool_name)
        item_key = f"{key}:{result_hash}"

        with self._lock:
            if isinstance(result, str) and item_key == self._streak_key:
                self._streak_count += 1
            else:
                self._streak_key = item_key if isinstance(result, str) else None
                self._streak_count = 1 if isinstance(result, str) else 0

            count = self._streak_count
            notice = kind = None
            if isinstance(result, str) and not repeatable and count >= IDENTICAL_HINT_AFTER:
                notice = (
                    f"[提示：这是连续第 {count} 次用相同参数调用 {tool_name}，且结果也相同。"
                    "不要再重复，请改参数、换工具，或基于已有结果继续。]"
                )
                kind = "identical_call_streak"
                if self.hard_stop_enabled and count >= IDENTICAL_HALT_AFTER and self._halt is None:
                    self._set_halt(
                        "halt",
                        "identical_call_streak_halt",
                        notice,
                        tool_name,
                        count,
                    )

            if isinstance(result, str):
                self._history.append((key, result_hash, repeatable))
            else:
                self._history.clear()

            if notice is None and isinstance(result, str):
                cycle = self._detect_cycle()
                if cycle is not None:
                    period, laps = cycle
                    notice = (
                        f"[提示：最近 {laps} 轮在重复同一组 {period} 个工具调用"
                        f"（以 {tool_name} 结束），参数和结果都相同。"
                        "不要再重复这组调用，请改参数、换工具，或基于已有结果继续。]"
                    )
                    kind = "identical_cycle"
                    if self.hard_stop_enabled and laps >= IDENTICAL_HALT_AFTER and self._halt is None:
                        self._set_halt(
                            "halt",
                            "identical_cycle_halt",
                            notice,
                            tool_name,
                            laps,
                            period=period,
                        )

            stub = None
            if (
                isinstance(result, str)
                and count >= 2
                and not failed
                and len(text) >= IDENTICAL_STUB_MIN_CHARS
            ):
                preview = canonical_args(args)
                if len(preview) > _STUB_ARGS_PREVIEW:
                    preview = preview[:_STUB_ARGS_PREVIEW] + "…"
                stub = (
                    f"[重复结果：与本轮先前 {tool_name} 的结果完全相同，"
                    f"请直接参考那次结果，内容没有变化。参数：{preview}]"
                )
            return CallObservation(notice=notice, stub=stub, kind=kind, count=count)

    def _set_halt(
        self,
        action: str,
        code: str,
        message: str,
        tool_name: str,
        count: int,
        period: int = 0,
    ) -> GuardDecision:
        decision = GuardDecision(action, code, message, tool_name, count, period)
        self._halt = decision
        return decision

    def _detect_cycle(self) -> tuple[int, int] | None:
        history = self._history
        for period in range(2, MAX_CYCLE_PERIOD + 1):
            if len(history) < period * IDENTICAL_HINT_AFTER:
                continue
            laps = 1
            for base in range(len(history) - 2 * period, -1, -period):
                if not all(
                    history[base + i][:2] == history[len(history) - period + i][:2]
                    for i in range(period)
                ):
                    break
                laps += 1
            if laps >= IDENTICAL_HINT_AFTER:
                tail = [history[len(history) - period + i] for i in range(period)]
                if all(repeatable for _, _, repeatable in tail):
                    continue
                return period, laps
        return None
