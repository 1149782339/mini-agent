"""Agent 主循环：请求 → 工具调用 → 回填结果 → 再请求，直到没有工具调用。"""
from __future__ import annotations

from typing import Callable

from .client import ContextOverflowError, LLM, LLMError
from .config import Config
from .context import Context
from .tools import ToolRegistry

EventHook = Callable[[str, dict], None]


def _noop(event: str, payload: dict) -> None:
    return None


class Agent:
    def __init__(
        self,
        cfg: Config,
        registry: ToolRegistry,
        llm: LLM | None = None,
        on_event: EventHook | None = None,
        store=None,
        session_id: str | None = None,
        on_delta: Callable[[str, str], None] | None = None,
    ):
        self.cfg = cfg
        self.registry = registry
        self.llm = llm or LLM(cfg)
        self.on_event = on_event or _noop
        self.on_delta = on_delta
        self.ctx = Context(cfg)
        self.store = store
        self.session_id = session_id
        if store is not None and session_id and store.message_count(session_id) > 0:
            self.ctx.messages = store.load(session_id)

    # ---------- 历史写入（内存 + 磁盘同步）----------
    def _add(self, message: dict) -> None:
        self.ctx.add(message)
        if self.store is not None and self.session_id:
            self.store.append(self.session_id, message)

    def _sync_after_compaction(self) -> None:
        if self.store is not None and self.session_id:
            self.store.replace(self.session_id, self.ctx.messages)

    # ---------- 对外 ----------
    def run(self, user_input: str) -> str:
        text = (user_input or "").strip()
        if not text:
            return ""

        self._add({"role": "user", "content": text})

        for step in range(1, self.cfg.max_steps + 1):
            self._compact_if_needed()

            # 上下文溢出（Pi 的 overflow recovery）：先压缩再重试本轮，恰好一次
            overflow_retried = False
            while True:
                try:
                    response = self.llm.chat_stream(
                        self.ctx.messages,
                        self.registry.schemas(),
                        on_delta=self.on_delta,
                        on_notice=lambda m: self.on_event("notice", {"message": m}),
                    )
                    break
                except ContextOverflowError as exc:
                    if overflow_retried:
                        self.on_event("error", {"message": str(exc), "step": step})
                        return f"ERROR: {exc}"
                    overflow_retried = True
                    self.on_event(
                        "notice", {"message": "上下文超出模型窗口，先压缩再重试本轮"}
                    )
                    self.ctx.compact(self.llm, self.registry.schemas(), force=True)
                    self._sync_after_compaction()
                except LLMError as exc:
                    self.on_event("error", {"message": str(exc), "step": step})
                    return f"ERROR: {exc}"

            assistant = {
                k: v
                for k, v in response.items()
                if not k.startswith("_") or k in ("_usage", "_finish_reason")
            }  # _usage/_finish_reason 是私有字段，发给 API 前由 _wire_messages 剥掉
            self._add(assistant)
            self.on_event(
                "assistant",
                {
                    "step": step,
                    "content": assistant.get("content", ""),
                    "reasoning": response.get("reasoning_content", ""),
                },
            )

            calls = response.get("tool_calls") or []
            if not calls:
                return response.get("content") or "(模型返回空内容)"

            # Pi 的 failToolCallsFromTruncatedMessage：finish_reason=length 说明
            # 思维链吃光了输出预算、工具参数被截断（残缺 JSON 能解析但 silently
            # 不完整，执行了更糟）。不派发，直接回合成错误让模型缩小输出重试。
            truncated = response.get("_finish_reason") == "length"
            for call in calls:
                name = call.get("function", {}).get("name", "?")
                args = call.get("function", {}).get("arguments", "{}")
                self.on_event("tool_call", {"name": name, "arguments": args})
                if truncated:
                    result = (
                        f"ERROR: {name} 的参数被输出预算截断（finish_reason=length），"
                        "内容不完整，未执行。请缩小单次输出后重试：写大文件时分成多个"
                        " Write（每个 ≤100 行），或减少不必要的推理长度。"
                    )
                else:
                    result = self.registry.dispatch(name, args)
                self.on_event("tool_result", {"name": name, "result": result})
                self._add(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": result,
                    }
                )

        self.on_event("error", {"message": "达到最大工具步数"})
        return f"（达到最大工具步数 {self.cfg.max_steps}，已停止）"

    # ---------- 内部 ----------
    def _compact_if_needed(self) -> None:
        report = self.ctx.compact(self.llm, self.registry.schemas())
        if report is not None:
            self._sync_after_compaction()
            self.on_event("compaction", report.__dict__)

    def force_compact(self):
        self.ctx.force_fit(self.registry.schemas())
        self._sync_after_compaction()
        return self.ctx.size(self.registry.schemas())
