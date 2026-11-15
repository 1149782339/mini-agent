"""Agent 主循环：请求 → 工具调用 → 回填结果 → 再请求，直到没有工具调用。"""
from __future__ import annotations

from typing import Callable

from .client import LLM, LLMError
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

            try:
                response = self.llm.chat_stream(
                    self.ctx.messages, self.registry.schemas(), on_delta=self.on_delta
                )
            except LLMError as exc:
                self.on_event("error", {"message": str(exc), "step": step})
                return f"ERROR: {exc}"

            assistant = {k: v for k, v in response.items() if not k.startswith("_")}
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

            for call in calls:
                name = call.get("function", {}).get("name", "?")
                args = call.get("function", {}).get("arguments", "{}")
                self.on_event("tool_call", {"name": name, "arguments": args})
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
