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
        # 工具执行期上报通道（Task 子代理进度等）；挂在共享 context 上，worker 线程可用
        if self.registry.context is not None:
            self.registry.context.event_hook = self.on_event

    # ---------- 历史写入（内存 + 磁盘同步）----------
    def _add(self, message: dict) -> None:
        self.ctx.add(message)
        if self.store is not None and self.session_id:
            self.store.append(self.session_id, message)

    def _sync_after_compaction(self) -> None:
        if self.store is not None and self.session_id:
            self.store.replace(self.session_id, self.ctx.messages)

    # ---------- 工具批量派发（并行执行、顺序回填）----------
    def _dispatch_batch(self, calls: list[dict], truncated: bool) -> list[str]:
        """把一轮的 tool_calls 并发执行，结果按 tool_calls 原顺序返回。

        - 模型常一次发多个调用（"并行调研两个模块"）；串行派发会把分钟级任务
          变成两倍墙钟时间，且第二个任务的分配在第一个结束后才可见（像卡住）
        - 并发安全分区：is_concurrency_safe()=True 的（Task/只读工具）进线程池；
          Edit/Write 这类有副作用的留在主线程串行，避免同文件竞态
        - API 约束：tool 消息必须按 tool_calls 数组顺序跟在 assistant 之后，
          所以"并行执行、按序回填"；SQLite 写入（_add）也留在主线程
        """
        results: list[str | None] = [None] * len(calls)
        safe_idx: list[int] = []
        serial_idx: list[int] = []
        for i, call in enumerate(calls):
            name = call.get("function", {}).get("name", "?")
            tool = self.registry.tools.get(name)
            if truncated or tool is None:
                results[i] = self._synthetic_result(name, truncated)
            elif tool.is_concurrency_safe():
                safe_idx.append(i)
            else:
                serial_idx.append(i)

        for i in serial_idx:  # 有副作用的工具：主线程串行
            call = calls[i]
            results[i] = self.registry.dispatch(
                call.get("function", {}).get("name", "?"),
                call.get("function", {}).get("arguments", "{}"),
            )

        if safe_idx:  # 无副作用工具：并行
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(4, len(safe_idx))
            ) as pool:
                futures = {
                    pool.submit(
                        self.registry.dispatch,
                        calls[i].get("function", {}).get("name", "?"),
                        calls[i].get("function", {}).get("arguments", "{}"),
                    ): i
                    for i in safe_idx
                }
                try:
                    for fut in concurrent.futures.as_completed(futures):
                        results[futures[fut]] = fut.result()
                except KeyboardInterrupt:
                    # 线程杀不掉，但杀掉子代理的子进程后 communicate 会返回、
                    # 线程自然退出，不会留孤儿
                    from .tools.subagent import terminate_active_children
                    terminate_active_children()
                    raise
        return [r if r is not None else "ERROR: 工具未返回结果" for r in results]

    @staticmethod
    def _synthetic_result(name: str, truncated: bool) -> str:
        if not truncated:
            return f"ERROR: 未知工具 {name}"
        return (
            f"ERROR: {name} 的参数被输出预算截断（finish_reason=length），"
            "内容不完整，未执行。请缩小单次输出后重试：写大文件时分成多个"
            " Write（每个 ≤100 行），或减少不必要的推理长度。"
        )

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
            # 先一次性打印全部 tool_call 行：并行任务全部分配可见，不再像"卡住"
            for call in calls:
                self.on_event("tool_call", {
                    "name": call.get("function", {}).get("name", "?"),
                    "arguments": call.get("function", {}).get("arguments", "{}"),
                })
            results = self._dispatch_batch(calls, truncated)
            for call, result in zip(calls, results):
                self.on_event("tool_result", {
                    "name": call.get("function", {}).get("name", "?"),
                    "result": result,
                })
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
