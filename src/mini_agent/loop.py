"""Agent 主循环：请求 → 工具调用 → 回填结果 → 再请求，直到没有工具调用。

每轮用户输入开始时重置 StallGuard。执行前询问 before_call（可拦截），
写回 tool 消息前走 after_call / observe_call（短引用、提示）；一批结果
写完后若已硬停，不再进入下一轮模型请求。不取代 max_steps 和溢出压缩。
"""
from __future__ import annotations

from collections import deque
import threading
from typing import Callable

from .client import ContextOverflowError, LLM, LLMError
from .config import Config
from .context import Context
from .guardrails import StallGuard, is_tool_failure, parse_tool_arguments
from .tools import ToolRegistry
from .tools.base import ExecutedTool

EventHook = Callable[[str, dict], None]


def _noop(event: str, payload: dict) -> None:
    return None


def _call_name(call: dict) -> str:
    return call.get("function", {}).get("name", "?")


def tool_message_content(result: ExecutedTool) -> str | list[dict]:
    """没有图片时仍是字符串。有图片时是 text 块加 image 块，base64 不进文本截断。"""
    images = [
        {"type": "image", "data": img["data"], "mimeType": img["mimeType"]}
        for img in result.images
        if img.get("data") and img.get("mimeType")
    ]
    if not images:
        return result.text
    blocks: list[dict] = []
    if result.text:
        blocks.append({"type": "text", "text": result.text})
    blocks.extend(images)
    return blocks


def _result_preview(result: ExecutedTool) -> str:
    text = result.text or ""
    if result.images:
        extra = f"[{len(result.images)} 张图片]"
        text = f"{text}\n{extra}" if text else extra
    return text


def _abort_running_children() -> None:
    """中断时杀掉还在跑的子代理和 shell，避免历史补完之后进程还活着。"""
    from .tools.shell import terminate_active_shells
    from .tools.subagent import terminate_active_children

    terminate_active_children()
    terminate_active_shells()


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
        hard_stop_enabled: bool = False,
    ):
        self.cfg = cfg
        self.registry = registry
        self.llm = llm or LLM(cfg)
        self.on_event = on_event or _noop
        self.on_delta = on_delta
        self.hard_stop_enabled = hard_stop_enabled
        self.guardrails = StallGuard(hard_stop_enabled=hard_stop_enabled)
        self.ctx = Context(cfg)
        self.store = store
        self.session_id = session_id
        self.jobs = getattr(registry, "jobs", None)
        self._pending_completions = deque()
        self._completion_lock = threading.Lock()
        if self.jobs is not None:
            self.jobs.on_complete(self._on_job_complete)
        if store is not None and session_id:
            view = store.load_view(session_id, system_prompt=self.ctx.system_prompt)
            # 新会话视图只有注入的 system，与 Context 默认值一致
            if len(view) > 1 or (view and store.message_count(session_id) > 0):
                self.ctx.messages = view
        # 工具执行期上报通道（Task 子代理进度等）；挂在共享 context 上，worker 线程可用
        if self.registry.context is not None:
            self.registry.context.event_hook = self.on_event
            self.registry.context.session_id = session_id

    # ---------- 历史写入（内存 + 磁盘同步）----------
    def _add(self, message: dict) -> None:
        self.ctx.add(message)
        if self.store is not None and self.session_id:
            self.store.append(self.session_id, message)

    def _sync_after_compaction(self, report=None) -> None:
        if self.store is not None and self.session_id:
            self.store.record_compaction(
                self.session_id,
                self.ctx.messages,
                before_tokens=getattr(report, "before_tokens", 0) or 0,
                after_tokens=getattr(report, "after_tokens", 0) or 0,
                dropped_messages=getattr(report, "dropped_messages", 0) or 0,
                summarised=bool(getattr(report, "summarised", False)),
            )

    # ---------- 工具批量派发（按序执行，连续的只读/Task 并行）----------
    def _dispatch_into(
        self, calls: list[dict], truncated: bool, results: list[ExecutedTool | None]
    ) -> None:
        """按 tool_calls 顺序执行，结果写入 results 的对应下标。

        连续的 is_concurrency_safe() 工具（Read/Ls/Glob/Grep/Task）并行。
        Edit/Write/run_command 是屏障，在主线程按原顺序执行——这样「先改文件
        再跑测试」不会把测试跑到修改前面。中断时已完成的下标保留，其余保持 None。
        """
        groups: list[tuple[str, list[int]]] = []
        parallel: list[int] = []

        def flush() -> None:
            nonlocal parallel
            if parallel:
                groups.append(("par", parallel))
                parallel = []

        for i, call in enumerate(calls):
            name = _call_name(call)
            tool = self.registry.tools.get(name)
            if truncated or tool is None:
                flush()
                results[i] = self._synthetic_result(name, truncated)
                continue
            if tool.is_concurrency_safe():
                parallel.append(i)
            else:
                flush()
                groups.append(("ser", [i]))
        flush()

        for kind, idxs in groups:
            if kind == "ser":
                results[idxs[0]] = self._dispatch_one(calls[idxs[0]])
            else:
                self._dispatch_parallel(calls, idxs, results)

    def _dispatch_one(self, call: dict) -> ExecutedTool:
        name = _call_name(call)
        raw_args = call.get("function", {}).get("arguments", "{}")
        decision = self.guardrails.before_call(name, parse_tool_arguments(raw_args))
        if not decision.allows_execution:
            return ExecutedTool(decision.synthetic_error)
        return self.registry.execute(name, raw_args)

    def _apply_guardrails(self, call: dict, result: ExecutedTool) -> ExecutedTool:
        """写回历史前：按需换成短引用、追加提示。多模态结果不进入相同调用盯梢。"""
        name = _call_name(call)
        args = parse_tool_arguments(call.get("function", {}).get("arguments", "{}"))
        text = result.text or ""
        failed = is_tool_failure(name, text)
        decision = self.guardrails.after_call(name, args, text, failed=failed)
        if result.images:
            observation = self.guardrails.observe_call(name, args, None, failed=failed)
        else:
            observation = self.guardrails.observe_call(
                name, args, text, failed=failed, tool_call_id=call.get("id", ""),
            )
        new_text = observation.stub if observation.stub else text
        for extra in (observation.notice, decision.message if decision.action in {"warn", "halt"} else None):
            if extra and extra not in new_text:
                new_text = f"{new_text}\n\n{extra}" if new_text else extra
        if new_text == result.text:
            return result
        return ExecutedTool(new_text, list(result.images))

    def _dispatch_parallel(
        self, calls: list[dict], idxs: list[int], results: list[ExecutedTool | None]
    ) -> None:
        import concurrent.futures

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(idxs)))
        futures: dict = {}
        try:
            futures = {pool.submit(self._dispatch_one, calls[i]): i for i in idxs}
            try:
                for fut in concurrent.futures.as_completed(futures):
                    results[futures[fut]] = fut.result()
            except KeyboardInterrupt:
                # 先杀掉子进程，worker 才能从 communicate / stdout 读里返回
                _abort_running_children()
                raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            for fut, i in futures.items():
                if results[i] is None and fut.done():
                    try:
                        results[i] = fut.result(timeout=0)
                    except BaseException:
                        pass

    @staticmethod
    def _synthetic_result(name: str, truncated: bool) -> ExecutedTool:
        if not truncated:
            return ExecutedTool(f"ERROR: 未知工具 {name}")
        return ExecutedTool(
            f"ERROR: {name} 的参数被输出预算截断（finish_reason=length），"
            "内容不完整，未执行。请缩小单次输出后重试：写大文件时分成多个"
            " Write（每个 ≤100 行），或减少不必要的推理长度。"
        )

    def _on_job_complete(self, completion) -> None:
        """任务 worker 只入队；不得在这里写 Context、Store 或调用 run。"""
        with self._completion_lock:
            self._pending_completions.append(completion)

    def _drain_completions(self) -> None:
        """在安全的 LLM step 边界消费通知，重复/已显式读取的事件直接丢弃。"""
        while True:
            with self._completion_lock:
                if not self._pending_completions:
                    return
                completion = self._pending_completions.popleft()
            if self.jobs is not None:
                completion = self.jobs.consume_notification(completion)
                if completion is None:
                    continue
            notice = (completion.message or "").strip()
            if not notice:
                continue
            if completion.delivery == "wake":
                notice += "\n\n（以上是后台 Subagent 的完成结果，请结合当前任务继续处理。）"
            self.record_event(notice)

    def record_event(self, text: str) -> None:
        """把一条通知写入会话历史，不发起模型轮次。"""
        text = (text or "").strip()
        if not text:
            return
        self._add({"role": "user", "content": text})

    # ---------- 对外 ----------
    def run(self, user_input: str, _event_override: EventHook | None = None) -> str:
        """跑一轮用户输入直到没有工具调用。_event_override 仅供 print 模式
        注入事件转发（把 assistant 事件实时转成 message_end 协议）。"""
        text = (user_input or "").strip()
        if not text:
            return ""

        on_event = _event_override or self.on_event
        self.guardrails.reset()
        self._add({"role": "user", "content": text})
        self._drain_completions()

        for step in range(1, self.cfg.max_steps + 1):
            self._compact_if_needed()

            # 上下文溢出：先压缩再重试本轮，恰好一次
            overflow_retried = False
            while True:
                try:
                    response = self.llm.chat_stream(
                        self.ctx.messages,
                        self.registry.schemas(),
                        on_delta=self.on_delta,
                        on_notice=lambda m: on_event("notice", {"message": m}),
                    )
                    break
                except ContextOverflowError as exc:
                    if overflow_retried:
                        on_event("error", {"message": str(exc), "step": step})
                        return f"ERROR: {exc}"
                    overflow_retried = True
                    on_event(
                        "notice", {"message": "上下文超出模型窗口，先压缩再重试本轮"}
                    )
                    overflow_report = self.ctx.compact(
                        self.llm, self.registry.schemas(), force=True
                    )
                    self._sync_after_compaction(overflow_report)
                except LLMError as exc:
                    on_event("error", {"message": str(exc), "step": step})
                    return f"ERROR: {exc}"

            assistant = {
                k: v
                for k, v in response.items()
                if not k.startswith("_") or k in ("_usage", "_finish_reason")
            }  # _usage/_finish_reason 是私有字段，发给 API 前由 _wire_messages 剥掉
            self._add(assistant)
            on_event(
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

            # finish_reason=length 说明思维链吃光了输出预算、工具参数被截断
            # （残缺 JSON 能解析但不完整，执行了更糟）。不派发，回合成错误让模型缩小输出重试。
            truncated = response.get("_finish_reason") == "length"
            # 先一次性打印全部 tool_call 行：并行任务全部分配可见，不再像"卡住"
            for call in calls:
                on_event("tool_call", {
                    "name": _call_name(call),
                    "arguments": call.get("function", {}).get("arguments", "{}"),
                })
            results: list[ExecutedTool | None] = [None] * len(calls)
            interrupted = False
            try:
                self._dispatch_into(calls, truncated, results)
            except KeyboardInterrupt:
                interrupted = True
                _abort_running_children()
            # 无论是否中断，每个 tool_call 都要有一条 tool 消息，否则下一轮 API 直接 400
            for i, call in enumerate(calls):
                if results[i] is None:
                    name = _call_name(call)
                    results[i] = ExecutedTool(
                        f"ERROR: {name} 已中断，未完成" if interrupted
                        else "ERROR: 工具未返回结果"
                    )
            for call, result in zip(calls, results):
                done = self._apply_guardrails(
                    call, result or ExecutedTool("ERROR: 工具未返回结果")
                )
                on_event("tool_result", {
                    "name": _call_name(call),
                    "result": _result_preview(done),
                })
                self._add(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": tool_message_content(done),
                    }
                )
            if interrupted:
                raise KeyboardInterrupt
            if self.guardrails.should_halt():
                msg = self.guardrails.halt_message()
                on_event("notice", {"message": msg})
                return msg
            # 所有 tool_call 的结果已补齐；此刻注入不会破坏 tool 消息配对。
            self._drain_completions()

        on_event("error", {"message": "达到最大工具步数"})
        return f"（达到最大工具步数 {self.cfg.max_steps}，已停止）"

    # ---------- 内部 ----------
    def _compact_if_needed(self) -> None:
        report = self.ctx.compact(self.llm, self.registry.schemas())
        if report is not None:
            self._sync_after_compaction(report)
            self.on_event("compaction", report.__dict__)

    def compact_now(self, force: bool = True):
        """手动压缩（/compact）：更新内存视图并记录压缩事件，不删原文。"""
        report = self.ctx.compact(self.llm, self.registry.schemas(), force=force)
        if report is not None:
            self._sync_after_compaction(report)
        return report

    def force_compact(self):
        self.ctx.force_fit(self.registry.schemas())
        self._sync_after_compaction()
        return self.ctx.size(self.registry.schemas())
