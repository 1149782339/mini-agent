"""离线测试：并行工具派发 + 子代理进度面板。

不需要 API key：LLM 与慢工具全用假实现。
覆盖：安全工具真并行（墙钟时间）、结果按 tool_calls 顺序回填、
Edit/Write 类不安全工具留主线程串行、截断调用零派发、
进度事件 describe_event 提取、非 TTY 面板静默降级。
    uv run pytest
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from mini_agent.config import Config  # noqa: E402
from mini_agent.loop import Agent  # noqa: E402
from mini_agent.tools import ToolRegistry, build_tools  # noqa: E402
from mini_agent.tools.base import Tool, ToolContext, ToolOutput  # noqa: E402
from mini_agent.tools.subagent import describe_event  # noqa: E402

MAIN_TID = threading.get_ident()



def check(name: str, cond: bool, detail: str = "") -> None:
    assert cond, f"{name}  {detail}"



class SlowSafeTool(Tool):
    """并发安全的慢工具（模拟 Task/只读工具）：睡 0.5s 后返回，记录每次调用的线程 id。"""
    name = "slow_safe"
    description = "慢的安全工具"
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    tids: list[int] = []

    def is_concurrency_safe(self) -> bool:
        return True

    def call(self, args, ctx) -> ToolOutput:
        SlowSafeTool.tids.append(threading.get_ident())
        time.sleep(0.5)
        return ToolOutput("safe-done")


class SlowUnsafeTool(Tool):
    """有副作用的慢工具（模拟 Edit/Write）：必须留在主线程串行。"""
    name = "slow_unsafe"
    description = "慢的不安全工具"
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    tid = None
    entered_at: list[float] = []

    def call(self, args, ctx) -> ToolOutput:
        SlowUnsafeTool.tid = threading.get_ident()
        SlowUnsafeTool.entered_at.append(time.monotonic())
        time.sleep(0.4)
        return ToolOutput("unsafe-done")


class BoomTool(Tool):
    """被调用即报错（验证截断时零派发）。"""
    name = "boom"
    description = "不应被调用"
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    called = False

    def is_concurrency_safe(self) -> bool:
        return True

    def call(self, args, ctx) -> ToolOutput:
        BoomTool.called = True
        return ToolOutput("boom")


class ScriptLLM:
    """按剧本吐响应的 LLM：一次工具调用批 → 最终答复。"""

    def __init__(self, batch: list[dict], finish: str = "stop"):
        self.batch = batch
        self.finish = finish
        self.calls = 0

    def chat_stream(self, messages, tools=None, on_delta=None, on_notice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant", "content": "",
                "tool_calls": self.batch,
                "_finish_reason": self.finish,
                "_usage": {"input": 1, "output": 2, "total": 3},
            }
        return {"role": "assistant", "content": "全部完成",
                "_finish_reason": "stop", "_usage": {"input": 1, "output": 2, "total": 3}}

    def summarize(self, prompt):
        return "摘要"


def make_agent(llm, extra_tools: list[Tool] | None = None) -> Agent:
    workdir = ROOT / "sandbox" / "parallel_test"
    workdir.mkdir(parents=True, exist_ok=True)
    reg = build_tools(Config(workdir=workdir, enable_shell=False, enable_subagent=False))
    for t in extra_tools or []:
        reg.register(t)
    return Agent(Config(workdir=workdir, enable_shell=False), reg, llm=llm)


def batch_call(tid: str, name: str, arguments: str = "{}") -> dict:
    return {"id": tid, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def test_parallel_suite() -> None:
    print("[真并行：同一安全工具被并发调两次]")
    SlowSafeTool.tids.clear()
    llm = ScriptLLM([batch_call("a", "slow_safe"), batch_call("b", "slow_safe")])
    agent = make_agent(llm, [SlowSafeTool()])
    start = time.monotonic()
    answer = agent.run("并行跑两个")
    wall = time.monotonic() - start
    check("墙钟 < 0.9s（串行需 ≥1.0s）", wall < 0.9, f"{wall:.2f}s")
    check("两个调用都执行了", len(SlowSafeTool.tids) == 2, SlowSafeTool.tids)
    check("跑在 worker 线程（非主线程）",
          all(t != MAIN_TID for t in SlowSafeTool.tids), SlowSafeTool.tids)
    check("答复正常", answer == "全部完成", answer)
    tool_msgs = [m for m in agent.ctx.messages if m.get("role") == "tool"]
    check("tool 消息按调用顺序回填（a 在前 b 在后）",
          [m["tool_call_id"] for m in tool_msgs] == ["a", "b"],
          [m["tool_call_id"] for m in tool_msgs])
    check("配对校验通过", not agent.ctx.validate(), agent.ctx.validate())

    print("[串行分区：不安全工具留主线程]")
    SlowUnsafeTool.entered_at.clear()
    llm2 = ScriptLLM([batch_call("x", "slow_unsafe"), batch_call("y", "slow_safe")])
    agent2 = make_agent(llm2, [SlowUnsafeTool(), SlowSafeTool()])
    start2 = time.monotonic()
    agent2.run("混合批")
    wall2 = time.monotonic() - start2
    check("不安全工具在主线程执行", SlowUnsafeTool.tid == MAIN_TID, SlowUnsafeTool.tid)
    check("混合批总时长 < 1.2s（屏障先跑完，后面的安全工具再跑）", wall2 < 1.2, f"{wall2:.2f}s")
    check("tool 顺序回填（x 前 y 后）",
          [m["tool_call_id"] for m in agent2.ctx.messages if m.get("role") == "tool"]
          == ["x", "y"])

    print("[截断：零派发]")
    BoomTool.called = False
    llm3 = ScriptLLM([batch_call("z", "boom")], finish="length")
    agent3 = make_agent(llm3, [BoomTool()])
    agent3.run("触发截断")
    check("截断时工具零派发", not BoomTool.called)
    z = [m for m in agent3.ctx.messages if m.get("role") == "tool" and m.get("tool_call_id") == "z"]
    check("合成截断错误文本", z and "被输出预算截断" in z[0]["content"], z)

    print("[进度事件提取 describe_event]")
    ok, tok, snip = describe_event({
        "type": "message_end",
        "message": {"role": "assistant",
                    "content": [{"type": "text", "text": "正在读取\nexecutor.c 的第 3 部分"}]},
        "usage": {"total": 12400},
    })
    check("assistant 步识别", ok)
    check("token 提取", tok == 12400, tok)
    check("摘要单行化+截40字", "\n" not in snip and len(snip) <= 40, repr(snip))
    not_ok = describe_event({"type": "message_end",
                             "message": {"role": "user", "content": []}})
    check("非 assistant 消息忽略", not_ok == (False, 0, ""))

    print("[面板：非 TTY 静默降级]")
    from mini_agent.cli import SubagentPanels
    panels = SubagentPanels(enabled=False)
    panels.update({"tag": "子代理#1", "step": 1, "tokens": 100, "snippet": "x", "status": "running"})
    panels.stop_all()
    check("非 TTY 更新不炸", True)

    print("[顺序：连续只读并行，屏障不提前]")
    order: list[tuple[str, str, float]] = []
    order_lock = threading.Lock()

    def mark(phase: str, key: str) -> None:
        with order_lock:
            order.append((phase, key, time.monotonic()))

    class OrderSafe(Tool):
        name = "order_safe"
        description = "可并行"
        parameters = {"type": "object", "properties": {"k": {"type": "string"}},
                      "required": ["k"], "additionalProperties": False}

        def is_concurrency_safe(self) -> bool:
            return True

        def call(self, args, ctx) -> ToolOutput:
            mark("start", args["k"])
            time.sleep(0.35)
            mark("end", args["k"])
            return ToolOutput(args["k"])

    class OrderBarrier(Tool):
        name = "order_barrier"
        description = "屏障"
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}
        tid = None

        def call(self, args, ctx) -> ToolOutput:
            OrderBarrier.tid = threading.get_ident()
            mark("start", "barrier")
            time.sleep(0.2)
            mark("end", "barrier")
            return ToolOutput("barrier")

    def stamped(phase: str, key: str) -> float:
        return next(t for p, k, t in order if p == phase and k == key)

    llm_order = ScriptLLM([
        batch_call("a", "order_safe", '{"k": "a"}'),
        batch_call("b", "order_safe", '{"k": "b"}'),
        batch_call("m", "order_barrier"),
        batch_call("c", "order_safe", '{"k": "c"}'),
    ])
    make_agent(llm_order, [OrderSafe(), OrderBarrier()]).run("按顺序")
    check("a 与 b 时间重叠",
          stamped("start", "b") < stamped("end", "a") and stamped("start", "a") < stamped("end", "b"))
    check("屏障不早于 a、b 结束",
          stamped("start", "barrier") >= stamped("end", "a") - 0.05
          and stamped("start", "barrier") >= stamped("end", "b") - 0.05)
    check("c 不早于屏障结束", stamped("start", "c") >= stamped("end", "barrier") - 0.05)
    check("屏障在主线程", OrderBarrier.tid == MAIN_TID, OrderBarrier.tid)

    print("[中断：补齐尚未回填的 tool 结果]")

    class InterruptTool(Tool):
        name = "interrupt"
        description = "中断"
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}

        def call(self, args, ctx) -> ToolOutput:
            raise KeyboardInterrupt()

    llm_stop = ScriptLLM([
        batch_call("a", "slow_safe"),
        batch_call("b", "interrupt"),
        batch_call("c", "slow_safe"),
    ])
    agent_stop = make_agent(llm_stop, [SlowSafeTool(), InterruptTool()])
    raised = False
    try:
        agent_stop.run("中途中断")
    except KeyboardInterrupt:
        raised = True
    stopped = [m for m in agent_stop.ctx.messages if m.get("role") == "tool"]
    check("中断继续向外抛", raised)
    check("三个 tool_call 都有结果",
          [m.get("tool_call_id") for m in stopped] == ["a", "b", "c"],
          [m.get("tool_call_id") for m in stopped])
    check("已完成的调用保留结果", stopped[0]["content"] == "safe-done", stopped[0]["content"])
    check("未完成的调用标成中断",
          "已中断" in stopped[1]["content"] and "已中断" in stopped[2]["content"],
          [stopped[1]["content"], stopped[2]["content"]])
    check("中断后配对仍然完整", not agent_stop.ctx.validate(), agent_stop.ctx.validate())

    print("[写工具是屏障]")
    from mini_agent.tools.edit import EditTool
    from mini_agent.tools.write import WriteTool
    check("Edit 不进并行组", EditTool().is_concurrency_safe() is False)
    check("Write 不进并行组", WriteTool().is_concurrency_safe() is False)

    print("[回归：SubagentTool 必须声明并发安全]")
    from mini_agent.tools.subagent import SubagentTool
    check("Task 并发安全（否则并行派发被静默降级为串行）",
          SubagentTool().is_concurrency_safe() is True)


