"""离线测试：并行工具派发 + 子代理进度面板。

不需要 API key：LLM 与慢工具全用假实现。
覆盖：安全工具真并行（墙钟时间）、结果按 tool_calls 顺序回填、
Edit/Write 类不安全工具留主线程串行、截断调用零派发、
进度事件 describe_event 提取、非 TTY 面板静默降级。
    uv run python examples/parallel_test.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_agent.config import Config  # noqa: E402
from mini_agent.loop import Agent  # noqa: E402
from mini_agent.tools import ToolRegistry, build_tools  # noqa: E402
from mini_agent.tools.base import Tool, ToolContext, ToolOutput  # noqa: E402
from mini_agent.tools.subagent import describe_event  # noqa: E402

PASSED = FAILED = 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok  {name}")
    else:
        FAILED += 1
        FAILURES.append(name)
        print(f"  FAIL {name}  {detail}")


MAIN_TID = threading.get_ident()


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


def batch_call(tid: str, name: str) -> dict:
    return {"id": tid, "type": "function",
            "function": {"name": name, "arguments": "{}"}}


def main() -> bool:
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
    check("混合批总时长 < 1.2s（并行段与安全工具重叠）", wall2 < 1.2, f"{wall2:.2f}s")
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

    print("=" * 60)
    print(f"PASS {PASSED}  FAIL {FAILED}")
    if FAILURES:
        print("失败用例:", ", ".join(FAILURES))
    return FAILED == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
