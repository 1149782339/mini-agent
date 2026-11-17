"""离线测试：防空转护栏（相同调用盯梢、周期循环、失败拦截）。

不打真实模型，不需要 API key。
"""
from __future__ import annotations

from pathlib import Path

from mini_agent.config import Config
from mini_agent.guardrails import (
    IDENTICAL_HALT_AFTER,
    IDENTICAL_STUB_MIN_CHARS,
    SAME_TOOL_FAILURE_HALT_AFTER,
    StallGuard,
    is_repeatable,
    is_tool_failure,
)
from mini_agent.loop import Agent
from mini_agent.tools import ToolRegistry
from mini_agent.tools.base import Tool, ToolOutput


LONG = "X" * (IDENTICAL_STUB_MIN_CHARS + 10)


class EchoTool(Tool):
    name = "Read"
    description = "测试用只读工具"
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    calls: list[dict] = []
    payload: str = LONG

    def is_concurrency_safe(self) -> bool:
        return True

    def call(self, args, ctx) -> ToolOutput:
        EchoTool.calls.append(dict(args))
        return ToolOutput(self.payload)


class FailTool(Tool):
    name = "Boom"
    description = "总是失败"
    parameters = {
        "type": "object",
        "properties": {"n": {"type": "string"}},
        "required": ["n"],
        "additionalProperties": False,
    }
    calls: list[str] = []

    def call(self, args, ctx) -> ToolOutput:
        FailTool.calls.append(str(args.get("n")))
        return ToolOutput(f"ERROR: boom {args.get('n')}")


class EditOkTool(Tool):
    name = "Edit"
    description = "假装编辑成功"
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def call(self, args, ctx) -> ToolOutput:
        return ToolOutput("已修改")


class RepeatLLM:
    """每一步都发起同一批 tool_calls，直到测试方改 stop。"""

    def __init__(self, batch: list[dict], *, stop_after: int | None = None):
        self.batch = batch
        self.stop_after = stop_after
        self.calls = 0

    def chat_stream(self, messages, tools=None, on_delta=None, on_notice=None):
        self.calls += 1
        if self.stop_after is not None and self.calls > self.stop_after:
            return {"role": "assistant", "content": "全部完成", "_finish_reason": "stop"}
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": self.batch,
            "_finish_reason": "stop",
        }

    def summarize(self, prompt):
        return "摘要"


def _call(tid: str, name: str, arguments: str) -> dict:
    return {"id": tid, "type": "function", "function": {"name": name, "arguments": arguments}}


def _agent(llm, tools: list[Tool], *, hard_stop: bool, workdir: Path) -> Agent:
    cfg = Config(workdir=workdir, enable_shell=False, enable_subagent=False, max_steps=20)
    reg = ToolRegistry(max_output=80000, context=None)
    for t in tools:
        reg.register(t)
    return Agent(cfg, reg, llm=llm, hard_stop_enabled=hard_stop)


def _observe(guard: StallGuard, tool: str, args: dict, result: str, *, failed: bool = False):
    return guard.observe_call(tool, args, result, failed=failed)


def test_second_identical_long_result_becomes_stub():
    g = StallGuard()
    first = _observe(g, "Read", {"path": "/a"}, LONG)
    assert first.stub is None and first.notice is None
    second = _observe(g, "Read", {"path": "/a"}, LONG)
    assert second.stub is not None
    assert "重复结果" in second.stub
    assert "Read" in second.stub
    assert second.notice is None
    short2 = StallGuard()
    _observe(short2, "Read", {"path": "/a"}, "short")
    obs = _observe(short2, "Read", {"path": "/a"}, "short")
    assert obs.stub is None
    # 失败结果不做短引用
    f = StallGuard()
    err = "ERROR: " + "z" * IDENTICAL_STUB_MIN_CHARS
    _observe(f, "Read", {"path": "/a"}, err, failed=True)
    obs_f = _observe(f, "Read", {"path": "/a"}, err, failed=True)
    assert obs_f.stub is None


def test_third_identical_call_gets_hint():
    g = StallGuard()
    for i in range(3):
        obs = _observe(g, "Read", {"path": "/a"}, LONG)
    assert obs.notice is not None and "连续第 3 次" in obs.notice
    assert obs.kind == "identical_call_streak"
    assert obs.stub is not None


def test_fifth_identical_call_hard_stop_policy():
    armed = StallGuard(hard_stop_enabled=True)
    for _ in range(IDENTICAL_HALT_AFTER):
        obs = _observe(armed, "Read", {"path": "/a"}, LONG)
    assert armed.should_halt()
    assert armed.halt_decision() is not None
    assert armed.halt_decision().code == "identical_call_streak_halt"
    assert "连续第 5 次" in (obs.notice or "")
    assert "我停了" in armed.halt_message()

    quiet = StallGuard(hard_stop_enabled=False)
    for _ in range(IDENTICAL_HALT_AFTER + 2):
        _observe(quiet, "Read", {"path": "/a"}, LONG)
    assert not quiet.should_halt()
    assert quiet.halt_decision() is None


def test_cycle_period_2_3_4_detected():
    def run_cycle(period: int):
        g = StallGuard()
        tools = [("Read", {"path": f"/p{i}"}, f"body-{i}-{LONG}") for i in range(period)]
        last = None
        # 3 圈后应提示
        for _ in range(3):
            for name, args, result in tools:
                last = _observe(g, name, args, result)
        assert last is not None
        assert last.kind == "identical_cycle"
        assert last.notice is not None and f"{period} 个工具调用" in last.notice
        assert not g.should_halt()
        return g, tools

    for period in (2, 3, 4):
        run_cycle(period)

    # 硬停：第 5 圈结束
    g = StallGuard(hard_stop_enabled=True)
    tools = [("Read", {"path": "/a"}, "A" + LONG), ("Grep", {"pattern": "x"}, "B" + LONG)]
    for _ in range(IDENTICAL_HALT_AFTER):
        for name, args, result in tools:
            _observe(g, name, args, result)
    assert g.should_halt()
    assert g.halt_decision().code == "identical_cycle_halt"


def test_identical_failure_warn_then_intercept():
    tool = FailTool()
    FailTool.calls.clear()
    llm = RepeatLLM(
        [_call("t1", "Boom", '{"n":"1"}')],
        stop_after=10,
    )
    agent = _agent(llm, [tool], hard_stop=True, workdir=Path("/tmp"))
    answer = agent.run("反复失败")
    # 前 4 次真正执行；第 5 次模型请求后在 before_call 拦截，然后硬停。
    assert FailTool.calls == ["1", "1", "1", "1"], FailTool.calls
    assert llm.calls == 5
    tool_msgs = [m["content"] for m in agent.ctx.messages if m.get("role") == "tool"]
    assert any("已经失败 2 次" in c for c in tool_msgs)
    assert tool_msgs[-1].startswith("ERROR: 已拦截")
    assert "第 5 次未执行" in tool_msgs[-1]
    assert "我停了" in answer


def test_run_command_different_params_do_not_hard_stop():
    g = StallGuard(hard_stop_enabled=True)
    for i in range(SAME_TOOL_FAILURE_HALT_AFTER + 3):
        d = g.after_call(
            "run_command",
            {"command": f"pytest test_{i}.py"},
            "ERROR: failed",
            failed=True,
        )
        assert not d.should_halt, i
    assert not g.should_halt()

    other = StallGuard(hard_stop_enabled=True)
    last = None
    for i in range(SAME_TOOL_FAILURE_HALT_AFTER):
        last = other.after_call("Write", {"path": f"/f{i}"}, "ERROR: no", failed=True)
    assert last is not None and last.should_halt
    assert other.should_halt()
    assert last.code == "same_tool_failure_halt"


def test_progress_success_clears_failure_counters():
    g = StallGuard(hard_stop_enabled=True)
    for i in range(4):
        g.after_call("Write", {"path": f"/f{i}"}, "ERROR: no", failed=True)
    # 还没到 8 次硬停；一次成功 Edit 清空计数
    cleared = g.after_call("Edit", {"path": "/x"}, "已修改", failed=False)
    assert not cleared.should_halt
    # 清空后再失败，不应被当作已经失败很多次
    for i in range(4):
        d = g.after_call("Write", {"path": f"/g{i}"}, "ERROR: no", failed=True)
        assert not d.should_halt
    assert g.before_call("Write", {"path": "/g0"}).allows_execution

    # run_command 成功同样清计数
    g2 = StallGuard(hard_stop_enabled=True)
    for _ in range(4):
        g2.after_call("Write", {"path": "/a"}, "ERROR: no", failed=True)
    g2.after_call("run_command", {"command": "pytest"}, "exit=0  time=0.1s", failed=False)
    d = g2.before_call("Write", {"path": "/a"})
    assert d.allows_execution
    assert is_tool_failure("run_command", "exit=1  time=0.1s  cwd=/tmp")
    assert not is_tool_failure("run_command", "exit=0  time=0.1s  cwd=/tmp")


def test_controller_resets_on_new_agent_run(tmp_path: Path):
    FailTool.calls.clear()
    llm = RepeatLLM([_call("t1", "Boom", '{"n":"1"}')], stop_after=2)
    agent = _agent(llm, [FailTool()], hard_stop=True, workdir=tmp_path)
    agent.run("第一轮")
    assert len(FailTool.calls) == 2
    # 新的用户轮应重置失败计数，前几次仍会真正执行
    FailTool.calls.clear()
    llm2 = RepeatLLM([_call("t1", "Boom", '{"n":"1"}')], stop_after=2)
    agent.llm = llm2
    agent.run("第二轮")
    assert FailTool.calls == ["1", "1"], FailTool.calls
    assert llm2.calls == 3  # 两次工具 + 一次最终答复


def test_identical_call_still_executes_until_intercept_or_halt(tmp_path: Path):
    EchoTool.calls.clear()
    EchoTool.payload = LONG
    llm = RepeatLLM([_call("r1", "Read", '{"path":"/a.txt"}')])
    agent = _agent(llm, [EchoTool()], hard_stop=True, workdir=tmp_path)
    answer = agent.run("反复读取")
    assert len(EchoTool.calls) == IDENTICAL_HALT_AFTER
    assert llm.calls == IDENTICAL_HALT_AFTER  # 第 5 次写回后硬停，不再请求
    texts = [m["content"] for m in agent.ctx.messages if m.get("role") == "tool"]
    assert texts[0] == LONG
    assert texts[1].startswith("[重复结果：")
    assert "连续第 3 次" in texts[2]
    assert "连续第 5 次" in texts[4]
    assert "我停了" in answer
    assert "Read" in answer

    # 交互模式：第 5 次之后仍继续请求模型
    EchoTool.calls.clear()
    llm_i = RepeatLLM([_call("r1", "Read", '{"path":"/a.txt"}')], stop_after=6)
    agent_i = _agent(llm_i, [EchoTool()], hard_stop=False, workdir=tmp_path)
    answer_i = agent_i.run("交互反复读取")
    assert len(EchoTool.calls) == 6
    assert llm_i.calls == 7  # 6 次工具 + 最终答复
    assert answer_i == "全部完成"
    assert not agent_i.guardrails.should_halt()


def test_repeatable_hook_skips_hint_but_keeps_stub():
    assert not is_repeatable("Read")
    assert is_repeatable("status_poll")
    g = StallGuard(hard_stop_enabled=True)
    last = None
    for _ in range(5):
        last = _observe(g, "status_poll", {"id": "1"}, LONG)
    assert last is not None
    assert last.stub is not None
    assert last.notice is None
    assert not g.should_halt()
