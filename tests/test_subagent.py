"""离线测试：子代理委派（print 模式 argv / JSON 事件解析 / 截断 / 注册开关）。

不需要 API key。真实端到端（父 agent 派 Task → 子进程跑真模型）见 smoke 或手动：
    uv run python -c \"...agent.run('用 Task ...')\"
    uv run pytest
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from mini_agent.config import Config  # noqa: E402
from mini_agent.jobs import TaskRegistry  # noqa: E402
from mini_agent.tools import ToolContext, ToolOutput, build_tools  # noqa: E402
from mini_agent.tools.base import ToolError  # noqa: E402
from mini_agent.tools.subagent import (  # noqa: E402
    CHILD_SYSTEM_PROMPT,
    ProcessTimeout,
    SubagentTool,
    build_child_argv,
    collect_stdout,
    parse_json_events,
    truncate_result,
)



def check(name: str, cond: bool, detail: str = "") -> None:
    assert cond, f"{name}  {detail}"



def test_subagent_suite() -> None:
    print("[build_child_argv]")
    argv = build_child_argv("做个任务", "/tmp/p.md")
    check("模块调用形态", argv[1:3] == ["-m", "mini_agent"], argv[:3])
    for flag in ("-p", "--json", "--no-session", "--append-system-prompt"):
        check(f"含 {flag}", flag in argv, argv)
    check("任务在 argv 末位附近", "做个任务" in argv, argv)
    check("默认带工具白名单", "--tools" in argv, argv)
    argv2 = build_child_argv("t", "/tmp/p.md", model="m1", tools="")
    check("model 显式传入", argv2[argv2.index("--model") + 1] == "m1", argv2)
    check("tools 空串不产出 flag", "--tools" not in argv2, argv2)

    print("[parse_json_events]")
    sample = "\n".join([
        "警告：某库输出了非 JSON 行",          # 应被跳过
        '{"type": "message_end", "message": {"role": "user", "content": []}}',
        '{"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "第一步"}]}, "stopReason": "tool_calls"}',
        '{"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "第二步"}]}, "stopReason": "tool_calls"}',
        '{"type": "result", "result": "最终答复"}',
    ])
    parsed = parse_json_events(sample)
    check("result 事件优先", parsed["final"] == "最终答复", parsed)
    check("assistant 计数", parsed["assistant_count"] == 2, parsed)
    check("末次 stopReason", parsed["last_stop"] == "tool_calls", parsed)

    no_result = parse_json_events(
        '{"type": "message_end", "message": {"role": "assistant", '
        '"content": [{"type": "text", "text": "唯一文本"}]}, "stopReason": "stop"}'
    )
    check("无 result 时取最后 assistant 文本", no_result["final"] == "唯一文本")

    err = parse_json_events('{"type": "error", "message": "API 挂了"}')
    check("error 事件解析", err["error"] == "API 挂了", err)

    print("[truncate_result]")
    big = "内容\n" * 30000
    t = truncate_result(big)
    check("超长截断", len(t) < 50050 and "已截断" in t, len(t))
    check("短文本原样", truncate_result("abc") == "abc")
    t2 = truncate_result("x" * 60000)
    check("无换行也可截断", "已截断" in t2)

    print("[注册开关]")
    reg = build_tools(Config(workdir=ROOT, enable_shell=True, enable_subagent=True))
    check("默认注册 Task", "Task" in reg.names(), reg.names())
    check("默认注册统一 task_manage", "task_manage" in reg.names(), reg.names())
    reg2 = build_tools(Config(workdir=ROOT, enable_shell=False, enable_subagent=False))
    check("enable_subagent=False 不注册", "Task" not in reg2.names(), reg2.names())
    # print 模式的防递归摘除（与 cli.run_print_mode 同款逻辑）
    reg.tools.pop("Task", None)
    check("摘除后 Task 消失", "Task" not in reg.names())

    print("[stdout 超时在进程退出前生效]")

    def spawn(code: str) -> subprocess.Popen:
        kwargs: dict = dict(
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        )
        if os.name != "nt":
            kwargs["start_new_session"] = True
        return subprocess.Popen([sys.executable, "-c", code], **kwargs)

    quick = spawn("print('hi-from-child')")
    try:
        got = collect_stdout(quick, 5, lambda _line: None)
    finally:
        if quick.poll() is None:
            quick.kill()
    check("时限内结束保留输出", any("hi-from-child" in line for line in got), got)

    hung = spawn("import time; time.sleep(30)")
    started = time.monotonic()
    timed_out = False
    try:
        try:
            collect_stdout(hung, 1, lambda _line: None)
        except ProcessTimeout:
            timed_out = True
    finally:
        if hung.poll() is None:
            hung.kill()
    elapsed = time.monotonic() - started
    check("挂起的子进程会在超时内被杀掉", timed_out and hung.poll() is not None, hung.poll())
    check("超时没有干等到子进程自己结束", elapsed < 8, f"{elapsed:.2f}s")

    print("[子代理 system prompt  sanity]")
    check("子代理 prompt 含返回格式要求", "返回格式" in CHILD_SYSTEM_PROMPT)
    check("子代理 prompt 不含 Task 字样（防诱导）", "Task" not in CHILD_SYSTEM_PROMPT)


def test_background_subagent_fast_completion_and_sync_compat(tmp_path, monkeypatch) -> None:
    jobs = TaskRegistry()
    tool = SubagentTool(jobs=jobs)
    ctx = ToolContext(tmp_path)
    ctx.session_id = "session-a"
    seen = []
    jobs.on_complete(seen.append)

    def fake_execute(task, timeout, ctx, tag, hook, job):
        return ToolOutput(f"result:{task}", {"type": "subagent", "tag": tag})

    monkeypatch.setattr(tool, "_execute", fake_execute)
    started = tool.call({"prompt": "background", "run_in_background": True}, ctx)
    task_id = started.meta["task_id"]
    assert "已启动后台子代理" in started.text
    job = jobs.get(task_id)
    assert job is not None
    assert job.done.wait(timeout=2)
    deadline = time.monotonic() + 2
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(seen) == 1
    assert job.owner_session_id == "session-a"
    assert job.delivery == "wake"

    # get 不消费；终态 output 消费后，排队中的自动通知失效。
    jobs.task_manage("get", task_id, owner_session_id="session-a")
    assert job.notification_state == "pending"
    output = jobs.task_manage("output", task_id, owner_session_id="session-a")
    assert "result:background" in output
    assert jobs.consume_notification(seen[0]) is None

    synchronous = tool.call({"prompt": "sync"}, ctx)
    assert synchronous.text == "result:sync"
    assert synchronous.meta["task_id"] != task_id
    assert len(seen) == 1  # 同步 Task 不产生后台通知


def test_print_mode_rejects_background_without_starting(tmp_path, monkeypatch) -> None:
    jobs = TaskRegistry(allow_notify=False)
    jobs.allow_background_subagents = False
    tool = SubagentTool(jobs=jobs)
    monkeypatch.setattr(
        tool,
        "_execute",
        lambda *args: (_ for _ in ()).throw(AssertionError("不应启动执行器")),
    )
    with pytest.raises(ToolError) as exc_info:
        tool.call(
            {"prompt": "not-allowed", "run_in_background": True},
            ToolContext(tmp_path),
        )
    assert exc_info.value.error_code == 24
    assert "使用同步 Task" in str(exc_info.value)
    assert jobs.all_jobs() == []


def test_background_subagent_runs_real_process_protocol(tmp_path, monkeypatch) -> None:
    # 不调用模型：用真实 Python 子进程模拟 mini-agent 的 JSONL result 事件。
    child_code = "print('{\"type\": \"result\", \"result\": \"process-done\"}')"
    monkeypatch.setattr(
        "mini_agent.tools.subagent.build_child_argv",
        lambda _task, _prompt_file: [sys.executable, "-c", child_code],
    )
    jobs = TaskRegistry()
    tool = SubagentTool(jobs=jobs)
    started = tool.call(
        {"prompt": "real-process", "run_in_background": True, "timeout": 60},
        ToolContext(tmp_path),
    )
    task_id = started.meta["task_id"]
    job = jobs.get(task_id)
    assert job is not None
    assert job.done.wait(timeout=5)
    assert job.exit_code == 0
    assert not job.error
    output = jobs.task_manage("output", task_id)
    assert "process-done" in output
    assert job.meta["type"] == "subagent"


