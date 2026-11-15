"""离线测试：子代理委派（print 模式 argv / JSON 事件解析 / 截断 / 注册开关）。

不需要 API key。真实端到端（父 agent 派 Task → 子进程跑真模型）见 smoke 或手动：
    uv run python -c \"...agent.run('用 Task ...')\"
    uv run python examples/subagent_test.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_agent.config import Config  # noqa: E402
from mini_agent.tools import build_tools  # noqa: E402
from mini_agent.tools.subagent import (  # noqa: E402
    CHILD_SYSTEM_PROMPT,
    build_child_argv,
    parse_json_events,
    truncate_result,
)

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


def main() -> bool:
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
    reg2 = build_tools(Config(workdir=ROOT, enable_shell=False, enable_subagent=False))
    check("enable_subagent=False 不注册", "Task" not in reg2.names(), reg2.names())
    # print 模式的防递归摘除（与 cli.run_print_mode 同款逻辑）
    reg.tools.pop("Task", None)
    check("摘除后 Task 消失", "Task" not in reg.names())

    print("[子代理 system prompt  sanity]")
    check("子代理 prompt 含返回格式要求", "返回格式" in CHILD_SYSTEM_PROMPT)
    check("子代理 prompt 不含 Task 字样（防诱导）", "Task" not in CHILD_SYSTEM_PROMPT)

    print("=" * 60)
    print(f"PASS {PASSED}  FAIL {FAILED}")
    if FAILURES:
        print("失败用例:", ", ".join(FAILURES))
    return FAILED == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
