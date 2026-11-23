from __future__ import annotations

import json
import threading
import time

from mini_agent.cli import InteractiveSession
from mini_agent.cli_interaction import CLIInteractionHandler
from mini_agent.config import Config
from mini_agent.tools import build_tools


def _args(*, multi_select: bool = False, question: str = "选择方案") -> dict:
    return {
        "questions": [{
            "question": question,
            "options": [
                {"label": "方案 A", "description": "使用现有结构"},
                {"label": "方案 B", "description": "引入新的抽象"},
            ],
            "multi_select": multi_select,
        }],
    }


def _normalized(*, multi_select: bool = False) -> list[dict]:
    return [{
        "qid": "q0",
        "question": "选择方案",
        "options": [
            {"label": "方案 A", "description": "使用现有结构"},
            {"label": "方案 B", "description": "引入新的抽象"},
        ],
        "multi_select": multi_select,
    }]


def _registry(tmp_path, handler=None):
    return build_tools(
        Config(workdir=tmp_path, enable_shell=False, enable_subagent=False),
        interaction_handler=handler,
    )


def _start(call):
    results = []
    thread = threading.Thread(target=lambda: results.append(call()), daemon=True)
    thread.start()
    return thread, results


def _wait_pending(handler: CLIInteractionHandler) -> None:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with handler._lock:
            if handler._pending is not None:
                return
        time.sleep(0.005)
    raise AssertionError("handler 没有进入等待回答状态")


def _route(handler: CLIInteractionHandler, kind: str, payload) -> None:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if handler.route(kind, payload):
            return
        time.sleep(0.005)
    raise AssertionError("handler 没有接受输入")


def _join(thread, results):
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert len(results) == 1
    return results[0]


def test_without_handler_returns_unavailable_immediately(tmp_path) -> None:
    started = time.monotonic()
    result = json.loads(_registry(tmp_path).execute("AskUser", _args()).text)
    assert time.monotonic() - started < 0.5
    assert result["outcome"] == "unavailable"
    assert result["responses"][0]["status"] == "unanswered"
    assert "不要重复调用 AskUser" in result["notice"]


def test_handler_receives_normalized_questions_and_formats_answers(tmp_path) -> None:
    seen = []

    def handler(questions):
        seen.extend(questions)
        return {"answers": {"q0": "方案 B", "q1": None}, "outcome": "submitted"}

    args = _args()
    args["questions"].append({
        "question": "还要实现哪些能力？",
        "options": [
            {"label": "测试", "description": "补充回归测试"},
            {"label": "文档", "description": "补充使用说明"},
        ],
        "multi_select": True,
    })
    result = json.loads(_registry(tmp_path, handler).execute("AskUser", args).text)

    assert [q["qid"] for q in seen] == ["q0", "q1"]
    assert seen[1]["multi_select"] is True
    assert result["outcome"] == "submitted"
    assert result["responses"][0] == {
        "question": "选择方案", "status": "answered", "user_response": "方案 B",
    }
    assert result["responses"][1]["status"] == "skipped"
    assert result["responses"][1]["user_response"] is None


def test_multi_select_and_custom_answers_are_preserved(tmp_path) -> None:
    def handler(_questions):
        return {
            "answers": {"q0": ["方案 A", "自定义方案"]},
            "outcome": "submitted",
        }

    result = json.loads(
        _registry(tmp_path, handler).execute("AskUser", _args(multi_select=True)).text
    )
    assert result["responses"][0]["user_response"] == ["方案 A", "自定义方案"]


def test_invalid_questions_are_rejected_before_handler(tmp_path) -> None:
    called = False

    def handler(_questions):
        nonlocal called
        called = True
        return {"answers": {}, "outcome": "submitted"}

    registry = _registry(tmp_path, handler)
    duplicate = _args()
    duplicate["questions"].append(_args()["questions"][0])
    one_option = _args()
    one_option["questions"][0]["options"] = one_option["questions"][0]["options"][:1]
    duplicate_label = _args()
    duplicate_label["questions"][0]["options"][1]["label"] = "方案 A"
    too_many = {
        "questions": [_args(question=f"问题 {i}")["questions"][0] for i in range(5)],
    }

    for args in ({}, {"questions": []}, duplicate, one_option, duplicate_label, too_many):
        assert registry.execute("AskUser", args).text.startswith("ERROR:")
    assert called is False


def test_handler_failure_becomes_tool_error(tmp_path) -> None:
    def handler(_questions):
        raise RuntimeError("UI unavailable")

    result = _registry(tmp_path, handler).execute("AskUser", _args()).text
    assert result.startswith("ERROR:")
    assert "UI unavailable" in result


def test_cli_handler_single_multi_and_custom_answers() -> None:
    output = []
    handler = CLIInteractionHandler(timeout=1, output=output.append)

    thread, results = _start(lambda: handler.ask(_normalized()))
    _route(handler, "user", "2")
    assert _join(thread, results)["answers"]["q0"] == "方案 B"

    thread, results = _start(lambda: handler.ask(_normalized(multi_select=True)))
    _route(handler, "user", "1, 自定义方案")
    assert _join(thread, results)["answers"]["q0"] == ["方案 A", "自定义方案"]

    thread, results = _start(lambda: handler.ask(_normalized()))
    _route(handler, "user", "直接回答")
    assert _join(thread, results)["answers"]["q0"] == "直接回答"
    assert any("Agent 请求你回答" in line for line in output)


def test_cli_handler_skip_cancel_and_timeout() -> None:
    handler = CLIInteractionHandler(timeout=1, output=lambda _line: None)
    thread, results = _start(lambda: handler.ask(_normalized()))
    _route(handler, "user", "/skip")
    reply = _join(thread, results)
    assert reply["outcome"] == "submitted"
    assert reply["answers"]["q0"] is None

    thread, results = _start(lambda: handler.ask(_normalized()))
    _route(handler, "user", "/cancel")
    assert _join(thread, results)["outcome"] == "cancelled"

    impatient = CLIInteractionHandler(timeout=0.02, output=lambda _line: None)
    assert impatient.ask(_normalized())["outcome"] == "timed_out"


def test_interactive_session_routes_answers_and_preserves_eof() -> None:
    handler = CLIInteractionHandler(timeout=1, output=lambda _line: None)
    session = InteractiveSession(
        None, None, "session", None, None, False, None, interaction=handler,
    )

    thread, results = _start(lambda: handler.ask(_normalized()))
    _wait_pending(handler)
    session.submit("user", "1")
    assert _join(thread, results)["answers"]["q0"] == "方案 A"
    assert session.events.empty()

    session.submit("user", "普通消息")
    assert session.events.get_nowait() == ("user", "普通消息")

    thread, results = _start(lambda: handler.ask(_normalized()))
    _wait_pending(handler)
    session.submit("eof", None)
    assert _join(thread, results)["outcome"] == "cancelled"
    assert session.events.get_nowait() == ("eof", None)


def test_real_registry_and_cli_handler_round_trip(tmp_path) -> None:
    handler = CLIInteractionHandler(timeout=1, output=lambda _line: None)
    registry = _registry(tmp_path, handler.ask)
    thread, results = _start(lambda: registry.execute("AskUser", _args()).text)
    _route(handler, "user", "1")
    result = json.loads(_join(thread, results))

    assert result["outcome"] == "submitted"
    assert result["responses"][0]["status"] == "answered"
    assert result["responses"][0]["user_response"] == "方案 A"
