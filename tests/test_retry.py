"""离线测试：API 错误重试 + 上下文溢出恢复。

不触网：monkeypatch 掉 openai client 的 chat.completions.create，
按"剧本"依次抛异常或返回假流；time.sleep 打桩成空转。
    uv run pytest
"""
from __future__ import annotations

import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import types
from openai import APIConnectionError, APIStatusError  # noqa: E402

import mini_agent.client as client_mod  # noqa: E402
from mini_agent.client import (  # noqa: E402
    ContextOverflowError,
    LLM,
    LLMError,
    context_tokens,
)
from mini_agent.config import Config  # noqa: E402
from mini_agent.loop import Agent  # noqa: E402
from mini_agent.tools import build_tools  # noqa: E402

import pytest


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """只在本模块测试期间打掉退避等待，并在结束后恢复，避免影响其他用例里的 time.sleep。"""
    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)



def check(name: str, cond: bool, detail: str = "") -> None:
    assert cond, f"{name}  {detail}"



def make_stream(text: str = "你好") -> list:
    """一段正常流：内容 chunk + 收尾 usage chunk。"""
    return [
        types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                finish_reason="stop",
                delta=types.SimpleNamespace(
                    reasoning_content=None, content=text, tool_calls=None),
            )],
            usage=None,
        ),
        types.SimpleNamespace(
            choices=[],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        ),
    ]


def make_llm_with_script(script: list) -> LLM:
    """LLM 实例，其底层 create 按剧本执行（异常实例或假流）。"""
    cfg = Config(api_key="test")
    llm = LLM(cfg)
    calls = {"n": 0}

    def create(**_kwargs):
        calls["n"] += 1
        item = script[min(calls["n"] - 1, len(script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    fake = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        )
    )
    llm.client = fake
    llm._call_count = calls  # type: ignore[attr-defined]
    return llm


def conn_error() -> APIConnectionError:
    return APIConnectionError(request=types.SimpleNamespace())


def status_400(message: str) -> APIStatusError:
    response = types.SimpleNamespace(
        status_code=400,
        headers={},
        request=types.SimpleNamespace(),
    )
    return APIStatusError(message, response=response, body=None)


def test_retry_suite() -> None:
    print("[可重试错误：重试后成功]")
    notices: list[str] = []
    llm = make_llm_with_script([conn_error(), conn_error(), make_stream("世界")])
    msg = llm.chat_stream([{"role": "user", "content": "hi"}],
                          on_notice=notices.append)
    check("第三次成功返回内容", msg["content"] == "世界", msg.get("content"))
    check("共调用 3 次", llm._call_count["n"] == 3, llm._call_count["n"])  # type: ignore[attr-defined]
    check("每次重试都有提示", len(notices) == 2, notices)
    check("usage 照常捕获", msg["_usage"]["total"] == 12, msg.get("_usage"))

    print("[可重试错误：超过上限放弃]")
    llm2 = make_llm_with_script([conn_error()])
    notices2: list[str] = []
    try:
        llm2.chat_stream([{"role": "user", "content": "hi"}], on_notice=notices2.append)
        check("重试耗尽应抛 LLMError", False, "no raise")
    except LLMError as exc:
        check("重试耗尽抛 LLMError", True)
        check("共尝试 1+3=4 次", llm2._call_count["n"] == 4, llm2._call_count["n"])  # type: ignore[attr-defined]
        check("提示了 3 次", len(notices2) == 3, notices2)

    print("[致命错误：400 不重试]")
    llm3 = make_llm_with_script([status_400("invalid messages format")])
    notices3: list[str] = []
    try:
        llm3.chat_stream([{"role": "user", "content": "hi"}], on_notice=notices3.append)
        check("400 应立即抛", False, "no raise")
    except LLMError:
        check("400 立即抛 LLMError", True)
        check("400 只调用 1 次", llm3._call_count["n"] == 1)  # type: ignore[attr-defined]
        check("400 无重试提示", not notices3)

    print("[上下文溢出：抛 ContextOverflowError]")
    llm4 = make_llm_with_script(
        [status_400("This model's maximum context length is 131072 tokens")]
    )
    try:
        llm4.chat_stream([{"role": "user", "content": "hi"}])
        check("溢出应抛 ContextOverflowError", False, "no raise")
    except ContextOverflowError:
        check("溢出抛 ContextOverflowError", True)
    except LLMError as exc:
        check("溢出抛 ContextOverflowError", False, f"got plain LLMError: {exc}")

    print("[summarize 重试]")
    good_resp = types.SimpleNamespace(choices=[types.SimpleNamespace(
        finish_reason="stop",
        message=types.SimpleNamespace(content="  摘要内容  "),
    )])
    llm5 = make_llm_with_script([conn_error(), good_resp])
    text = llm5.summarize("prompt")
    check("summarize 重试后成功", text == "摘要内容", text)
    llm6 = make_llm_with_script([status_400("bad request")])
    try:
        llm6.summarize("prompt")
        check("summarize 400 不重试", False, "no raise")
    except LLMError:
        check("summarize 400 立即失败", True)
        check("summarize 400 只调 1 次", llm6._call_count["n"] == 1)  # type: ignore[attr-defined]

    print("[loop 层：溢出恢复 = 压缩 + 重试一次]")
    workdir = ROOT / "sandbox" / "retry_test"
    workdir.mkdir(parents=True, exist_ok=True)
    reg = build_tools(Config(workdir=workdir, enable_shell=False))

    class OnceOverflowLLM:
        def __init__(self):
            self.calls = 0
        def chat_stream(self, messages, tools=None, on_delta=None, on_notice=None):
            self.calls += 1
            if self.calls == 1:
                raise ContextOverflowError("maximum context length exceeded")
            return {"role": "assistant", "content": "恢复成功",
                    "_finish_reason": "stop", "_usage": {"input": 1, "output": 2, "total": 3}}
        def summarize(self, prompt):
            return "## Goal\n目标"

    fake = OnceOverflowLLM()
    notices4: list[str] = []
    agent = Agent(Config(workdir=workdir, enable_shell=False), reg, llm=fake,
                  on_event=lambda e, p: notices4.append(p["message"]) if e == "notice" else None)
    agent.ctx.add({"role": "user", "content": "做件事"})
    agent.ctx.add({"role": "assistant", "content": "第一步"})
    answer = agent.run("继续")
    check("溢出后重试本轮并成功", answer == "恢复成功", answer)
    check("chat_stream 被调 2 次", fake.calls == 2, fake.calls)
    check("有溢出提示", any("压缩" in n for n in notices4), notices4)


