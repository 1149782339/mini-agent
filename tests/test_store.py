"""压缩后原文仍在库里；发给模型的是摘要视图。"""
from __future__ import annotations

from pathlib import Path

from mini_agent.config import Config, SYSTEM_PROMPT
from mini_agent.context import COMPACTION_MARKER, Context
from mini_agent.loop import Agent
from mini_agent.store import SessionStore
from mini_agent.tools import build_tools

from test_context import FakeLLM, history


def test_compaction_keeps_originals_and_builds_view(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    sid = store.create(title="t", model="m", workdir=str(tmp_path))

    original = history(8, filler_len=300)
    for msg in original:
        if msg.get("role") == "system":
            continue
        store.append(sid, msg)

    raw_before = store.message_count(sid)
    assert raw_before == len(original) - 1  # system 通常不入库

    cfg = Config(context_window=4000, compact_ratio=0.5, keep_recent_tokens=800)
    ctx = Context(cfg)
    # 带上 _seq，便于 record_compaction 定位保留起点
    ctx.messages = [{"role": "system", "content": original[0]["content"]}, *store.load(sid)]

    summary = "## Goal\n修Bug " * 40
    llm = FakeLLM([summary, summary, summary])
    report = ctx.compact(llm)
    assert report is not None and report.summarised
    assert ctx.messages[1]["content"].startswith(COMPACTION_MARKER)
    assert not ctx.validate()

    store.record_compaction(
        sid,
        ctx.messages,
        before_tokens=report.before_tokens,
        after_tokens=report.after_tokens,
        dropped_messages=report.dropped_messages,
        summarised=report.summarised,
    )

    assert store.message_count(sid) == raw_before, "压缩不得删除原文"
    loaded = store.load(sid)
    assert any("第0条" in str(m.get("content") or "") for m in loaded)

    view = store.load_view(sid, system_prompt="sys")
    assert view[0]["role"] == "system"
    assert view[1]["content"].startswith(COMPACTION_MARKER)
    assert len(view) < raw_before + 1
    # 视图里的 tool 消息都能对上 tool_call
    ctx_view = Context(cfg)
    ctx_view.messages = view
    assert not ctx_view.validate(), ctx_view.validate()


def test_resume_loads_compacted_view_not_full_transcript(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    sid = store.create(model="m", workdir=str(tmp_path))
    cfg = Config(
        context_window=20000,
        compact_ratio=0.5,
        keep_recent_tokens=800,
        workdir=tmp_path,
        enable_shell=False,
        enable_subagent=False,
    )
    summary = "## Goal\n修Bug " * 40
    llm = FakeLLM([summary, summary, summary])
    agent = Agent(cfg, build_tools(cfg), llm=llm, store=store, session_id=sid)

    for msg in history(8, filler_len=300)[1:]:
        agent._add(msg)

    raw_n = store.message_count(sid)
    report = agent.compact_now(force=True)
    assert report is not None and report.summarised
    assert agent.ctx.messages[1]["content"].startswith(COMPACTION_MARKER)
    view_n = len(agent.ctx.messages)
    assert store.message_count(sid) == raw_n
    assert view_n < raw_n

    # 再追加一条，原文和视图都要能看到
    agent._add({"role": "user", "content": "压缩后继续"})
    assert store.message_count(sid) == raw_n + 1
    assert agent.ctx.messages[-1]["content"] == "压缩后继续"

    resumed = Agent(cfg, build_tools(cfg), llm=FakeLLM([]), store=store, session_id=sid)
    assert resumed.ctx.messages[1]["content"].startswith(COMPACTION_MARKER)
    assert resumed.ctx.messages[-1]["content"] == "压缩后继续"
    assert len(resumed.ctx.messages) == len(agent.ctx.messages)
    assert any("第0条" in str(m.get("content") or "") for m in store.load(sid))
    # resume 后的视图仍带 reasoning_content（若原文有）
    assert not resumed.ctx.validate(), resumed.ctx.validate()


def test_mechanical_fallback_also_keeps_originals(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    sid = store.create(model="m", workdir=str(tmp_path))
    cfg = Config(
        context_window=4000,
        compact_ratio=0.5,
        keep_recent_tokens=800,
        workdir=tmp_path,
        enable_shell=False,
        enable_subagent=False,
    )
    from mini_agent.client import LLMError

    agent = Agent(cfg, build_tools(cfg), llm=FakeLLM([LLMError("truncated")]), store=store, session_id=sid)
    for msg in history(8, filler_len=300)[1:]:
        agent._add(msg)
    raw_n = store.message_count(sid)
    report = agent.compact_now(force=True)
    assert report is not None and not report.summarised
    assert store.message_count(sid) == raw_n
    assert len(agent.ctx.messages) < raw_n
    assert not agent.ctx.validate(), agent.ctx.validate()

    resumed = Agent(cfg, build_tools(cfg), llm=FakeLLM([]), store=store, session_id=sid)
    assert len(resumed.ctx.messages) == len(agent.ctx.messages)
    assert not any(
        str(m.get("content") or "").startswith(COMPACTION_MARKER) for m in resumed.ctx.messages
    )


def test_clear_resets_view_but_keeps_originals(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    sid = store.create(model="m", workdir=str(tmp_path))
    store.append(sid, {"role": "user", "content": "旧问题"})
    store.append(sid, {"role": "assistant", "content": "旧回答", "reasoning_content": "思考过程"})
    store.mark_clear(sid)
    assert store.message_count(sid) == 2
    view = store.load_view(sid, system_prompt=SYSTEM_PROMPT)
    assert view == [{"role": "system", "content": SYSTEM_PROMPT}]
    assert store.current_user_message_count(sid, system_prompt=SYSTEM_PROMPT) == 0

    store.append(sid, {"role": "user", "content": "新问题"})
    view = store.load_view(sid, system_prompt=SYSTEM_PROMPT)
    assert [m.get("content") for m in view if m.get("role") != "system"] == ["新问题"]
    raw = store.load(sid)
    assert [m["content"] for m in raw if m.get("role") in {"user", "assistant"}] == [
        "旧问题",
        "旧回答",
        "新问题",
    ]
    assert raw[1]["reasoning_content"] == "思考过程"


def test_resume_restores_reasoning_content(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    sid = store.create(model="m", workdir=str(tmp_path))
    store.append(sid, {"role": "user", "content": "问"})
    store.append(
        sid,
        {
            "role": "assistant",
            "content": "答",
            "reasoning_content": "完整思维链",
            "_usage": {"total": 10, "input": 4, "output": 6},
        },
    )
    cfg = Config(workdir=tmp_path, enable_shell=False, enable_subagent=False)
    agent = Agent(cfg, build_tools(cfg), llm=FakeLLM([]), store=store, session_id=sid)
    assert agent.ctx.messages[-1]["reasoning_content"] == "完整思维链"
    assert agent.ctx.messages[-1]["_usage"]["total"] == 10
