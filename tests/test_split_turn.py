"""拆轮双摘要：切点落在一轮中间时，历史和当前轮前缀分开压缩。"""
from __future__ import annotations

import json
from pathlib import Path

from mini_agent.client import LLMError
from mini_agent.compaction import (
    build_turn_prefix_prompt,
    find_cut_point,
    merge_split_summaries,
    prepare_compaction,
)
from mini_agent.config import Config
from mini_agent.context import COMPACTION_MARKER, Context
from mini_agent.loop import Agent
from mini_agent.store import SessionStore
from mini_agent.tools import build_tools

from test_context import FakeLLM


HISTORY_SUMMARY = "## Goal\n完成历史任务 HISTORY_SUMMARY_TOKEN " + ("进展 " * 30)
PREFIX_SUMMARY = "## Original Request\n请修复登录 PREFIX_SUMMARY_TOKEN " + ("上下文 " * 20)
CURRENT_REQUEST = "请修复登录 CURRENT_REQUEST "
CURRENT_SUFFIX = "进行中 CURRENT_SUFFIX "
KEEP_REASONING = "完整思维链 KEEP_REASONING"


def mid_turn_messages() -> list[dict]:
    """前面几轮较短，当前轮带多组大工具结果，便于把切点放在本轮中间。"""
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(3):
        msgs.append({"role": "user", "content": f"历史任务{i} OLD_TURN_{i} " + "甲" * 120})
        msgs.append({"role": "assistant", "content": f"已完成历史任务{i} " + "乙" * 120})
    msgs.append({"role": "user", "content": CURRENT_REQUEST + "丙" * 40})
    for j in range(4):
        cid = f"cur{j}"
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": cid,
                "function": {"name": "Read", "arguments": json.dumps({"file_path": f"src/f{j}.py"})},
            }],
        })
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "T" * 1200})
    msgs.append({
        "role": "assistant",
        "content": CURRENT_SUFFIX + "丁" * 80,
        "reasoning_content": KEEP_REASONING,
    })
    return msgs


def mid_turn_cfg(keep_recent_tokens: int, context_window: int = 8000) -> Config:
    return Config(
        context_window=context_window,
        compact_ratio=0.5,
        keep_recent_tokens=keep_recent_tokens,
    )


def mid_turn_plan(keep: int):
    return prepare_compaction(mid_turn_messages(), keep)


def test_prepare_compaction_mid_turn_vs_boundary() -> None:
    split = mid_turn_plan(400)
    assert split.is_split_turn
    assert split.turn_start > 1
    msgs = mid_turn_messages()
    assert msgs[split.turn_start]["role"] == "user"
    assert CURRENT_REQUEST in msgs[split.turn_start]["content"]
    assert split.cut > split.turn_start
    assert msgs[split.cut].get("role") != "user"
    # 历史不含当前请求，后缀不含当前请求原文（请求在前缀里）
    history = msgs[1:split.turn_start]
    prefix = msgs[split.turn_start:split.cut]
    tail = msgs[split.cut:]
    assert any("OLD_TURN_0" in str(m.get("content") or "") for m in history)
    assert any(CURRENT_REQUEST in str(m.get("content") or "") for m in prefix)
    assert all(CURRENT_REQUEST not in str(m.get("content") or "") for m in tail)
    assert any(CURRENT_SUFFIX in str(m.get("content") or "") for m in tail)

    boundary = mid_turn_plan(1600)
    assert not boundary.is_split_turn
    assert boundary.turn_start == -1
    assert 1 < boundary.cut < len(msgs)
    assert msgs[boundary.cut]["role"] == "user"


def test_prepare_compaction_does_not_split_tool_pairs() -> None:
    msgs = mid_turn_messages()
    for budget in (200, 400, 800, 1200, 1600, 2000):
        plan = prepare_compaction(msgs, budget)
        cut = find_cut_point(msgs, budget)
        assert plan.cut == cut
        tail = msgs[cut:]
        if tail:
            assert tail[0].get("role") != "tool"
        if plan.is_split_turn:
            prefix = msgs[plan.turn_start:cut]
            if prefix:
                assert prefix[-1].get("role") != "assistant" or not prefix[-1].get("tool_calls")
        # 任意一侧都不得把 tool_call 和结果劈开
        for part in (msgs[1:cut], tail):
            open_ids: set[str] = set()
            for msg in part:
                if msg.get("role") == "assistant":
                    for call in msg.get("tool_calls") or []:
                        open_ids.add(call["id"])
                elif msg.get("role") == "tool":
                    call_id = msg.get("tool_call_id")
                    assert call_id in open_ids, f"budget={budget} dangling tool {call_id}"
                    open_ids.discard(call_id)
            assert not open_ids, f"budget={budget} missing results {open_ids}"


def test_mid_turn_compact_produces_two_summaries_and_keeps_suffix() -> None:
    msgs = mid_turn_messages()
    plan = prepare_compaction(msgs, 400)
    assert plan.is_split_turn
    suffix = [dict(m) for m in msgs[plan.cut:]]

    cfg = mid_turn_cfg(400)
    ctx = Context(cfg)
    ctx.messages = msgs
    llm = FakeLLM([HISTORY_SUMMARY, PREFIX_SUMMARY])
    report = ctx.compact(llm, force=True)

    assert report is not None and report.summarised and report.is_split_turn
    assert len(llm.prompts) == 2
    # 历史 prompt 是六节模板，且含旧轮、不含当前请求
    assert "## Goal" in llm.prompts[0]
    assert "OLD_TURN_0" in llm.prompts[0]
    assert CURRENT_REQUEST not in llm.prompts[0]
    assert CURRENT_SUFFIX not in llm.prompts[0]
    # 前缀 prompt 是三节模板，含当前请求，不含更早轮和后缀
    assert "## Original Request" in llm.prompts[1]
    assert CURRENT_REQUEST in llm.prompts[1]
    assert "OLD_TURN_0" not in llm.prompts[1]
    assert CURRENT_SUFFIX not in llm.prompts[1]

    marker = ctx.messages[1]["content"]
    assert marker.startswith(COMPACTION_MARKER)
    assert "HISTORY_SUMMARY_TOKEN" in marker
    assert "PREFIX_SUMMARY_TOKEN" in marker
    assert "**Turn Context (split turn):**" in marker
    assert "<read-files>" in marker and "src/f0.py" in marker

    # 后缀仍是原文，含思维链
    kept = ctx.messages[2:]
    assert [m.get("content") for m in kept] == [m.get("content") for m in suffix]
    assert any(CURRENT_SUFFIX in str(m.get("content") or "") for m in kept)
    assert kept[-1].get("reasoning_content") == KEEP_REASONING
    assert all(CURRENT_REQUEST not in str(m.get("content") or "") for m in kept)
    assert all("OLD_TURN_0" not in str(m.get("content") or "") for m in kept)
    assert not ctx.validate(), ctx.validate()


def test_turn_boundary_cut_stays_single_summary() -> None:
    msgs = mid_turn_messages()
    plan = prepare_compaction(msgs, 1600)
    assert not plan.is_split_turn

    cfg = mid_turn_cfg(1600, context_window=10000)
    ctx = Context(cfg)
    ctx.messages = msgs
    llm = FakeLLM([HISTORY_SUMMARY, PREFIX_SUMMARY])
    report = ctx.compact(llm, force=True)

    assert report is not None and report.summarised
    assert not report.is_split_turn
    assert len(llm.prompts) == 1
    assert "## Goal" in llm.prompts[0]
    assert "## Original Request" not in llm.prompts[0]
    marker = ctx.messages[1]["content"]
    assert "HISTORY_SUMMARY_TOKEN" in marker
    assert "PREFIX_SUMMARY_TOKEN" not in marker
    assert "Turn Context (split turn)" not in marker
    # 切在轮边界时，被保留的原文里仍能看到当前请求
    assert any(CURRENT_REQUEST in str(m.get("content") or "") for m in ctx.messages[2:])
    assert not ctx.validate(), ctx.validate()


def test_bad_or_failed_summaries_fall_back_safely() -> None:
    def _run(results: list) -> tuple:
        ctx = Context(mid_turn_cfg(400))
        ctx.messages = mid_turn_messages()
        before = [dict(m) for m in ctx.messages]
        llm = FakeLLM(results)
        report = ctx.compact(llm, force=True)
        return ctx, llm, report, before

    # 历史摘要失败：不写前缀，不留下半成品摘要
    ctx, llm, report, before = _run([LLMError("history failed"), PREFIX_SUMMARY])
    assert report is not None and not report.summarised
    assert len(llm.prompts) == 1
    assert not any(
        str(m.get("content") or "").startswith(COMPACTION_MARKER) for m in ctx.messages
    )
    assert "Turn Context (split turn)" not in "\n".join(
        str(m.get("content") or "") for m in ctx.messages
    )
    assert len(ctx.messages) < len(before)
    assert not ctx.validate(), ctx.validate()

    # 历史通过、前缀被质量闸拒收：同样整段回退
    ctx, llm, report, before = _run([HISTORY_SUMMARY, "[User]: 让我续写 " * 30])
    assert report is not None and not report.summarised
    assert len(llm.prompts) == 2
    joined = "\n".join(str(m.get("content") or "") for m in ctx.messages)
    assert "HISTORY_SUMMARY_TOKEN" not in joined
    assert "Turn Context (split turn)" not in joined
    assert not any(str(m.get("content") or "").startswith(COMPACTION_MARKER) for m in ctx.messages)
    assert not ctx.validate(), ctx.validate()

    # 过短的前缀摘要也被拒收
    ctx, llm, report, _ = _run([HISTORY_SUMMARY, "太短"])
    assert report is not None and not report.summarised
    assert not any(str(m.get("content") or "").startswith(COMPACTION_MARKER) for m in ctx.messages)


def test_split_turn_keeps_originals_in_sqlite(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    sid = store.create(title="split", model="m", workdir=str(tmp_path))
    original = mid_turn_messages()
    for msg in original:
        if msg.get("role") == "system":
            continue
        store.append(sid, msg)
    raw_n = store.message_count(sid)

    cfg = mid_turn_cfg(400)
    cfg.workdir = tmp_path
    cfg.enable_shell = False
    cfg.enable_subagent = False
    llm = FakeLLM([HISTORY_SUMMARY, PREFIX_SUMMARY])
    agent = Agent(cfg, build_tools(cfg), llm=llm, store=store, session_id=sid)

    report = agent.compact_now(force=True)
    assert report is not None and report.summarised and report.is_split_turn
    assert store.message_count(sid) == raw_n, "压缩不得删除原文"

    loaded = store.load(sid)
    assert any(CURRENT_REQUEST in str(m.get("content") or "") for m in loaded)
    assert any("OLD_TURN_0" in str(m.get("content") or "") for m in loaded)
    assert any(m.get("reasoning_content") == KEEP_REASONING for m in loaded)

    view = store.load_view(sid, system_prompt="sys")
    assert view[1]["content"].startswith(COMPACTION_MARKER)
    assert "Turn Context (split turn)" in view[1]["content"]
    assert any(CURRENT_SUFFIX in str(m.get("content") or "") for m in view)
    assert all(CURRENT_REQUEST not in str(m.get("content") or "") for m in view[2:])
    assert all("OLD_TURN_0" not in str(m.get("content") or "") for m in view[2:])
    # 后缀思维链在视图里
    assert view[-1].get("reasoning_content") == KEEP_REASONING

    resumed = Agent(cfg, build_tools(cfg), llm=FakeLLM([]), store=store, session_id=sid)
    assert resumed.ctx.messages[1]["content"].startswith(COMPACTION_MARKER)
    assert "Turn Context (split turn)" in resumed.ctx.messages[1]["content"]
    assert resumed.ctx.messages[-1].get("reasoning_content") == KEEP_REASONING
    assert len(resumed.ctx.messages) == len(agent.ctx.messages)
    assert not resumed.ctx.validate(), resumed.ctx.validate()


def test_merge_and_prefix_prompt_helpers() -> None:
    prompt = build_turn_prefix_prompt("[User]: 请修复登录")
    assert "## Original Request" in prompt
    assert "## Progress So Far" in prompt
    assert "## Context Needed to Continue" in prompt
    assert "请修复登录" in prompt
    merged = merge_split_summaries("历史摘要正文", "前缀摘要正文")
    assert merged.startswith("历史摘要正文")
    assert "**Turn Context (split turn):**" in merged
    assert merged.endswith("前缀摘要正文")
    assert merge_split_summaries("  ", "前缀").startswith("无更早历史。")
