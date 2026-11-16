"""离线测试：上下文管理（切点/序列化/文件清单/摘要/锚点记账）。

不需要 API key：LLM 用假实现（只实现 summarize 接口）。
    uv run pytest
"""
from __future__ import annotations

from mini_agent.client import LLM, LLMError, context_tokens, request_size  # noqa: E402
from mini_agent.tokens import count_tokens_json  # noqa: E402
from mini_agent.compaction import (  # noqa: E402
    append_file_lists,
    build_summary_prompt,
    extract_file_operations,
    find_cut_point,
    looks_like_bad_summary,
    serialize_for_summary,
)
from mini_agent.config import Config  # noqa: E402
from mini_agent.context import COMPACTION_MARKER, Context  # noqa: E402



def check(name: str, cond: bool, detail: str = "") -> None:
    assert cond, f"{name}  {detail}"



def make_pair(content_len: int = 100) -> list[dict]:
    """一轮 toolCall+toolResult 配对。"""
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "Read", "arguments": '{"file_path": "a.py"}'}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * content_len},
    ]


class FakeLLM:
    """假 LLM：按队列返回摘要，或抛 LLMError 模拟失败/截断。"""

    def __init__(self, results: list):
        self.results = list(results)
        self.prompts: list[str] = []

    def summarize(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.results:
            raise LLMError("no more canned results")
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def history(n_turns: int, filler_len: int = 200) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"第{i}条 " + "字" * filler_len})
        msgs.extend(make_pair(filler_len))
        msgs.append({"role": "assistant", "content": f"完成第{i}条 " + "字" * filler_len})
    return msgs


def test_context_suite() -> None:
    print("[find_cut_point]")
    msgs = history(3)
    cut = find_cut_point(msgs, 10**9)
    check("预算充足不切", cut == len(msgs), f"cut={cut}")
    cut = find_cut_point(msgs, 10)
    check("预算紧张产生切点", 1 < cut < len(msgs), f"cut={cut}")
    check("不切到 system", cut >= 1)

    # 配对完整性：对多种预算，tail 首条绝不孤立
    ok_pairs = True
    for budget in (10, 50, 200, 500, 1000):
        c = find_cut_point(msgs, budget)
        tail = msgs[c:]
        if tail and tail[0].get("role") == "tool":
            ok_pairs = False
        # assistant(tool_calls) 后必须紧跟其 tool 结果
        for i, m in enumerate(tail):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                ids = {tc["id"] for tc in m["tool_calls"]}
                got = {t.get("tool_call_id") for t in tail[i + 1: i + 1 + len(ids)]}
                if not ids.issubset(got):
                    ok_pairs = False
    check("任意预算下 toolCall/toolResult 配对完整", ok_pairs)

    print("[serialize_for_summary]")
    serialized = serialize_for_summary([
        {"role": "system", "content": "sysprompt"},
        {"role": "user", "content": "需求内容"},
        {"role": "assistant", "content": "回答", "reasoning_content": "思维链不应出现"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c9", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}},
        ]},
        {"role": "tool", "tool_call_id": "c9", "content": "y" * 5000},
    ])
    check("system 不进序列化", "sysprompt" not in serialized)
    check("思维链被丢弃", "思维链不应出现" not in serialized)
    check("标签格式", "[User]: 需求内容" in serialized and "[Assistant tool calls]:" in serialized)
    check("工具结果 2000 字符截断", "截断，原 5000 字符" in serialized)

    print("[extract_file_operations]")
    rf, mf = extract_file_operations([
        {"role": "assistant", "tool_calls": [
            {"id": "a", "function": {"name": "Read", "arguments": '{"file_path": "r1.py"}'}},
            {"id": "b", "function": {"name": "Grep", "arguments": '{"path": "src", "pattern": "x"}'}},
            {"id": "c", "function": {"name": "Write", "arguments": '{"file_path": "w1.py"}'}},
            {"id": "d", "function": {"name": "Edit", "arguments": '{broken json'}},
        ]},
    ])
    check("读取/修改分类正确", rf == ["r1.py", "src"] and mf == ["w1.py"], f"{rf} {mf}")

    print("[build_summary_prompt]")
    p = build_summary_prompt("对话文本", previous_summary="旧摘要ABC",
                             read_files=["a.py"], modified_files=["b.py"])
    check("6 节模板", all(s in p for s in ("## Goal", "## Progress", "## Next Steps", "## Critical Context")))
    check("更新模式含旧摘要", "旧摘要ABC" in p)
    check("文件清单入 prompt", "<read-files>" in p and "a.py" in p)
    p2 = build_summary_prompt("对话文本")
    check("无旧摘要不带更新段", "<旧摘要>" not in p2)

    print("[append_file_lists / looks_like_bad_summary]")
    s = append_file_lists("## Goal\n内容", ["r.py"], ["w.py"])
    check("文件清单附加在摘要尾", s.endswith("</modified-files>") and "<read-files>" in s)
    check("拒收空摘要", looks_like_bad_summary(""))
    check("拒收 ERROR 开头", looks_like_bad_summary("ERROR: something"))
    check("拒收续写对话", looks_like_bad_summary("[User]: 你好"))
    check("拒收过短", looks_like_bad_summary("太短"))
    check("放行正常摘要", not looks_like_bad_summary("## Goal\n修复登录Bug " * 30))

    print("[context_tokens 锚点记账]")
    anchored = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "答", "_usage": {"input": 1000, "output": 50, "total": 1050}},
        {"role": "user", "content": "后续 " * 100},
    ]
    t = context_tokens(anchored)
    check("锚点 = usage.total + 之后增量", 1050 < t < 1050 + 400, f"t={t}")
    no_anchor = [{"role": "user", "content": "abcd " * 100}]
    check("无锚点退回启发式", context_tokens(no_anchor) > 0)
    stale = anchored + [{"role": "user", "content": f"{COMPACTION_MARKER} 摘要……\n\n[以上为历史摘要]"}]
    check("锚点后出现压缩标记 → 锚点作废", context_tokens(stale) != t, f"{context_tokens(stale)} vs {t}")

    print("[Context.compact 端到端（假 LLM）]")
    cfg = Config(context_window=4000, compact_ratio=0.5, keep_recent_tokens=800)
    ctx = Context(cfg)
    ctx.messages = history(8, filler_len=300)

    llm = FakeLLM(["## Goal\n修Bug " * 40])
    report = ctx.compact(llm)
    check("压缩执行", report is not None and report.summarised, report)
    check("摘要替换头部", ctx.messages[1]["content"].startswith(COMPACTION_MARKER))
    check("摘要含文件清单", "<read-files>" in ctx.messages[1]["content"], ctx.messages[1]["content"][:200])
    check("校验无问题", not ctx.validate(), ctx.validate())
    check("prompt 用了结构化模板", "## Goal" in llm.prompts[0])

    # 第二次压缩：走更新模式（旧摘要进 prompt）
    for _ in range(6):
        ctx.add({"role": "user", "content": "更多 " * 300})
        ctx.add({"role": "assistant", "content": "答 " * 300})
    llm2 = FakeLLM(["## Goal\n继续修 " * 40])
    ctx.compact(llm2)
    check("二次压缩走更新模式", "<旧摘要>" in llm2.prompts[0])

    # 摘要失败 → 机械裁剪兜底
    cfg3 = Config(context_window=4000, compact_ratio=0.5, keep_recent_tokens=800)
    ctx3 = Context(cfg3)
    ctx3.messages = history(8, filler_len=300)
    n_before = len(ctx3.messages)
    llm3 = FakeLLM([LLMError("summary truncated (finish_reason=length)")])
    report3 = ctx3.compact(llm3)
    check("摘要失败降级机械裁剪", report3 is not None and not report3.summarised)
    check("机械裁剪真的丢了消息", len(ctx3.messages) < n_before)
    check("机械裁剪后配对完整", not ctx3.validate(), ctx3.validate())

    # 坏摘要（续写对话）被拒收
    cfg4 = Config(context_window=4000, compact_ratio=0.5, keep_recent_tokens=800)
    ctx4 = Context(cfg4)
    ctx4.messages = history(8, filler_len=300)
    llm4 = FakeLLM(["[User]: 让我续写 " * 30])
    report4 = ctx4.compact(llm4)
    check("坏摘要被质量闸拒收", report4 is not None and not report4.summarised)

    print("[第二次压缩失败不回滚第一次摘要]")
    cfg_keep = Config(context_window=500, compact_ratio=0.9, keep_recent_tokens=100000)
    ctx_keep = Context(cfg_keep)
    ctx_keep.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "原始问题MARKER"},
        {"role": "assistant", "content": "原始回答"},
    ]
    good = "## Goal\nUNIQUE_SUMMARY_TOKEN " + ("进展 " * 30)
    huge = "## Goal\n" + ("超" * 5000)
    ctx_keep.compact(FakeLLM([good, huge]), force=True, max_attempts=2)
    kept = "\n".join(str(m.get("content") or "") for m in ctx_keep.messages)
    check("第一次摘要还在", "UNIQUE_SUMMARY_TOKEN" in kept, kept[:240])
    check("没有退回压缩前的原文", "原始问题MARKER" not in kept, kept[:240])

    print("[单条超长消息可截断]")
    cfg_big = Config(context_window=4000, compact_ratio=0.5, keep_recent_tokens=20000)
    ctx_big = Context(cfg_big)
    ctx_big.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "大" * 20000, "reasoning_content": "思" * 5000},
    ]
    report_big = ctx_big.compact(FakeLLM([]), force=True)
    check("超长消息产生了压缩报告", report_big is not None)
    check("思维链收成空格", ctx_big.messages[1].get("reasoning_content") == " ")
    check("正文被截断", "已截断" in ctx_big.messages[1]["content"])

    print("[request_size 不重复计算 schema / 旧思维链]")
    tools = [{"type": "function", "function": {
        "name": "Read", "description": "d" * 400, "parameters": {"type": "object"},
    }}]
    anchored_only = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "答", "_usage": {"input": 1000, "output": 50, "total": 1050}},
    ]
    check("有锚点不再加一遍 schema",
          request_size(anchored_only, tools) == context_tokens(anchored_only))
    plain = [{"role": "user", "content": "hi"}]
    check("无锚点加一次 schema",
          request_size(plain, tools) == context_tokens(plain) + count_tokens_json(tools))
    long_reason = "思" * 800
    reasoned = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "旧", "reasoning_content": long_reason,
         "_usage": {"total": 5000, "input": 100, "output": 4900}},
        {"role": "user", "content": "继续"},
        {"role": "assistant", "content": "新", "reasoning_content": "短"},
    ]
    full = context_tokens(reasoned)
    collapsed = context_tokens(reasoned, collapse_old_reasoning=True)
    check("旧思维链不再按全文计入", collapsed < full - 500, f"{collapsed} vs {full}")
    wire = LLM(Config(api_key="test", model="deepseek-v4"))._wire_messages(reasoned)
    check("线上只保留最后一条思维链", wire[1]["reasoning_content"] == " " and wire[3]["reasoning_content"] == "短")
    check("会话里的思维链原文还在", reasoned[1]["reasoning_content"] == long_reason)
    other = LLM(Config(api_key="test", model="gpt-4o"))._wire_messages(reasoned)
    check("非 must-echo 模型不改思维链", other[1]["reasoning_content"] == long_reason)

    print("[keep_recent_tokens 自适应]")
    small = Config(context_window=8000, compact_ratio=0.5, keep_recent_tokens=20000)
    c = Context(small)
    check("小窗口收缩保留预算", c._keep_tokens() <= small.compact_threshold // 2)


