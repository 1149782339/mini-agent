"""LLM 客户端：OpenAI 兼容接口，消息与工具调用统一转成纯 dict。

要点：
- 丢弃 reasoning_content（部分推理模型会返回，回传会报 400）。
- tool_calls 转成可序列化 dict，便于存历史。
"""
from __future__ import annotations

from typing import Any, Callable

from openai import OpenAI

from .config import Config
from .tokens import count_tokens_json, message_tokens


def _field(obj: Any, name: str):
    """取字段：普通属性取不到时，回退到 pydantic 的 model_extra。

    openai SDK 不认 reasoning_content 这类 provider 扩展字段，
    流式 delta 上会落到 model_extra 里。
    """
    value = getattr(obj, name, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name)
    return None


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=120.0,
            max_retries=2,
        )

    # ---------- 内部转换 ----------
    @staticmethod
    def _clean_assistant(message: Any) -> dict:
        out: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        # 按 Hermes 对 must-echo provider 的处理：思维链随历史回传
        reasoning = _field(message, "reasoning_content")
        if reasoning:
            out["reasoning_content"] = reasoning
        calls = getattr(message, "tool_calls", None)
        if calls:
            out["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments or "{}",
                    },
                }
                for c in calls
            ]
        return out

    # ---------- 公共调用 ----------
    @property
    def must_echo_reasoning(self) -> bool:
        """这些 provider 要求 assistant 历史带 reasoning_content，否则 400。"""
        name = (self.cfg.model or "").lower()
        return any(k in name for k in ("deepseek", "kimi", "minimax", "mimo"))

    def _wire_messages(self, messages: list[dict]) -> list[dict]:
        """发给 API 的消息：去掉私有字段；must-echo provider 每条 assistant 补 pad。

        DeepSeek V4 拒绝空字符串，所以缺失时补一个空格而不是空串。
        """
        out: list[dict] = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if not k.startswith("_")}
            if self.must_echo_reasoning and clean.get("role") == "assistant":
                if not clean.get("reasoning_content"):
                    clean["reasoning_content"] = " "
            out.append(clean)
        return out

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_delta: Callable[[str, str], None] | None = None,
    ) -> dict:
        """流式请求：边收边回调，最后返回拼装完整的消息。

        on_delta(kind, text)：kind 为 "reasoning" 或 "content"。
        tool_calls 是分片下发的，按 index 聚合 name/arguments。
        """
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": self._wire_messages(messages),
            "temperature": self.cfg.temperature,
            "stream": True,
            # 让最后一个流式 chunk 带回 usage（Pi/Claude-Code 的 token 锚点记账
            # 都靠它）；个别 OpenAI 兼容端点不认这个参数，失败时退回去重试一次
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # 思维链计入 completion_tokens：预算被吃光时正文为空、finish_reason=length。
        # 若这轮还没有任何 delta 输出到终端，就加倍预算静默重试；
        # 已经吐过字则不重试，避免重复打印。
        budget = self.cfg.max_output_tokens
        ceiling = max(budget, 32768)
        emitted = False

        while True:
            kwargs["max_tokens"] = budget
            try:
                stream = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                msg = str(exc)
                if "stream_options" in msg and "stream_options" in kwargs:
                    # 端点不支持 include_usage：去掉参数重试，本轮退回纯估算
                    kwargs.pop("stream_options")
                    continue
                raise LLMError(f"{type(exc).__name__}: {exc}") from exc

            content: list[str] = []
            reasoning: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            finish: str | None = None
            usage: dict[str, int] | None = None

            try:
                for chunk in stream:
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        # include_usage 的收尾 chunk：choices 为空，只有 usage
                        prompt_toks = int(getattr(chunk_usage, "prompt_tokens", 0) or 0)
                        completion_toks = int(getattr(chunk_usage, "completion_tokens", 0) or 0)
                        usage = {
                            "input": prompt_toks,
                            "output": completion_toks,
                            "total": prompt_toks + completion_toks,
                        }
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish = choice.finish_reason
                    delta = choice.delta
                    if delta is None:
                        continue

                    piece = _field(delta, "reasoning_content")
                    if piece:
                        reasoning.append(piece)
                        emitted = True
                        if on_delta:
                            on_delta("reasoning", piece)

                    text = _field(delta, "content")
                    if text:
                        content.append(text)
                        emitted = True
                        if on_delta:
                            on_delta("content", text)

                    for tc in getattr(delta, "tool_calls", None) or []:
                        emitted = True
                        slot = calls.setdefault(tc.index or 0, {"id": "", "name": "", "arguments": ""})
                        if getattr(tc, "id", None):
                            slot["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot["name"] += fn.name
                            if getattr(fn, "arguments", None):
                                slot["arguments"] += fn.arguments
            except Exception as exc:
                raise LLMError(f"stream interrupted: {type(exc).__name__}: {exc}") from exc

            starved = finish == "length" and not content and not calls
            if not starved or emitted or budget >= ceiling:
                break
            budget = min(budget * 2, ceiling)

        msg: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
        if calls:
            msg["tool_calls"] = [
                {
                    "id": slot["id"] or f"call_{index}",
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"] or "{}"},
                }
                for index, slot in sorted(calls.items())
            ]
        msg["_finish_reason"] = finish
        if usage is not None:
            msg["_usage"] = usage  # 私有字段：发给 API 前由 _wire_messages 剥掉
        return msg

    def summarize(self, prompt: str) -> str:
        """执行一次摘要请求。prompt 由 compaction.build_summary_prompt 构建。

        Pi 的"拒收半成品"原则：finish_reason 为 length/content_filter 或正文为空，
        一律抛 LLMError，调用方走机械裁剪兜底——坏摘要永久污染会话比不摘要更糟。
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as exc:
            raise LLMError(f"summary failed: {type(exc).__name__}: {exc}") from exc

        choice = resp.choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish in ("length", "content_filter"):
            raise LLMError(f"summary truncated (finish_reason={finish})，拒收半成品")
        text = (choice.message.content or "").strip()
        if not text:
            raise LLMError("summary returned empty content")
        return text


def context_tokens(messages: list[dict]) -> int:
    """Pi/Claude-Code 同款的"锚点 + 粗估"上下文记账。

    从后往前找最近一条带真实 usage 的 assistant 消息（私有字段 _usage，
    chat_stream 在 include_usage 时填充）：锚点 total ≈ 当时请求的 input+output，
    即当时的上下文规模；锚点之后的消息用启发式逐条估算。
    锚点之前若发生过压缩（[COMPACTION] 标记在其后），锚点作废、全量估算——
    它统计过的上下文大部分已被丢进摘要。
    """
    anchor_total: int | None = None
    anchor_index = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant":
            usage = msg.get("_usage")
            if isinstance(usage, dict) and usage.get("total"):
                anchor_total = int(usage["total"])
                anchor_index = i
                break

    if anchor_total is None:
        return count_tokens(messages)

    # 锚点之后发生过压缩 → 锚点上下文已被摘要替换，作废
    for msg in messages[anchor_index + 1:]:
        content = str(msg.get("content") or "")
        if content.startswith("[COMPACTION]"):
            return count_tokens(messages)

    total = anchor_total
    for msg in messages[anchor_index + 1:]:
        total += message_tokens(msg)
    return total


def count_tokens(messages: list[dict]) -> int:
    """无锚点时的全量启发式估算。"""
    return sum(message_tokens(m) for m in messages)


def request_size(messages: list[dict], tools: list[dict] | None = None) -> int:
    """一次请求的估算 token 总量（含工具 schema）：锚点记账 + schema 估算。"""
    total = context_tokens(messages)
    if tools:
        total += count_tokens_json(tools)
    return total
