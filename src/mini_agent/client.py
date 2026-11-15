"""LLM 客户端：OpenAI 兼容接口，消息与工具调用统一转成纯 dict。

要点：
- 丢弃 reasoning_content（部分推理模型会返回，回传会报 400）。
- tool_calls 转成可序列化 dict，便于存历史。
"""
from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI

from .config import Config
from .tokens import count_tokens_json


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
                raise LLMError(f"{type(exc).__name__}: {exc}") from exc

            content: list[str] = []
            reasoning: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            finish: str | None = None

            try:
                for chunk in stream:
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
        return msg

    def summarize(self, history: list[dict]) -> str:
        """把一段历史压成交接笔记。失败抛 LLMError，调用方处理兜底。"""
        body = json.dumps(history, ensure_ascii=False)
        prompt = (
            "请把下面的对话压缩成交接笔记，不超过 400 字。必须保留：\n"
            "1) 已经完成的事情和改过的文件路径；\n"
            "2) 尚未完成的待办；\n"
            "3) 关键事实、参数、用户明确表达的偏好；\n"
            "4) 正在进行的工具调用链（如果被截断）。\n"
            "只写对话里真实出现过的内容，不要编造，不要寒暄。\n\n"
            f"对话记录：\n{body}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1200,
            )
        except Exception as exc:
            raise LLMError(f"summary failed: {type(exc).__name__}: {exc}") from exc

        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise LLMError("summary returned empty content")
        return text


def request_size(messages: list[dict], tools: list[dict] | None = None) -> int:
    """一次请求的估算 token 总量（含工具 schema）。"""
    total = count_tokens_json(messages)
    if tools:
        total += count_tokens_json(tools)
    return total
