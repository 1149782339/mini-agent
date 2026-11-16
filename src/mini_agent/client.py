"""LLM 客户端：OpenAI 兼容接口，消息与工具调用统一转成纯 dict。

要点：
- must-echo provider（DeepSeek / Kimi / MiniMax）的 assistant 历史带回
  reasoning_content；缺失时补一个空格（DeepSeek V4 拒绝空字符串）。
- tool_calls 转成可序列化 dict，便于存历史。
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .config import Config
from .tokens import content_text, count_tokens_json, estimate_text_tokens, message_tokens

_MUST_ECHO_KEYS = ("deepseek", "kimi", "minimax", "mimo")
_REASONING_PAD = " "
_TOOL_IMAGE_OMITTED = "(tool image omitted: model does not support images)"
_ATTACHED_IMAGES = "Attached image(s) from tool result:"


def must_echo_model(model: str) -> bool:
    """这些 provider 要求 assistant 历史带非空 reasoning_content，否则 400。"""
    name = (model or "").lower()
    return any(k in name for k in _MUST_ECHO_KEYS)


def _last_assistant_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    return -1


def _fresh_anchor(messages: list[dict]) -> tuple[int, int] | None:
    """最近一条带 usage 的 assistant。其后出现过压缩标记则锚点作废。"""
    anchor_total: int | None = None
    anchor_index = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "assistant":
            continue
        usage = msg.get("_usage")
        if isinstance(usage, dict) and usage.get("total"):
            anchor_total = int(usage["total"])
            anchor_index = i
            break
    if anchor_total is None:
        return None
    for msg in messages[anchor_index + 1 :]:
        if content_text(msg.get("content")).startswith("[COMPACTION]"):
            return None
    return anchor_index, anchor_total


# ---------- 错误分类与退避（API 重试） ----------
# 场景：长任务跑到一半网络抖动/限流，旧行为直接 LLMError 终止本轮，用户手动重发。
# 分层：SDK(max_retries=2) 只管"请求阶段"；这层管"流中途断"和漏网的 429/5xx。

MAX_ERROR_ATTEMPTS = 3      # 本层重试上限（不含 SDK 内部的 2 次）
_OVERFLOW_MARKERS = (
    "context length", "maximum context length", "context window",
    "too long", "reduce the length", "exceed",
)


class LLMError(RuntimeError):
    pass


class ContextOverflowError(LLMError):
    """上下文超出模型窗口：不是普通 400，压缩后重试本轮才有意义。"""


def _classify_api_error(exc: Exception) -> str:
    """retry=可重试 / overflow=上下文溢出 / fatal=重试无意义。"""
    text = str(exc).lower()
    if isinstance(exc, APIStatusError):
        if exc.status_code == 400:
            if any(m in text for m in _OVERFLOW_MARKERS):
                return "overflow"
            return "fatal"  # 参数错误等，重试不会变好
        if exc.status_code == 429 or exc.status_code >= 500:
            return "retry"
        return "fatal"
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return "retry"
    return "retry"  # 未知异常（如流中途断的底层错误）倾向重试，次数上限兜底


def _retry_after_seconds(exc: Exception) -> float | None:
    """尊重 429 的 Retry-After 响应头。"""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                pass
    return None


def _backoff_seconds(attempt: int, exc: Exception) -> float:
    return _retry_after_seconds(exc) or min(2 ** attempt, 8) + random.uniform(0, 0.5)


def _image_blocks(content: object) -> list[dict]:
    if not isinstance(content, list):
        return []
    return [
        block for block in content
        if isinstance(block, dict) and block.get("type") == "image" and block.get("data")
    ]


def _image_urls(images: list[dict]) -> list[dict]:
    urls = []
    for img in images:
        mime = img.get("mimeType") or "image/png"
        urls.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{img['data']}"},
        })
    return urls


def _tool_wire_text(content: object, images: list[dict], *, attach: bool) -> str:
    """tool 消息发给接口时只留文字。不看图就换成占位句，看图则另附 user 消息。"""
    text = content_text(content)
    if images and not attach:
        return f"{text}\n{_TOOL_IMAGE_OMITTED}" if text else _TOOL_IMAGE_OMITTED
    if images and not text.strip():
        return "(see attached image)"
    return text or "(no tool output)"


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
        # must-echo provider：思维链随历史回传
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
        return must_echo_model(self.cfg.model)

    def _wire_messages(self, messages: list[dict]) -> list[dict]:
        """发给 API 的消息：去掉私有字段。

        must-echo provider 每条 assistant 都带 reasoning_content。只有最后一条
        保留原文；更早的收成一个空格（DeepSeek V4 拒绝空字符串）。全文仍留在
        会话库里，这里只改发出去的副本。

        图片块也留在会话里。tool 消息发出去时只剩文字；模型能看图时，这一批
        工具结果后面追加一条 user 消息，用 image_url data URL 带上图片。
        不能看图时，文字末尾换成不支持图片的说明，不附 data URL。
        """
        last = _last_assistant_index(messages)
        out: list[dict] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") != "tool":
                out.append(self._wire_plain(msg, i, last))
                i += 1
                continue
            # 连续的 tool 结果先变成纯文本。能看图时，这一批的图片跟在后面的 user 消息里。
            # OpenAI 的 tool 角色只能放字符串，图片不能塞进 tool content。
            attached: list[dict] = []
            while i < len(messages) and messages[i].get("role") == "tool":
                clean = {k: v for k, v in messages[i].items() if not k.startswith("_")}
                images = _image_blocks(clean.get("content"))
                if images or isinstance(clean.get("content"), list):
                    clean["content"] = _tool_wire_text(
                        clean.get("content"), images, attach=self.cfg.accepts_images
                    )
                    if self.cfg.accepts_images:
                        attached.extend(_image_urls(images))
                out.append(clean)
                i += 1
            if attached:
                out.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _ATTACHED_IMAGES},
                        *attached,
                    ],
                })
        return out

    def _wire_plain(self, msg: dict, index: int, last: int) -> dict:
        clean = {k: v for k, v in msg.items() if not k.startswith("_")}
        if self.must_echo_reasoning and clean.get("role") == "assistant":
            if index != last or not clean.get("reasoning_content"):
                clean["reasoning_content"] = _REASONING_PAD
        return clean

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_delta: Callable[[str, str], None] | None = None,
        on_notice: Callable[[str], None] | None = None,
    ) -> dict:
        """流式请求：边收边回调，最后返回拼装完整的消息。

        on_delta(kind, text)：kind 为 "reasoning" 或 "content"。
        tool_calls 是分片下发的，按 index 聚合 name/arguments。
        on_notice(text)：重试等异常情况时给用户看的提示行。
        """
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": self._wire_messages(messages),
            "temperature": self.cfg.temperature,
            "stream": True,
            # 让最后一个流式 chunk 带回 usage，供锚点记账；
            # 个别 OpenAI 兼容端点不认这个参数，失败时退回去重试一次
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
        error_attempts = 0

        def _handle_error(exc: Exception) -> bool:
            """可重试 → 退避并返回 True（continue）；否则抛对应错误。"""
            nonlocal error_attempts
            action = _classify_api_error(exc)
            if action == "overflow":
                raise ContextOverflowError(str(exc)) from exc
            if action == "retry" and error_attempts < MAX_ERROR_ATTEMPTS:
                error_attempts += 1
                wait = _backoff_seconds(error_attempts, exc)
                if on_notice:
                    on_notice(
                        f"API 错误（{type(exc).__name__}），{wait:.1f}s 后重试"
                        f"（{error_attempts}/{MAX_ERROR_ATTEMPTS}）"
                    )
                time.sleep(wait)
                return True
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        while True:
            emitted = False  # 每次尝试单独记账：重试后 budget 饥饿判定才正确
            kwargs["max_tokens"] = budget
            try:
                stream = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                msg = str(exc)
                if "stream_options" in msg and "stream_options" in kwargs:
                    # 端点不支持 include_usage：去掉参数重试，本轮退回纯估算
                    kwargs.pop("stream_options")
                    continue
                if _handle_error(exc):
                    continue

            content: list[str] = []
            reasoning: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            finish: str | None = None
            usage: dict[str, int] | None = None
            last_progress = [0]

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
                                # 大文件 Write/Edit 的参数可能有几千 token 的 JSON，
                                # 流式期间终端本来一片死寂——每 ~500 字符报一次进度。
                                # 工具名通常随首个 delta 就到，带上它让等待可辨识
                                arg_chars = sum(len(s["arguments"]) for s in calls.values())
                                if on_delta and arg_chars - last_progress[0] >= 500:
                                    last_progress[0] = arg_chars
                                    names = "+".join(
                                        sorted({s["name"] for s in calls.values() if s["name"]})
                                    ) or "工具"
                                    on_delta("progress", f"{names} {arg_chars}")
            except Exception as exc:
                if _handle_error(exc):
                    continue  # 流中途断：整请求重试（已吐出的字会重印，提示行已说明）
                raise  # 不可达，_handle_error 内部非 True 即抛

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

        finish_reason 为 length/content_filter 或正文为空时拒收：
        一律抛 LLMError，调用方走机械裁剪兜底——坏摘要永久污染会话比不摘要更糟。
        网络类错误重试 2 次（压缩失败本来就有机械兜底，这里少重试无妨）。
        """
        resp: Any = None
        for attempt in range(1, 3 + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2048,
                )
                break
            except Exception as exc:
                if _classify_api_error(exc) == "retry" and attempt < 3:
                    time.sleep(_backoff_seconds(attempt, exc))
                    continue
                raise LLMError(f"summary failed: {type(exc).__name__}: {exc}") from exc

        choice = resp.choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish in ("length", "content_filter"):
            raise LLMError(f"summary truncated (finish_reason={finish})，拒收半成品")
        text = (choice.message.content or "").strip()
        if not text:
            raise LLMError("summary returned empty content")
        return text


def context_tokens(messages: list[dict], *, collapse_old_reasoning: bool = False) -> int:
    """锚点 + 粗估的上下文记账。

    从后往前找最近一条带真实 usage 的 assistant 消息（私有字段 _usage，
    chat_stream 在 include_usage 时填充）：锚点 total ≈ 当时请求的 input+output，
    即当时的上下文规模；锚点之后的消息用启发式逐条估算。
    锚点之后若发生过压缩（[COMPACTION] 标记），锚点作废、全量估算——
    它统计过的上下文大部分已被丢进摘要。

    collapse_old_reasoning：与 _wire_messages 一致，非最后一条 assistant 的思维链
    只计 1 token。锚点里包含的是「当时最后一条」的全文，所以只从锚点那条里扣掉
    即将收成空格的部分，避免把上一轮已经收过的思维链再扣一次。
    """
    last = _last_assistant_index(messages)

    def tok(i: int, msg: dict) -> int:
        collapse = collapse_old_reasoning and msg.get("role") == "assistant" and i != last
        return message_tokens(msg, collapse_reasoning=collapse)

    anchor = _fresh_anchor(messages)
    if anchor is None:
        return sum(tok(i, m) for i, m in enumerate(messages))

    anchor_index, anchor_total = anchor
    total = anchor_total
    if collapse_old_reasoning and anchor_index != last:
        reasoning = str(messages[anchor_index].get("reasoning_content") or "")
        if reasoning and reasoning != _REASONING_PAD:
            total -= max(0, estimate_text_tokens(reasoning) - 1)
    for i in range(anchor_index + 1, len(messages)):
        total += tok(i, messages[i])
    return max(0, total)


def count_tokens(messages: list[dict]) -> int:
    """无锚点时的全量启发式估算。"""
    return sum(message_tokens(m) for m in messages)


def request_size(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    collapse_old_reasoning: bool = False,
) -> int:
    """一次请求的估算 token 总量。

    有新鲜 usage 锚点时不再加工具 schema：prompt_tokens 里已经含过。
    没有锚点时加一次。
    """
    total = context_tokens(messages, collapse_old_reasoning=collapse_old_reasoning)
    if tools and _fresh_anchor(messages) is None:
        total += count_tokens_json(tools)
    return total
