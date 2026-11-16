"""粗略 token 估算：不依赖 tiktoken，避免额外下载编码表。

规则：ASCII 约 3.5 字符/token，非 ASCII（中文等）约 1 字符/token，
每条消息再加固定开销。误差 ±15%，压缩阈值留了余量，够用。
"""
from __future__ import annotations

import json

PER_MESSAGE_OVERHEAD = 4
ASCII_RATIO = 3.5
IMAGE_TOKEN_ESTIMATE = 1600  # 一张图按固定开销计，不按 base64 长度


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii = len(text) - ascii_chars
    return int(ascii_chars / ASCII_RATIO) + non_ascii + 1


def content_text(content: object) -> str:
    """取出消息正文。content 可以是字符串，或 text/image 块列表。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    return str(content)


def content_image_count(content: object) -> int:
    if not isinstance(content, list):
        return 0
    return sum(
        1
        for block in content
        if isinstance(block, dict) and block.get("type") == "image" and block.get("data")
    )


def message_tokens(msg: dict, *, collapse_reasoning: bool = False) -> int:
    total = PER_MESSAGE_OVERHEAD
    content = msg.get("content")
    total += estimate_text_tokens(content_text(content))
    total += content_image_count(content) * IMAGE_TOKEN_ESTIMATE
    reasoning = str(msg.get("reasoning_content") or "")
    if collapse_reasoning and msg.get("role") == "assistant":
        # 发给 must-echo 端点时，非最后一条只回传一个空格
        total += 1
    else:
        total += estimate_text_tokens(reasoning)
    for call in msg.get("tool_calls") or []:
        fn = call.get("function", {})
        total += estimate_text_tokens(fn.get("name", ""))
        total += estimate_text_tokens(fn.get("arguments", ""))
    return total


def count_tokens(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages)


def count_tokens_json(obj: object) -> int:
    return estimate_text_tokens(json.dumps(obj, ensure_ascii=False))
