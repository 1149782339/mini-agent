"""粗略 token 估算：不依赖 tiktoken，避免额外下载编码表。

规则：ASCII 约 3.5 字符/token，非 ASCII（中文等）约 1 字符/token，
每条消息再加固定开销。误差 ±15%，压缩阈值留了余量，够用。
"""
from __future__ import annotations

import json

PER_MESSAGE_OVERHEAD = 4
ASCII_RATIO = 3.5


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii = len(text) - ascii_chars
    return int(ascii_chars / ASCII_RATIO) + non_ascii + 1


def message_tokens(msg: dict) -> int:
    total = PER_MESSAGE_OVERHEAD
    total += estimate_text_tokens(str(msg.get("content") or ""))
    total += estimate_text_tokens(str(msg.get("reasoning_content") or ""))
    for call in msg.get("tool_calls") or []:
        fn = call.get("function", {})
        total += estimate_text_tokens(fn.get("name", ""))
        total += estimate_text_tokens(fn.get("arguments", ""))
    return total


def count_tokens(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages)


def count_tokens_json(obj: object) -> int:
    return estimate_text_tokens(json.dumps(obj, ensure_ascii=False))
