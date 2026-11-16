"""上下文压缩的纯函数组件，移植自 Pi（packages/coding-agent/src/core/compaction/）。

设计取自 Pi 且与 mini-agent 现状对齐的部分：
- find_cut_point：从最新往回按 token 预算找切点；绝不切断 toolCall/toolResult 配对
  （切点绝不落在 tool 消息上，也绝不把 assistant(tool_calls) 和它的结果劈开）
- serialize_for_summary：把消息序列化成 [User]/[Assistant]/[Tool result] 标签文本
  再交给 LLM 摘要——防模型把摘要请求当对话续写；工具结果限 2000 字符、丢弃思维链
- extract_file_operations：从工具调用参数里机械提取读过的/改过的文件清单，
  以 <read-files>/<modified-files> 确定性附加到摘要尾（LLM 摘要常丢精确路径）
- build_summary_prompt：固定 6 节结构模板；有旧摘要时用"更新合并而非重写"模式

没移植的：split-turn 双摘要（单轮就超预算时退化为整体摘要）、分支摘要、
append-only 事件树投影（那是架构级改造，见 README）。
"""
from __future__ import annotations

import json

from .tokens import message_tokens

# ---------- 常量（对齐 Pi 默认） ----------
SUMMARY_TOOL_RESULT_CHARS = 2000   # 摘要输入里单条工具结果上限
SUMMARY_CONTENT_CHARS = 4000       # 摘要输入里单条 user/assistant 正文上限
FILE_LIST_CAP = 200                # 文件清单最多列多少条

TOOL_NAME_READ = {"Read", "Glob", "Grep", "Ls"}
TOOL_NAME_MODIFY = {"Edit", "Write"}


# ---------- 切点 ----------
def find_cut_point(messages: list[dict], keep_tokens: int) -> int:
    """返回切点 cut：tail = messages[cut:] 保留，head = messages[1:cut] 进摘要。

    messages[0] 是 system，永不切到 0。算法（Pi findProjectedCutPoint 同款）：
    从最新往回累加估算 token，达到 keep_tokens 即停；然后合法化切点——
    切点绝不落在 tool 消息上（工具结果必须跟随其 toolCall），
    也绝不把 assistant(tool_calls) 与它的 tool 结果劈开（宁可多保留往前挪）。
    """
    n = len(messages)
    if n <= 2:
        return n

    total = 0
    cut = 1
    for i in range(n - 1, 0, -1):
        total += message_tokens(messages[i])
        if total >= keep_tokens:
            cut = i
            break
    else:
        return n  # 全史都在预算内，无需压缩

    # 合法化：messages[cut] 是 tail 的第一条
    while cut > 1:
        first = messages[cut]
        if first.get("role") == "tool":
            # tail 以 tool 消息开头 = 它的 assistant 被劈进了 head，整个挪进 tail
            cut -= 1
            continue
        prev = messages[cut - 1]
        if prev.get("role") == "assistant" and prev.get("tool_calls"):
            # tail 紧跟一个带 tool_calls 的 assistant，但该 assistant 的结果在 head：
            # 这个 assistant 也必须整体进 tail
            cut -= 1
            continue
        break
    return max(1, cut)


# ---------- 摘要输入序列化 ----------
def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f" …[截断，原 {len(text)} 字符]"


def serialize_for_summary(messages: list[dict]) -> str:
    """把消息压成 [User]/[Assistant]/[Tool result] 标签文本（Pi serializeConversation）。

    思维链（reasoning_content）整体丢弃——交接不需要，还省 token。
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue  # system prompt 每轮都在，不进摘要输入
        if role == "user":
            parts.append(f"[User]: {_clip(str(msg.get('content') or ''), SUMMARY_CONTENT_CHARS)}")
        elif role == "assistant":
            calls = msg.get("tool_calls") or []
            if calls:
                names = []
                for c in calls:
                    fn = c.get("function", {})
                    names.append(
                        f"{fn.get('name', '?')}({_clip(fn.get('arguments', ''), 200)})"
                    )
                parts.append(f"[Assistant tool calls]: {'; '.join(names)}")
            content = str(msg.get("content") or "").strip()
            if content:
                parts.append(f"[Assistant]: {_clip(content, SUMMARY_CONTENT_CHARS)}")
        elif role == "tool":
            call_id = msg.get("tool_call_id", "")
            parts.append(
                f"[Tool result {call_id}]: "
                f"{_clip(str(msg.get('content') or ''), SUMMARY_TOOL_RESULT_CHARS)}"
            )
    return "\n".join(parts)


# ---------- 文件清单（确定性提取，跨压缩随摘要文本累积） ----------
def extract_file_operations(messages: list[dict]) -> tuple[list[str], list[str]]:
    """从工具调用参数里提取读过的文件 / 改过的文件（Pi extractFileOperations）。

    只认参数（file_path/path/pattern 所在目录），解析失败跳过——
    确定性提取，不靠 LLM，保证路径精确。
    """
    read_files: set[str] = set()
    modified_files: set[str] = set()

    def _add(args_raw: str, bucket: set[str], keys: tuple[str, ...]) -> None:
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(args, dict):
            return
        for key in keys:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                bucket.add(value.strip())
                return

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            if name in TOOL_NAME_READ:
                _add(args, read_files, ("file_path", "path", "pattern"))
            elif name in TOOL_NAME_MODIFY:
                _add(args, modified_files, ("file_path",))

    return (
        sorted(read_files)[:FILE_LIST_CAP],
        sorted(modified_files)[:FILE_LIST_CAP],
    )


# ---------- 摘要 prompt ----------
SUMMARY_SECTIONS = (
    "## Goal（用户的目标与当前任务）\n"
    "## Constraints（约束、偏好、环境）\n"
    "## Progress（已完成事项；改过的文件逐个列路径）\n"
    "## Key Decisions（关键决策及理由）\n"
    "## Next Steps（待办与下一步）\n"
    "## Critical Context（关键参数、错误与修复、进行中的调用链）"
)

_UPDATE_MODE_NOTE = (
    "已有一份旧摘要（见下）。请基于新出现的对话更新合并它：保留仍然成立的内容，"
    "补充新的进展，修正已被推翻的结论。不要重写一份无关的新摘要。\n\n"
    "<旧摘要>\n{previous}\n</旧摘要>\n\n"
)


def build_summary_prompt(
    serialized: str,
    previous_summary: str = "",
    read_files: list[str] | None = None,
    modified_files: list[str] | None = None,
) -> str:
    """6 节结构化交接摘要 prompt（Pi 模板本地化）。文件清单确定性附加。"""
    update = ""
    if previous_summary.strip():
        update = _UPDATE_MODE_NOTE.format(previous=previous_summary.strip())

    lists = []
    if read_files:
        lists.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        lists.append(
            "<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>"
        )
    file_note = (
        "\n\n此外，以下文件清单是从工具调用记录机械提取的（确定性事实），"
        "请原样保留在摘要对应小节里：\n" + "\n".join(lists)
        if lists else ""
    )

    return (
        "请把下面的对话压缩成一份结构化交接摘要，供另一个 AI 接手继续工作。\n"
        "硬性要求：\n"
        "- 只输出摘要本身：不要调用任何工具，不要寒暄，不要复述本指令。\n"
        "- 只写对话里真实出现过的内容，禁止编造；关键结论尽量引用原文短语。\n"
        "- 严格遵守下面的小节结构（markdown 标题）：\n"
        f"{SUMMARY_SECTIONS}\n"
        "- 如果某项没有内容，写「无」。\n"
        f"{update}"
        f"<对话>\n{serialized}\n</对话>"
        f"{file_note}"
    )


def append_file_lists(summary: str, read_files: list[str],
                      modified_files: list[str]) -> str:
    """把文件清单确定性附加到摘要末尾（不进 LLM 自由发挥范围）。"""
    blocks = []
    if read_files:
        blocks.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        blocks.append(
            "<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>"
        )
    return summary.rstrip() + ("\n\n" + "\n".join(blocks) if blocks else "")


def looks_like_bad_summary(text: str) -> bool:
    """摘要质量闸（Pi getSummarizationFailure 的简化）：拒收明显坏掉的摘要。"""
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith(("ERROR", "Error", "error:", "API Error")):
        return True
    # 模型擅自开始"续写对话"或试图调工具
    if "[User]:" in stripped or "[Assistant]:" in stripped:
        return True
    if "<tool_call" in stripped or "Action:" in stripped[:200]:
        return True
    # 太短 = 基本没干活（正常摘要至少几百字）
    if len(stripped) < 80:
        return True
    return False
