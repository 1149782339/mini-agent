"""对话历史 + 自动压缩（context compaction）。

压缩三段式：
1. 触发：请求估算超过窗口的 compact_ratio；
2. 切分：从尾部保留 keep_recent 条，向前挪动切点，避免劈开 assistant(tool_calls) / tool 配对；
3. 摘要：把头部交给 LLM 生成交接笔记，替换成一条 [COMPACTION] 消息；
   失败则退化为机械裁剪（丢掉最老一半），仍然保证配对完整。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .client import LLM, LLMError, request_size
from .config import Config, SYSTEM_PROMPT


COMPACTION_MARKER = "[COMPACTION]"


@dataclass
class CompactionReport:
    before_tokens: int
    after_tokens: int
    dropped_messages: int
    summarised: bool
    attempts: int = 1


@dataclass
class Context:
    cfg: Config
    system_prompt: str = SYSTEM_PROMPT
    messages: list[dict] = field(default_factory=list)
    log: list[CompactionReport] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]

    # ---------- 基础 ----------
    def add(self, message: dict) -> None:
        self.messages.append(message)

    def size(self, tools: list[dict] | None = None) -> int:
        return request_size(self.messages, tools)

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.log.clear()

    # ---------- 切分 ----------
    def _find_split(self) -> int:
        """返回切点 cut：head = messages[1:cut]，tail = messages[cut:]。"""
        n = len(self.messages)
        if n <= 2:
            return n
        cut = max(1, n - self.cfg.keep_recent)
        # 若切点落在 tool 消息上，往前挪到它的 assistant 之前，保证配对不裂
        while cut > 1 and self.messages[cut].get("role") == "tool":
            cut -= 1
        # 切点若紧跟 assistant(tool_calls)，同样前移，避免尾部以孤立 tool 开头
        while cut > 1:
            prev = self.messages[cut - 1]
            if prev.get("role") == "assistant" and prev.get("tool_calls"):
                cut -= 1
            else:
                break
        return max(1, cut)

    # ---------- 校验 ----------
    def validate(self, tools: list[dict] | None = None) -> list[str]:
        problems: list[str] = []
        if not self.messages or self.messages[0].get("role") != "system":
            problems.append("第一条消息不是 system")

        open_calls: dict[str, str] = {}
        for index, msg in enumerate(self.messages):
            role = msg.get("role")
            if role == "assistant":
                for call in msg.get("tool_calls") or []:
                    open_calls[call.get("id", "")] = index
            elif role == "tool":
                call_id = msg.get("tool_call_id", "")
                if call_id not in open_calls:
                    problems.append(f"第 {index} 条 tool 消息找不到对应 tool_call（{call_id}）")
                else:
                    open_calls.pop(call_id)
        for call_id, index in open_calls.items():
            problems.append(f"第 {index} 条 assistant 的 tool_call {call_id} 没有结果")

        size = self.size(tools)
        if size > self.cfg.context_window:
            problems.append(f"总 token {size} 超过窗口 {self.cfg.context_window}")
        return problems

    # ---------- 压缩 ----------
    def should_compact(self, tools: list[dict] | None = None) -> bool:
        return self.size(tools) > self.cfg.compact_threshold

    def compact(
        self,
        llm: LLM,
        tools: list[dict] | None = None,
        max_attempts: int = 2,
        force: bool = False,
    ) -> CompactionReport | None:
        """必要时压缩；返回报告，未压缩返回 None。force=True 时忽略阈值。"""
        if len(self.messages) <= 2:
            return None
        if not force and not self.should_compact(tools):
            return None

        before = self.size(tools)
        attempts = 0
        summarised = False
        snapshot = list(self.messages)
        dropped = 0

        while attempts < max_attempts and (force or self.should_compact(tools)):
            attempts += 1
            cut = self._find_split()
            if cut <= 1:
                break
            head = self.messages[1:cut]
            tail = self.messages[cut:]
            dropped += len(head)

            summary_text = ""
            try:
                summary_text = llm.summarize(head)
                summarised = True
            except LLMError:
                summary_text = ""

            if summary_text:
                summary_msg = {
                    "role": "user",
                    "content": (
                        f"{COMPACTION_MARKER} 以下是此前对话的压缩摘要"
                        f"（压缩发生于第 {len(self.messages)} 条消息之后）：\n\n"
                        f"{summary_text}\n\n"
                        "[以上为历史摘要；以下为最近的真实对话，请接着处理]"
                    ),
                }
                self.messages = [self.messages[0], summary_msg, *tail]
            else:
                # 兜底：机械保留后半段
                half = max(1, len(self.messages) // 2)
                while half > 1 and self.messages[half].get("role") == "tool":
                    half -= 1
                dropped += half - 1
                self.messages = [self.messages[0], *self.messages[half:]]

            problems = self.validate(tools)
            if problems:
                # 校验失败：回滚并走机械裁剪，最多两次
                self.messages = snapshot
                self._mechanical_trim()
                if not self.validate(tools):
                    break

        after = self.size(tools)
        report = CompactionReport(
            before_tokens=before,
            after_tokens=after,
            dropped_messages=dropped,
            summarised=summarised,
            attempts=attempts,
        )
        self.log.append(report)
        return report

    def _mechanical_trim(self) -> None:
        n = len(self.messages)
        if n <= 2:
            return
        half = max(1, n // 2)
        while half > 1 and self.messages[half].get("role") == "tool":
            half -= 1
        self.messages = [self.messages[0], *self.messages[half:]]

    def force_fit(self, tools: list[dict] | None = None) -> None:
        """最后一道保险：反复机械裁剪直到进入窗口。"""
        guard = 0
        while self.size(tools) > self.cfg.context_window and len(self.messages) > 2 and guard < 40:
            self._mechanical_trim()
            guard += 1
