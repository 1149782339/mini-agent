"""对话历史 + 自动压缩（context compaction），压缩管线移植自 Pi。

相比旧版（按条数切 + 自由文本摘要）的改动：
1. 切分：token 预算制（keep_recent_tokens，Pi keepRecentTokens）——从最新往回
   累加估算到预算即停；切点绝不切断 toolCall/toolResult 配对
   （compaction.find_cut_point）；
2. 摘要：输入先序列化成 [User]/[Tool result] 标签文本（防模型续写对话、
   工具结果限 2000 字符、思维链丢弃），固定 6 节结构模板；有旧摘要时走
   "更新合并"模式而非重写；
3. 文件清单：<read-files>/<modified-files> 从工具调用参数机械提取，
   确定性附加在摘要尾（跨次压缩随摘要文本累积）；
4. 质量闸：摘要为空/被截断(finish_reason=length)/像续写或报错 → 拒收，
   走机械裁剪兜底（坏摘要永久污染会话比不摘要更糟）；
5. 记账：request_size 用 usage 锚点 + 增量粗估（client.context_tokens）。

未移植（见 README）：split-turn 双摘要、分支摘要、append-only 事件树投影、
溢出自动恢复（provider 报错格式相关）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .client import LLM, LLMError, request_size
from .compaction import (
    append_file_lists,
    build_summary_prompt,
    extract_file_operations,
    find_cut_point,
    looks_like_bad_summary,
    serialize_for_summary,
)
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
    def _keep_tokens(self, tools: list[dict] | None = None) -> int:
        """尾部保留预算：Pi 默认 20K；窗口很小的场景（测试/小模型）自适应收缩，
        且永远不超过压缩阈值的一半，保证切完真的能降下来。"""
        threshold = self.cfg.compact_threshold
        return min(self.cfg.keep_recent_tokens, max(1000, threshold // 2))

    def _find_split(self, tools: list[dict] | None = None) -> int:
        """切点：tail = messages[cut:] 保留，head = messages[1:cut] 进摘要。"""
        return find_cut_point(self.messages, self._keep_tokens(tools))

    def _previous_summary(self) -> str:
        """从历史里取出上一份压缩摘要（迭代式摘要的"旧摘要"输入）。"""
        for msg in self.messages:
            content = str(msg.get("content") or "")
            if content.startswith(COMPACTION_MARKER):
                body = content[len(COMPACTION_MARKER):]
                parts = body.split("\n\n", 1)
                text = parts[1] if len(parts) == 2 else body
                return text.rsplit("[以上为历史摘要", 1)[0].strip()
        return ""

    @staticmethod
    def _marker_message(summary_text: str) -> dict:
        return {
            "role": "user",
            "content": (
                f"{COMPACTION_MARKER} 以下是此前对话的压缩摘要：\n\n"
                f"{summary_text}\n\n"
                "[以上为历史摘要；以下为最近的真实对话，请接着处理]"
            ),
        }

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
        dropped = 0
        snapshot = list(self.messages)

        while attempts < max_attempts and (force or self.should_compact(tools)):
            attempts += 1
            cut = self._find_split(tools)
            if cut <= 1:
                break
            head = self.messages[1:cut]
            tail = self.messages[cut:]

            previous = self._previous_summary()
            read_files, modified_files = extract_file_operations(head)
            prompt = build_summary_prompt(
                serialize_for_summary(head),
                previous,
                read_files,
                modified_files,
            )

            summary_text = ""
            try:
                candidate = llm.summarize(prompt)
                if looks_like_bad_summary(candidate):
                    raise LLMError("summary rejected by quality gate")
                # 文件清单确定性附加，不让 LLM 自由发挥路径
                summary_text = append_file_lists(candidate, read_files, modified_files)
                summarised = True
            except LLMError:
                summary_text = ""

            if summary_text:
                dropped += len(head)
                self.messages = [self.messages[0], self._marker_message(summary_text), *tail]
            else:
                # 摘要失败：机械裁掉最老一半（保底，仍保证配对完整）
                kept = self._mechanical_trim()
                if kept is None:
                    break

            problems = self.validate(tools)
            if problems:
                # 校验失败：回滚本轮改动，机械裁剪一次，仍不行就放弃
                self.messages = snapshot
                if self._mechanical_trim() is None:
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

    def _mechanical_trim(self) -> int | None:
        """机械保留较新的一半（不切 tool 消息）。返回切点；无需裁剪返回 None。"""
        n = len(self.messages)
        if n <= 2:
            return None
        half = max(1, n // 2)
        while half > 1 and self.messages[half].get("role") == "tool":
            half -= 1
        self.messages = [self.messages[0], *self.messages[half:]]
        return half

    def force_fit(self, tools: list[dict] | None = None) -> None:
        """最后一道保险：反复机械裁剪直到进入窗口。"""
        guard = 0
        while self.size(tools) > self.cfg.context_window and len(self.messages) > 2 and guard < 40:
            if self._mechanical_trim() is None:
                break
            guard += 1
