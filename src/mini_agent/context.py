"""对话历史 + 自动压缩。

1. 切分：token 预算制（keep_recent_tokens）——从最新往回累加估算到预算即停；
   切点绝不切断 toolCall/toolResult 配对（compaction.find_cut_point）；
2. 摘要：输入先序列化成 [User]/[Tool result] 标签文本（防模型续写对话、
   工具结果限 2000 字符、思维链丢弃），固定 6 节结构模板；有旧摘要时走
   "更新合并"模式而非重写；
3. 文件清单：<read-files>/<modified-files> 从工具调用参数机械提取，
   确定性附加在摘要尾（跨次压缩随摘要文本累积）；
4. 质量闸：摘要为空/被截断(finish_reason=length)/像续写或报错 → 拒收，
   走机械裁剪兜底（坏摘要永久污染会话比不摘要更糟）；
5. 记账：request_size 用 usage 锚点 + 增量粗估（client.context_tokens）。

上下文溢出恢复在 loop.py：provider 报窗口超限时强制压缩，并重试本轮一次。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .client import LLM, LLMError, must_echo_model, request_size
from .tokens import content_image_count, content_text, message_tokens
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
        return request_size(
            self.messages,
            tools,
            collapse_old_reasoning=must_echo_model(self.cfg.model),
        )

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.log.clear()

    # ---------- 切分 ----------
    def _keep_tokens(self, tools: list[dict] | None = None) -> int:
        """尾部保留预算。窗口很小的场景（测试/小模型）自适应收缩，
        且永远不超过压缩阈值的一半，保证切完真的能降下来。"""
        threshold = self.cfg.compact_threshold
        return min(self.cfg.keep_recent_tokens, max(1000, threshold // 2))

    def _find_split(self, tools: list[dict] | None = None) -> int:
        """切点：tail = messages[cut:] 保留，head = messages[1:cut] 进摘要。"""
        return find_cut_point(self.messages, self._keep_tokens(tools))

    def _previous_summary(self) -> str:
        """从历史里取出上一份压缩摘要（迭代式摘要的"旧摘要"输入）。"""
        for msg in self.messages:
            content = content_text(msg.get("content"))
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
        if len(self.messages) <= 1:
            return None
        if not force and not self.should_compact(tools):
            return None

        before = self.size(tools)
        attempts = 0
        summarised = False
        dropped = 0

        # 只有 system + 一条超长消息时切点无从下手，直接截断这条。
        if len(self.messages) <= 2:
            if not self._shrink_oversized():
                return None
        else:
            # force 只放行“首次即使未超阈值也压缩”；一旦已经压到阈值以下就停，
            # 避免 /compact 第二轮摘要失败时把刚写好的摘要又机械裁掉。
            while attempts < max_attempts and (attempts == 0 or self.should_compact(tools)):
                attempts += 1
                snapshot = list(self.messages)  # 只回滚这一次，不丢掉上一轮已经成功的摘要
                cut = self._find_split(tools)
                if cut <= 1:
                    if not self._shrink_oversized():
                        break
                    continue
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
                    # 摘要失败：机械裁掉最老一半；裁不动就截断单条超长消息
                    kept = self._mechanical_trim()
                    if kept is None and not self._shrink_oversized():
                        break

                problems = self.validate(tools)
                if problems:
                    # 校验失败：只回滚这一次，再机械裁剪；仍不行就截断超长消息
                    self.messages = list(snapshot)
                    if self._mechanical_trim() is None and not self._shrink_oversized():
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

    def _shrink_oversized(self) -> bool:
        """单条消息超过窗口四分之一时，收思维链、截正文。返回是否改动了历史。

        裁剪以消息为单位，一条超长 reasoning 或粘贴会让切点和机械裁剪都挪不动。
        """
        cap = max(1000, self.cfg.context_window // 4)
        changed = False
        for msg in self.messages:
            if msg.get("role") == "system":
                continue
            if message_tokens(msg) <= cap:
                continue
            reasoning = str(msg.get("reasoning_content") or "")
            if len(reasoning) > 1:
                msg["reasoning_content"] = " "
                changed = True
            if message_tokens(msg) <= cap:
                continue
            content = msg.get("content")
            text = content_text(content)
            images = content_image_count(content)
            # 图片按固定估算。超限时丢掉图片块，避免把 base64 截成一段残缺文本。
            if images and message_tokens(msg) > cap:
                note = f"{text}\n[已省略 {images} 张图片]" if text else f"[已省略 {images} 张图片]"
                if len(note) > cap:
                    note = note[:cap] + f"\n…[单条消息过长已截断，原 {len(text)} 字符]"
                if note != content:
                    msg["content"] = note
                    changed = True
                continue
            if len(text) > cap:
                clipped = text[:cap] + f"\n…[单条消息过长已截断，原 {len(text)} 字符]"
                if clipped != content:
                    msg["content"] = clipped
                    changed = True
        return changed

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
        while self.size(tools) > self.cfg.context_window and len(self.messages) > 1 and guard < 40:
            if self._mechanical_trim() is None and not self._shrink_oversized():
                break
            guard += 1
