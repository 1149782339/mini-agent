"""结构化用户提问；具体交互由宿主注入的可选 handler 完成。"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base import Tool, ToolContext, ToolError, ToolOutput, ValidationResult, semantic_bool

InteractionHandler = Callable[[list[dict[str, Any]]], dict[str, Any]]
_VALID_OUTCOMES = {"submitted", "cancelled", "timed_out", "unavailable", "undelivered"}


def _normalize_questions(raw: Any) -> tuple[list[dict[str, Any]] | None, str]:
    if not isinstance(raw, list) or not raw:
        return None, "questions 必须是非空数组"
    if len(raw) > 4:
        return None, "一次最多询问 4 个问题"

    normalized: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            return None, f"questions[{index}] 必须是对象"
        question = str(item.get("question") or "").strip()
        if not question:
            return None, f"questions[{index}].question 不能为空"
        if question in seen_questions:
            return None, f"问题文本不能重复：{question}"
        seen_questions.add(question)

        raw_options = item.get("options")
        if not isinstance(raw_options, list) or not 2 <= len(raw_options) <= 4:
            return None, f"questions[{index}].options 必须包含 2-4 个选项"
        options: list[dict[str, str]] = []
        seen_labels: set[str] = set()
        for option_index, option in enumerate(raw_options):
            if not isinstance(option, dict):
                return None, f"questions[{index}].options[{option_index}] 必须是对象"
            label = str(option.get("label") or "").strip()
            description = str(option.get("description") or "").strip()
            if not label or not description:
                return None, (
                    f"questions[{index}].options[{option_index}] "
                    "必须提供非空 label 和 description"
                )
            if label in seen_labels:
                return None, f"questions[{index}] 的选项 label 不能重复：{label}"
            seen_labels.add(label)
            options.append({"label": label, "description": description})

        normalized.append({
            "qid": f"q{index}",
            "question": question,
            "options": options,
            "multi_select": semantic_bool(item.get("multi_select"), False),
        })
    return normalized, ""


def _clean_answer(raw: Any, multi_select: bool) -> str | list[str] | None:
    if raw is None:
        return None
    if multi_select:
        values = raw if isinstance(raw, list) else [raw]
        cleaned = [
            str(value).strip() for value in values
            if value is not None and str(value).strip()
        ]
        return cleaned or None
    return str(raw).strip() or None


class AskUserTool(Tool):
    name = "AskUser"
    description = (
        "Ask the user structured multiple-choice questions when missing information "
        "would materially change the result. Ask 1-4 independent questions in one call. "
        "Each question needs 2-4 options with concise labels and useful descriptions. "
        "Users can always provide a custom answer. Use multi_select=true for choices "
        "that are not mutually exclusive. Do not ask about trivial decisions that you "
        "can infer, and do not use this tool for plan approval."
    )
    parameters = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["label", "description"],
                            },
                        },
                        "multi_select": {"type": "boolean", "default": False},
                    },
                    "required": ["question", "options"],
                },
            },
        },
        "required": ["questions"],
    }

    def __init__(self, interaction_handler: InteractionHandler | None = None):
        self.interaction_handler = interaction_handler

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return False

    def validate_input(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        _, error = _normalize_questions(args.get("questions"))
        if error:
            return ValidationResult.fail(error, error_code=1)
        return ValidationResult.pass_()

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        questions, error = _normalize_questions(args.get("questions"))
        if error or questions is None:
            raise ToolError(error or "questions 参数无效", error_code=1)

        if self.interaction_handler is None:
            reply = {
                "answers": {},
                "outcome": "unavailable",
                "notice": (
                    "当前执行环境无法向用户提问。请根据已有信息作出最合理的判断并继续，"
                    "不要重复调用 AskUser。"
                ),
            }
        else:
            try:
                reply = self.interaction_handler(questions)
            except Exception as exc:
                raise ToolError(f"获取用户回答失败：{exc}", error_code=2) from exc
            if not isinstance(reply, dict):
                raise ToolError("interaction_handler 必须返回字典", error_code=3)

        outcome = str(reply.get("outcome") or "").strip()
        if outcome not in _VALID_OUTCOMES:
            raise ToolError(f"无效的提问结果 outcome：{outcome or '缺失'}", error_code=4)
        answers = reply.get("answers") or {}
        if not isinstance(answers, dict):
            raise ToolError("interaction_handler 的 answers 必须是字典", error_code=5)

        responses = []
        for question in questions:
            qid = question["qid"]
            value = _clean_answer(answers.get(qid), question["multi_select"])
            if value is not None:
                status = "answered"
            elif qid in answers:
                status = "skipped"
            else:
                status = "unanswered"
            responses.append({
                "question": question["question"],
                "status": status,
                "user_response": value,
            })

        result: dict[str, Any] = {"responses": responses, "outcome": outcome}
        if reply.get("notice"):
            result["notice"] = str(reply["notice"])
        return ToolOutput(text=json.dumps(result, ensure_ascii=False))
