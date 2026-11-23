"""交互式 CLI 的提问适配器；stdin 仍由 REPL 输入线程统一读取。"""
from __future__ import annotations

import queue
import re
import threading
import time
from typing import Any, Callable


def _resolve_token(text: str, options: list[dict[str, str]]) -> tuple[str, str]:
    if text.isdecimal():
        index = int(text) - 1
        if 0 <= index < len(options):
            return options[index]["label"], ""
        return "", f"编号 {text} 超出范围，请输入 1-{len(options)}。"
    for option in options:
        if option["label"].casefold() == text.casefold():
            return option["label"], ""
    return text, ""


def _parse_answer(text: str, question: dict[str, Any]) -> tuple[Any, str]:
    options = question["options"]
    if not question["multi_select"]:
        return _resolve_token(text, options)

    tokens = [token.strip() for token in re.split(r"[,，]", text) if token.strip()]
    if not tokens:
        return None, "多选回答不能为空；如需跳过请输入 /skip。"
    answers: list[str] = []
    for token in tokens:
        answer, error = _resolve_token(token, options)
        if error:
            return None, error
        if answer not in answers:
            answers.append(answer)
    return answers, ""


class CLIInteractionHandler:
    def __init__(self, timeout: float = 90.0, output: Callable[[str], None] | None = None):
        self.timeout = float(timeout)
        self._output = output or print
        self._pending: queue.Queue[tuple[str, Any]] | None = None
        self._lock = threading.Lock()

    def route(self, kind: str, payload: Any) -> bool:
        if kind not in {"user", "eof"}:
            return False
        with self._lock:
            if self._pending is None:
                return False
            self._pending.put((kind, payload))
            return True

    def close(self) -> None:
        with self._lock:
            if self._pending is not None:
                self._pending.put(("eof", None))

    def ask(self, questions: list[dict[str, Any]]) -> dict[str, Any]:
        responses: queue.Queue[tuple[str, Any]] = queue.Queue()
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("已有正在等待回答的问题")
            self._pending = responses

        deadline = None if self.timeout <= 0 else time.monotonic() + self.timeout
        answers: dict[str, Any] = {}
        outcome = "submitted"
        notice = ""
        try:
            self._output("\nAgent 请求你回答以下问题：")
            for question in questions:
                self._print_question(question)
                while True:
                    kind, payload = self._next_response(responses, deadline)
                    if kind == "timeout":
                        outcome = "timed_out"
                        notice = "等待回答超时，请根据已有信息继续。"
                        break
                    if kind == "eof":
                        outcome = "cancelled"
                        notice = "输入已关闭，本次提问取消。"
                        break

                    text = str(payload).strip()
                    if text == "/cancel":
                        outcome = "cancelled"
                        notice = "用户取消了本次提问。"
                        break
                    if not text or text == "/skip":
                        answers[question["qid"]] = None
                        break
                    answer, error = _parse_answer(text, question)
                    if error:
                        self._output(error)
                        continue
                    answers[question["qid"]] = answer
                    break
                if outcome != "submitted":
                    break

            result: dict[str, Any] = {"answers": answers, "outcome": outcome}
            if notice:
                result["notice"] = notice
            return result
        finally:
            with self._lock:
                if self._pending is responses:
                    self._pending = None

    def _print_question(self, question: dict[str, Any]) -> None:
        self._output(f"\n{question['question']}")
        for index, option in enumerate(question["options"], 1):
            self._output(f"  {index}. {option['label']} — {option['description']}")
        if question["multi_select"]:
            self._output("多选请用逗号分隔编号、选项或自定义内容。")
        else:
            self._output("请输入编号、选项原文或自定义回答。")
        self._output("输入 /skip 跳过，/cancel 取消提问。")

    @staticmethod
    def _next_response(
        responses: queue.Queue[tuple[str, Any]], deadline: float | None
    ) -> tuple[str, Any]:
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return "timeout", None
            wait = 1.0 if remaining is None else min(1.0, remaining)
            try:
                return responses.get(timeout=wait)
            except queue.Empty:
                continue
