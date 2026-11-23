"""模型可见提示词的 Markdown 资产。

提示词作为独立文件维护，便于审阅与 diff；工具描述仍留在各自模块，
因为它们与 schema、条件分支耦合。
"""
from __future__ import annotations

from importlib.resources import files


def load_prompt(name: str) -> str:
    """读取 prompts/ 下的提示词文件；缺失时直接抛错，不静默降级。"""
    return files(__package__).joinpath(name).read_text(encoding="utf-8")
