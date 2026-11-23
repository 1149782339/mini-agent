"""工具包导出 + 默认工具组装。"""
from __future__ import annotations

from pathlib import Path

from .base import (
    FnTool,
    Tool,
    ToolContext,
    ToolError,
    ToolOutput,
    ToolRegistry,
    ValidationResult,
    decode_output,
    semantic_bool,
    semantic_int,
)
from .ask_user import AskUserTool
from .edit import EditTool
from .filestate import FileState, FileStateCache, FileToolContext
from .glob_tool import GlobTool
from .grep_tool import GrepTool
from .ls import LsTool
from .read import ReadTool
from .rm_mv import MVTool, RMTool
from .shell import ShellTools
from .subagent import SubagentTool, TaskManageTool
from .write import WriteTool

__all__ = [
    "FnTool", "Tool", "ToolContext", "ToolError", "ToolOutput", "ToolRegistry",
    "ValidationResult", "decode_output", "semantic_bool", "semantic_int",
    "FileState", "FileStateCache", "FileToolContext",
    "AskUserTool", "ReadTool", "EditTool", "WriteTool", "RMTool", "MVTool",
    "GlobTool", "GrepTool", "LsTool", "ShellTools", "SubagentTool", "TaskManageTool",
    "build_tools",
]


def build_tools(
    cfg,
    spill_dir: Path | None = None,
    task_store=None,
    session_id: str | None = None,
    interaction_handler=None,
) -> ToolRegistry:
    """按配置组装注册表：文件/搜索工具、AskUser 与可选 Shell、Task。"""
    fctx = FileToolContext(cfg.workdir)
    if spill_dir is not None:
        registry = ToolRegistry(
            max_output=cfg.max_tool_output, spill_dir=spill_dir, context=fctx
        )
    else:
        registry = ToolRegistry(max_output=cfg.max_tool_output, context=fctx)
    for tool in (ReadTool(), EditTool(), WriteTool(), RMTool(), MVTool(), GlobTool(), GrepTool(), LsTool()):
        registry.register(tool)
    registry.register(AskUserTool(interaction_handler))
    from ..jobs import TaskRegistry
    jobs = TaskRegistry(task_store=task_store, owner_session_id=session_id)
    registry.jobs = jobs
    registry.register(TaskManageTool(jobs))
    if cfg.enable_shell:
        for tool in ShellTools(
            cfg.workdir,
            shell=cfg.shell,
            jobs=jobs,
            foreground_block_budget=cfg.foreground_block_budget,
        ).tools():
            registry.register(tool)
    if cfg.enable_subagent:
        registry.register(SubagentTool(jobs=jobs, config=cfg))
    return registry
