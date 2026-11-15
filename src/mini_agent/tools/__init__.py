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
from .edit import EditTool
from .filestate import FileState, FileStateCache, FileToolContext
from .glob_tool import GlobTool
from .grep_tool import GrepTool
from .read import ReadTool
from .shell import ShellTools
from .write import WriteTool

__all__ = [
    "FnTool", "Tool", "ToolContext", "ToolError", "ToolOutput", "ToolRegistry",
    "ValidationResult", "decode_output", "semantic_bool", "semantic_int",
    "FileState", "FileStateCache", "FileToolContext",
    "ReadTool", "EditTool", "WriteTool", "GlobTool", "GrepTool", "ShellTools",
    "build_tools",
]


def build_tools(cfg, spill_dir: Path | None = None) -> ToolRegistry:
    """按配置组装注册表：Read/Edit/Write/Glob/Grep + 可选 run_command。

    五个文件工具共享一个 FileToolContext（workdir + 读状态缓存）——
    read-before-write、staleness 闸、file_unchanged 去重都靠它串联。
    """
    fctx = FileToolContext(cfg.workdir)
    if spill_dir is not None:
        registry = ToolRegistry(
            max_output=cfg.max_tool_output, spill_dir=spill_dir, context=fctx
        )
    else:
        registry = ToolRegistry(max_output=cfg.max_tool_output, context=fctx)
    for tool in (ReadTool(), EditTool(), WriteTool(), GlobTool(), GrepTool()):
        registry.register(tool)
    if cfg.enable_shell:
        for tool in ShellTools(cfg.workdir, shell=cfg.shell).tools():
            registry.register(tool)
    return registry
