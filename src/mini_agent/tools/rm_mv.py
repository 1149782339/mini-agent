"""RM / MV 工具：受管控的删除与移动。

路径必须是字面绝对路径（相对路径和 ~/xxx 一律拒绝，不会先展开再接受）。
解析软链接后的真实路径仍须落在工作目录内，且不得操作工作目录本身。
RM 默认不递归，删除目录必须显式 recursive=true。
MV 在目标已存在时拒绝覆盖。
"""
from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import (
    Tool,
    ToolContext,
    ToolError,
    ToolOutput,
    ValidationResult,
    semantic_bool,
)
from .filestate import FileToolContext, is_unc


def is_literal_absolute(raw: str) -> bool:
    """只看字面输入是否为绝对路径，不先 expanduser / normalize。"""
    if not raw or raw.startswith("~"):
        return False
    return Path(raw).is_absolute()


@dataclass(frozen=True)
class BoundPath:
    """given 是用户给出的绝对路径；real 是 resolve() 后的真实路径。"""

    given: Path
    real: Path


def bind_literal_abs_in_workdir(
    workdir: Path, raw: str, *, label: str
) -> BoundPath:
    """校验字面绝对路径，并约束 resolve 后仍在 workdir 内、不是 workdir 本身。"""
    if not isinstance(raw, str) or not raw.strip():
        raise ToolError(f"{label} 必须是非空字符串", 1)
    text = raw.strip()
    if is_unc(text):
        raise ToolError(
            f"不支持 UNC 路径 {raw!r}（防止 SMB/NTLM 凭据泄漏）", 2
        )
    if not is_literal_absolute(text):
        raise ToolError(
            f"{label} 必须是字面绝对路径，不接受相对路径或 ~/xxx：{raw!r}",
            3,
        )
    given = Path(text)
    real = given.resolve()
    root = workdir.resolve()
    try:
        real.relative_to(root)
    except ValueError:
        raise ToolError(
            f"越界：{real} 不在工作目录 {root} 内", 4
        ) from None
    if real == root:
        raise ToolError(f"禁止删除或移动工作目录本身：{real}", 5)
    return BoundPath(given=given, real=real)


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


class RMTool(Tool):
    name = "RM"
    description = (
        "删除工作目录内的文件或目录。\n"
        "用法：\n"
        "- path 必须是字面绝对路径（POSIX 以 / 开头，Windows 为盘符路径）。"
        "相对路径和 ~/xxx 一律拒绝，不会先展开再判断。\n"
        "- 解析软链接后的真实路径必须仍在工作目录内；禁止越界，"
        "也禁止删除工作目录本身。\n"
        "- 默认不递归。删除目录必须显式设置 recursive=true；删文件不需要。"
        "若目标是目录且未设 recursive，会明确拒绝。\n"
        "- 不要用 run_command 执行 rm（包括 /bin/rm、/usr/bin/rm），请用本工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "要删除的字面绝对路径，必须落在工作目录内；"
                    "不接受相对路径或 ~/xxx"
                ),
            },
            "recursive": {
                "type": "boolean",
                "description": (
                    "删除目录时必须为 true，默认 false。删文件不需要此标志"
                ),
                "default": False,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def validate_input(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        if not isinstance(ctx, FileToolContext):
            return ValidationResult.fail("RM 需要 FileToolContext", 0)
        try:
            bind_literal_abs_in_workdir(ctx.workdir, args.get("path"), label="path")
        except ToolError as exc:
            return ValidationResult.fail(str(exc), error_code=exc.error_code)
        return ValidationResult.pass_()

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        if not isinstance(ctx, FileToolContext):
            raise ToolError("RM 需要 FileToolContext", 0)
        bound = bind_literal_abs_in_workdir(
            ctx.workdir, args["path"], label="path"
        )
        recursive = semantic_bool(args.get("recursive"), False)
        target = bound.given

        with ctx.exclusive_path(target):
            try:
                st = target.lstat()
            except FileNotFoundError:
                raise ToolError(f"不存在：{target}", 6) from None
            except OSError as exc:
                raise ToolError(f"无法访问 {target}：{exc}", 8) from exc

            try:
                if stat.S_ISLNK(st.st_mode):
                    target.unlink()
                    ctx.state.delete(bound.real)
                    ctx.state.delete(target)
                    return ToolOutput(
                        f"已删除符号链接 {target}",
                        {"type": "rm", "path": str(target), "kind": "symlink"},
                    )
                if stat.S_ISDIR(st.st_mode):
                    if not recursive:
                        raise ToolError(
                            f"目标是目录，删除目录必须显式设置 recursive=true：{target}",
                            7,
                        )
                    shutil.rmtree(target)
                    ctx.state.delete_tree(target)
                    ctx.state.delete_tree(bound.real)
                    return ToolOutput(
                        f"已递归删除目录 {target}",
                        {
                            "type": "rm",
                            "path": str(target),
                            "kind": "directory",
                            "recursive": True,
                        },
                    )
                target.unlink()
                ctx.state.delete(target)
                ctx.state.delete(bound.real)
                return ToolOutput(
                    f"已删除文件 {target}",
                    {"type": "rm", "path": str(target), "kind": "file"},
                )
            except ToolError:
                raise
            except OSError as exc:
                raise ToolError(f"删除失败：{target}：{exc}", 8) from exc


class MVTool(Tool):
    name = "MV"
    description = (
        "移动或重命名工作目录内的文件或目录。\n"
        "用法：\n"
        "- source 和 destination 都必须是字面绝对路径（POSIX 以 / 开头，"
        "Windows 为盘符路径）。相对路径和 ~/xxx 一律拒绝，不会先展开再判断。\n"
        "- 解析软链接后的真实路径必须仍在工作目录内；禁止越界，"
        "也禁止移动工作目录本身。\n"
        "- destination 必须是最终路径（不要只写目标目录）。"
        "若该路径已经存在（文件、目录或符号链接），一律拒绝覆盖。\n"
        "- 不要用 run_command 执行 mv（包括 /bin/mv、/usr/bin/mv），请用本工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": (
                    "要移动的字面绝对路径，必须落在工作目录内；"
                    "不接受相对路径或 ~/xxx"
                ),
            },
            "destination": {
                "type": "string",
                "description": (
                    "移动后的字面绝对最终路径，必须落在工作目录内；"
                    "已存在则拒绝覆盖，不接受相对路径或 ~/xxx"
                ),
            },
        },
        "required": ["source", "destination"],
        "additionalProperties": False,
    }

    def validate_input(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        if not isinstance(ctx, FileToolContext):
            return ValidationResult.fail("MV 需要 FileToolContext", 0)
        try:
            bind_literal_abs_in_workdir(
                ctx.workdir, args.get("source"), label="source"
            )
            bind_literal_abs_in_workdir(
                ctx.workdir, args.get("destination"), label="destination"
            )
        except ToolError as exc:
            return ValidationResult.fail(str(exc), error_code=exc.error_code)
        return ValidationResult.pass_()

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        if not isinstance(ctx, FileToolContext):
            raise ToolError("MV 需要 FileToolContext", 0)
        src = bind_literal_abs_in_workdir(
            ctx.workdir, args["source"], label="source"
        )
        dest = bind_literal_abs_in_workdir(
            ctx.workdir, args["destination"], label="destination"
        )

        # 先锁源再锁目标，按路径字符串排序避免与并发工具互相死锁
        first, second = sorted(
            (src.given, dest.given), key=lambda p: os.path.normcase(str(p))
        )
        with ctx.exclusive_path(first):
            with ctx.exclusive_path(second) if second != first else _nullcontext():
                if not _lexists(src.given):
                    raise ToolError(f"源不存在：{src.given}", 6)
                if _lexists(dest.given):
                    raise ToolError(
                        f"目标已存在，拒绝覆盖：{dest.given}", 7
                    )
                parent = dest.given.parent
                if not parent.exists() or not parent.is_dir():
                    raise ToolError(
                        f"目标的父目录不存在：{parent}", 8
                    )
                # 不能把目录移动到它自己内部
                try:
                    st = src.given.lstat()
                except OSError as exc:
                    raise ToolError(f"无法访问 {src.given}：{exc}", 10) from exc
                if stat.S_ISDIR(st.st_mode):
                    try:
                        dest.real.relative_to(src.real)
                    except ValueError:
                        pass
                    else:
                        raise ToolError(
                            f"不能把目录移动到它自己内部：{src.given} → {dest.given}",
                            9,
                        )
                try:
                    shutil.move(str(src.given), str(dest.given))
                except OSError as exc:
                    raise ToolError(
                        f"移动失败：{src.given} → {dest.given}：{exc}", 10
                    ) from exc

                ctx.state.delete(src.given)
                ctx.state.delete(src.real)
                if stat.S_ISDIR(st.st_mode):
                    ctx.state.delete_tree(src.given)
                    ctx.state.delete_tree(src.real)

        return ToolOutput(
            f"已移动 {src.given} → {dest.given}",
            {
                "type": "mv",
                "source": str(src.given),
                "destination": str(dest.given),
            },
        )


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
