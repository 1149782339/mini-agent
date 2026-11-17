"""RM / MV 工具与 run_command 对 rm/mv 的拦截。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mini_agent.config import Config
from mini_agent.tools import build_tools
from mini_agent.tools.shell import ShellTools, direct_rm_mv_invocation


def _reg(workdir: Path, *, shell: bool = True):
    cfg = Config(workdir=workdir, enable_shell=shell, enable_subagent=False)
    return build_tools(cfg)


def _d(reg, name, **kw) -> str:
    return reg.dispatch(name, json.dumps(kw, ensure_ascii=False))


def test_rm_rejects_relative_and_home_paths(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("x\n", encoding="utf-8")
    reg = _reg(tmp_path)
    for raw in ("a.txt", "./a.txt", "../a.txt", "~/a.txt", "~/xxx"):
        out = _d(reg, "RM", path=raw)
        assert out.startswith("ERROR:"), out
        assert "字面绝对路径" in out, out
        assert target.exists()


def test_mv_rejects_relative_and_home_paths(tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("x\n", encoding="utf-8")
    reg = _reg(tmp_path)
    abs_src = str(src)
    abs_dest = str(tmp_path / "b.txt")
    out = _d(reg, "MV", source="a.txt", destination=abs_dest)
    assert out.startswith("ERROR:") and "字面绝对路径" in out, out
    out = _d(reg, "MV", source=abs_src, destination="b.txt")
    assert out.startswith("ERROR:") and "字面绝对路径" in out, out
    out = _d(reg, "MV", source="~/a.txt", destination=abs_dest)
    assert out.startswith("ERROR:") and "字面绝对路径" in out, out
    assert src.exists()
    assert not (tmp_path / "b.txt").exists()


def test_rm_rejects_absolute_path_outside_workdir(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"outside_rm_{tmp_path.name}.txt"
    outside.write_text("keep\n", encoding="utf-8")
    try:
        reg = _reg(tmp_path)
        out = _d(reg, "RM", path=str(outside))
        assert out.startswith("ERROR:") and "越界" in out, out
        assert outside.exists()
    finally:
        outside.unlink(missing_ok=True)


def test_mv_rejects_absolute_path_outside_workdir(tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("x\n", encoding="utf-8")
    outside = tmp_path.parent / f"outside_mv_{tmp_path.name}.txt"
    if outside.exists():
        outside.unlink()
    reg = _reg(tmp_path)
    out = _d(reg, "MV", source=str(src), destination=str(outside))
    assert out.startswith("ERROR:") and "越界" in out, out
    assert src.exists()
    assert not outside.exists()


def test_rm_and_mv_reject_workdir_itself(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("k\n", encoding="utf-8")
    reg = _reg(tmp_path)
    out = _d(reg, "RM", path=str(tmp_path), recursive=True)
    assert out.startswith("ERROR:") and "工作目录本身" in out, out
    assert tmp_path.is_dir()
    assert (tmp_path / "keep.txt").exists()

    dest = tmp_path / "elsewhere"
    dest.mkdir()
    out = _d(reg, "MV", source=str(tmp_path), destination=str(dest / "moved"))
    assert out.startswith("ERROR:") and "工作目录本身" in out, out
    assert tmp_path.is_dir()


def test_rm_directory_requires_recursive(tmp_path: Path) -> None:
    folder = tmp_path / "dir"
    folder.mkdir()
    (folder / "inner.txt").write_text("in\n", encoding="utf-8")
    reg = _reg(tmp_path)
    out = _d(reg, "RM", path=str(folder))
    assert out.startswith("ERROR:"), out
    assert "recursive" in out and "目录" in out, out
    assert folder.is_dir()
    assert (folder / "inner.txt").exists()


def test_rm_recursive_deletes_directory_inside_workdir(tmp_path: Path) -> None:
    folder = tmp_path / "dir"
    folder.mkdir()
    (folder / "inner.txt").write_text("in\n", encoding="utf-8")
    (folder / "sub").mkdir()
    (folder / "sub" / "deep.txt").write_text("deep\n", encoding="utf-8")
    reg = _reg(tmp_path)
    out = _d(reg, "RM", path=str(folder), recursive=True)
    assert "已递归删除" in out, out
    assert not folder.exists()


def test_rm_file_does_not_need_recursive(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("x\n", encoding="utf-8")
    reg = _reg(tmp_path)
    out = _d(reg, "RM", path=str(target))
    assert "已删除文件" in out, out
    assert not target.exists()


def test_mv_refuses_to_overwrite_existing_destination(tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    dest = tmp_path / "b.txt"
    src.write_text("src\n", encoding="utf-8")
    dest.write_text("dest\n", encoding="utf-8")
    reg = _reg(tmp_path)
    out = _d(reg, "MV", source=str(src), destination=str(dest))
    assert out.startswith("ERROR:") and "拒绝覆盖" in out, out
    assert src.read_text(encoding="utf-8") == "src\n"
    assert dest.read_text(encoding="utf-8") == "dest\n"


def test_mv_moves_when_destination_is_free(tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    dest = tmp_path / "b.txt"
    src.write_text("src\n", encoding="utf-8")
    reg = _reg(tmp_path)
    out = _d(reg, "MV", source=str(src), destination=str(dest))
    assert "已移动" in out, out
    assert not src.exists()
    assert dest.read_text(encoding="utf-8") == "src\n"


def test_rm_rejects_symlink_escaping_workdir(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"escape_rm_{tmp_path.name}.txt"
    outside.write_text("secret\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
        reg = _reg(tmp_path)
        out = _d(reg, "RM", path=str(link))
        assert out.startswith("ERROR:") and "越界" in out, out
        assert link.is_symlink()
        assert outside.read_text(encoding="utf-8") == "secret\n"
    finally:
        outside.unlink(missing_ok=True)


def test_rm_unlinks_symlink_inside_workdir_without_deleting_target(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real.txt"
    real.write_text("keep\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    reg = _reg(tmp_path)
    out = _d(reg, "RM", path=str(link))
    assert "已删除符号链接" in out, out
    assert not link.exists()
    assert real.read_text(encoding="utf-8") == "keep\n"


def test_shell_rejects_direct_rm_and_mv(tmp_path: Path) -> None:
    marker = tmp_path / "victim.txt"
    marker.write_text("stay\n", encoding="utf-8")
    shell = ShellTools(tmp_path, default_timeout=10)
    for command in (
        "rm victim.txt",
        "rm -rf victim.txt",
        "/bin/rm victim.txt",
        "mv victim.txt gone.txt",
        "/usr/bin/mv victim.txt gone.txt",
        "ls && rm victim.txt",
    ):
        out = shell.run_command(command)
        assert out.startswith("ERROR:"), (command, out)
        assert "请改用" in out, (command, out)
        if "rm" in command.split()[0] or command.startswith("/bin/rm") or "&& rm" in command:
            assert "RM" in out, (command, out)
        if "mv" in command.split()[0] or command.startswith("/usr/bin/mv"):
            assert "MV" in out, (command, out)
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "stay\n"


def test_shell_detector_allows_non_direct_invocations() -> None:
    assert direct_rm_mv_invocation("echo rm") is None
    assert direct_rm_mv_invocation("echo 'rm file'") is None
    assert direct_rm_mv_invocation("git rm file") is None
    assert direct_rm_mv_invocation("find . -delete") is None
    assert direct_rm_mv_invocation("unlink file") is None
    assert direct_rm_mv_invocation("python -c 'import os; os.remove(\"x\")'") is None
    assert direct_rm_mv_invocation("rm file") == "rm"
    assert direct_rm_mv_invocation("/bin/rm -rf x") == "rm"
    assert direct_rm_mv_invocation("/usr/bin/mv a b") == "mv"
    assert direct_rm_mv_invocation("sudo rm -rf x") == "rm"


def test_write_script_containing_rm_is_still_allowed(tmp_path: Path) -> None:
    reg = _reg(tmp_path, shell=True)
    content = "#!/bin/sh\nrm -rf /tmp/cleanup-demo\n"
    script = tmp_path / "cleanup.sh"
    out = _d(reg, "Write", file_path="cleanup.sh", content=content)
    assert "已新建" in out, out
    assert script.read_text(encoding="utf-8") == content
