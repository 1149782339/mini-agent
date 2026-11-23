"""run_command：非密钥环境继承、退出码、超时杀进程、cwd 约束。"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from mini_agent.jobs import TaskRegistry
from mini_agent.tools.shell import ShellTools

ROOT = Path(__file__).resolve().parents[1]


def test_run_command_basic(tmp_path: Path) -> None:
    os.environ["MINI_TEST_VAR"] = "inherit-ok"
    shell = ShellTools(tmp_path, default_timeout=10)

    out = shell.run_command('echo "MINI_TEST_VAR=$MINI_TEST_VAR"; echo "HOME_SET=$HOME"')
    assert "MINI_TEST_VAR=inherit-ok" in out
    assert "HOME_SET=" in out
    assert "exit=0" in out

    out = shell.run_command("echo 正常输出; exit 3")
    assert "exit=3" in out
    assert "正常输出" in out

    out = shell.run_command("echo hello | tr a-z A-Z && echo 多命令OK")
    assert "HELLO" in out
    assert "多命令OK" in out
    assert "exit=0" in out


def test_run_command_timeout_kills_tree(tmp_path: Path) -> None:
    shell = ShellTools(tmp_path, default_timeout=10, hard_timeout=30, jobs=TaskRegistry(allow_notify=False))
    started = time.monotonic()
    out = shell.run_command("echo 开始; sleep 30; echo 不该到这里", timeout=2)
    elapsed = time.monotonic() - started
    assert "TIMEOUT" in out
    assert "不该到这里" not in out
    assert elapsed < 15


def test_run_command_cwd_must_stay_inside_workdir(tmp_path: Path) -> None:
    inside = tmp_path / "src"
    inside.mkdir()
    shell = ShellTools(tmp_path, default_timeout=10)
    out = shell.run_command("pwd", cwd="src")
    assert "exit=0" in out
    out = shell.run_command("pwd", cwd="/etc")
    assert out.startswith("ERROR:")
    assert "越界" in out


def test_timeout_kills_process_before_marker(tmp_path: Path) -> None:
    """子进程睡 5 秒后写标记文件；超时 1 秒，标记文件不应出现。"""
    work = tmp_path / "timeout_check"
    work.mkdir()
    marker = work / "marker.txt"
    exe = f'"{sys.executable.replace(chr(92), "/")}"'
    command = (
        f"{exe} -c \"import time, pathlib; time.sleep(5); "
        f"pathlib.Path('marker.txt').write_text('alive')\""
    )
    shell = ShellTools(work, default_timeout=10, jobs=TaskRegistry(allow_notify=False))
    started = time.monotonic()
    result = shell.run_command(command, timeout=1)
    elapsed = time.monotonic() - started
    # 等一会儿，确认被杀的进程没有晚写文件
    time.sleep(1.5)
    assert "TIMEOUT" in result
    assert not marker.exists(), "超时后进程仍写了标记文件，说明进程树没被杀掉"
    assert elapsed < 8
