"""子进程环境不得包含 API key 等密钥。"""
from __future__ import annotations

import os
import textwrap

from mini_agent.envutil import is_secret_env_name, sanitize_subprocess_env
from mini_agent.tools.shell import ShellTools
from mini_agent.tools.subagent import CHILD_MAX_STEPS, build_child_env


def test_secret_name_detection() -> None:
    assert is_secret_env_name("MINI_AGENT_API_KEY")
    assert is_secret_env_name("ARK_API_KEY")
    assert is_secret_env_name("OPENAI_API_KEY")
    assert is_secret_env_name("ANTHROPIC_API_KEY")
    assert is_secret_env_name("AWS_SECRET_ACCESS_KEY")
    assert is_secret_env_name("AWS_ACCESS_KEY_ID")
    assert is_secret_env_name("GITHUB_TOKEN")
    assert not is_secret_env_name("PATH")
    assert not is_secret_env_name("HOME")
    assert not is_secret_env_name("MINI_AGENT_MAX_STEPS")
    assert not is_secret_env_name("MINI_AGENT_MODEL")
    assert not is_secret_env_name("MINI_AGENT_DEBUG")
    assert not is_secret_env_name("PYTHONPATH")
    assert not is_secret_env_name("TERM")


def test_sanitize_strips_keys_keeps_operational_vars() -> None:
    # 只用占位符，不断言或打印具体密钥值
    src = {
        "PATH": "/usr/bin",
        "HOME": "/tmp/home",
        "MINI_TEST_VAR": "inherit-ok",
        "MINI_AGENT_MAX_STEPS": "150",
        "MINI_AGENT_API_KEY": "placeholder-should-be-stripped",
        "ARK_API_KEY": "placeholder-should-be-stripped",
        "OPENAI_API_KEY": "placeholder-should-be-stripped",
        "ANTHROPIC_API_KEY": "placeholder-should-be-stripped",
        "AWS_SECRET_ACCESS_KEY": "placeholder-should-be-stripped",
    }
    out = sanitize_subprocess_env(src)
    assert out["PATH"] == "/usr/bin"
    assert out["HOME"] == "/tmp/home"
    assert out["MINI_TEST_VAR"] == "inherit-ok"
    assert out["MINI_AGENT_MAX_STEPS"] == "150"
    for key in (
        "MINI_AGENT_API_KEY",
        "ARK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
    ):
        assert key not in out


def test_shell_child_env_has_no_api_keys(tmp_path) -> None:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin"),
        "HOME": str(tmp_path),
        "MINI_TEST_VAR": "inherit-ok",
        "MINI_AGENT_API_KEY": "placeholder-should-be-stripped",
        "ARK_API_KEY": "placeholder-should-be-stripped",
    }
    shell = ShellTools(tmp_path, env=env)
    assert "MINI_AGENT_API_KEY" not in shell.env
    assert "ARK_API_KEY" not in shell.env
    assert shell.env.get("MINI_TEST_VAR") == "inherit-ok"
    assert shell.env.get("PYTHONUTF8") == "1"

    probe = tmp_path / "probe_env.py"
    probe.write_text(
        textwrap.dedent(
            """\
            import os
            names = ("MINI_AGENT_API_KEY", "ARK_API_KEY")
            print("present=" + ",".join(n for n in names if n in os.environ))
            print("mini_test=" + os.environ.get("MINI_TEST_VAR", ""))
            """
        ),
        encoding="utf-8",
    )
    result = shell.run_command(f'python "{probe.as_posix()}"')
    assert "present=" in result
    present_line = [line for line in result.splitlines() if line.startswith("present=")]
    assert present_line, result
    assert present_line[0] == "present="
    assert "mini_test=inherit-ok" in result


def test_task_child_env_scrubs_keys_keeps_max_steps() -> None:
    src = {
        "PATH": "/usr/bin",
        "HOME": "/tmp",
        "MINI_AGENT_DEBUG": "1",
        "MINI_AGENT_API_KEY": "placeholder-should-be-stripped",
        "ARK_API_KEY": "placeholder-should-be-stripped",
        "OPENAI_API_KEY": "placeholder-should-be-stripped",
    }
    env = build_child_env(src)
    assert env["MINI_AGENT_MAX_STEPS"] == str(CHILD_MAX_STEPS)
    assert env.get("MINI_AGENT_DEBUG") == "1"
    assert "PATH" in env
    for key in ("MINI_AGENT_API_KEY", "ARK_API_KEY", "OPENAI_API_KEY"):
        assert key not in env
