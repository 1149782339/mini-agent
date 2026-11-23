"""配置加载：来源优先级、TOML 校验、.env 查找与无副作用。"""
from __future__ import annotations

import os

import pytest

from mini_agent.config import Config, load_config


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MINI_AGENT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _user_config(tmp_path, text: str):
    path = tmp_path / "home" / ".config" / "mini-agent" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_any_config(tmp_path) -> None:
    cfg = load_config()
    assert cfg.model == Config.model
    assert cfg.api_key == ""
    assert cfg.show_reasoning is True


def test_precedence_env_over_dotenv_over_toml(tmp_path, monkeypatch) -> None:
    _user_config(tmp_path, 'model = "toml-model"\nmax_steps = 7\n')
    _write(tmp_path, ".env", "MINI_AGENT_MODEL=dotenv-model\nMINI_AGENT_API_KEY=k\n")

    cfg = load_config()
    assert cfg.model == "dotenv-model"
    assert cfg.api_key == "k"
    assert cfg.max_steps == 7

    monkeypatch.setenv("MINI_AGENT_MODEL", "env-model")
    assert load_config().model == "env-model"


def test_toml_types_and_vision_auto(tmp_path) -> None:
    _user_config(
        tmp_path,
        "\n".join(
            [
                "show_reasoning = false",
                "enable_shell = false",
                "context_window = 1234",
                "compact_ratio = 0.5",
                'vision = "auto"',
                'workdir = "sub"',
            ]
        ),
    )
    cfg = load_config()
    assert cfg.show_reasoning is False
    assert cfg.enable_shell is False
    assert cfg.context_window == 1234
    assert cfg.compact_ratio == 0.5
    assert cfg.vision is None
    assert cfg.workdir == (tmp_path / "sub").resolve()


def test_unknown_toml_key_fails_loud(tmp_path) -> None:
    path = _write(tmp_path, "config.toml", 'modle = "typo"\n')
    with pytest.raises(SystemExit, match="未知键"):
        load_config(path)


def test_toml_rejects_api_key(tmp_path) -> None:
    path = _write(tmp_path, "config.toml", 'api_key = "secret"\n')
    with pytest.raises(SystemExit, match="不保存 api_key"):
        load_config(path)


def test_explicit_missing_config_fails_loud(tmp_path) -> None:
    with pytest.raises(SystemExit, match="配置文件不存在"):
        load_config(tmp_path / "nope.toml")


def test_bad_integer_names_the_field(monkeypatch) -> None:
    monkeypatch.setenv("MINI_AGENT_CONTEXT_WINDOW", "abc")
    with pytest.raises(SystemExit, match="context_window"):
        load_config()


def test_no_hermes_fallback(tmp_path) -> None:
    hermes = tmp_path / "home" / ".hermes"
    hermes.mkdir(parents=True)
    (hermes / ".env").write_text("ARK_API_KEY=should-not-be-read\n", encoding="utf-8")
    assert load_config().api_key == ""


def test_env_file_found_upwards(tmp_path, monkeypatch) -> None:
    _write(tmp_path, ".env", "MINI_AGENT_MODEL=from-parent\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert load_config().model == "from-parent"


def test_load_does_not_mutate_environ(tmp_path) -> None:
    _write(tmp_path, ".env", "MINI_AGENT_MODEL=dotenv-model\n")
    load_config()
    assert "MINI_AGENT_MODEL" not in os.environ


def test_validate_message_points_to_env() -> None:
    with pytest.raises(SystemExit) as exc:
        Config().validate()
    message = str(exc.value)
    assert "MINI_AGENT_API_KEY" in message
    assert ".env" in message
