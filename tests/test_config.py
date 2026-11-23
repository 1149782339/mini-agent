"""配置加载：用户 TOML、环境变量覆盖、校验与无副作用。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mini_agent.config import Config, load_config, user_config_path


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MINI_AGENT_"):
            monkeypatch.delenv(key, raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
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


def test_user_config_path_is_dot_config_on_every_platform(tmp_path, monkeypatch) -> None:
    expected = tmp_path / "home" / ".config" / "mini-agent" / "config.toml"
    assert user_config_path() == expected

    override = tmp_path / "custom.toml"
    monkeypatch.setenv("MINI_AGENT_CONFIG", str(override))
    assert user_config_path() == override


def test_defaults_without_any_config(tmp_path) -> None:
    cfg = load_config()
    assert cfg.model == Config.model
    assert cfg.api_key == ""
    assert cfg.show_reasoning is True


def test_reads_connection_settings_from_user_toml(tmp_path) -> None:
    _user_config(
        tmp_path,
        "\n".join(
            [
                'api_key = " toml-key "',
                'base_url = "https://toml.example/v3"',
                'model = "toml-model"',
                "max_steps = 7",
            ]
        ),
    )

    cfg = load_config()
    assert cfg.api_key == "toml-key"
    assert cfg.base_url == "https://toml.example/v3"
    assert cfg.model == "toml-model"
    assert cfg.max_steps == 7


def test_precedence_env_over_toml_and_dotenv_ignored(tmp_path, monkeypatch) -> None:
    _user_config(
        tmp_path,
        'api_key = "toml-key"\nbase_url = "https://toml.example/v3"\nmodel = "toml-model"\n',
    )
    _write(
        tmp_path,
        ".env",
        "MINI_AGENT_API_KEY=dotenv-key\nMINI_AGENT_BASE_URL=https://dotenv.example/v3\n"
        "MINI_AGENT_MODEL=dotenv-model\n",
    )

    cfg = load_config()
    assert cfg.api_key == "toml-key"
    assert cfg.base_url == "https://toml.example/v3"
    assert cfg.model == "toml-model"

    monkeypatch.setenv("MINI_AGENT_API_KEY", "env-key")
    monkeypatch.setenv("MINI_AGENT_BASE_URL", "https://env.example/v3")
    monkeypatch.setenv("MINI_AGENT_MODEL", "env-model")
    cfg = load_config()
    assert cfg.api_key == "env-key"
    assert cfg.base_url == "https://env.example/v3"
    assert cfg.model == "env-model"


def test_dotenv_is_ignored_without_toml(tmp_path, monkeypatch) -> None:
    _write(tmp_path, ".env", "MINI_AGENT_MODEL=from-dotenv\nMINI_AGENT_API_KEY=k\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    cfg = load_config()
    assert cfg.model == Config.model
    assert cfg.api_key == ""


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


def test_load_does_not_mutate_environ(tmp_path) -> None:
    _user_config(tmp_path, 'model = "toml-model"\n')
    assert load_config().model == "toml-model"
    assert "MINI_AGENT_MODEL" not in os.environ


def test_validate_message_points_to_user_config() -> None:
    with pytest.raises(SystemExit) as exc:
        Config().validate()
    message = str(exc.value)
    assert str(user_config_path()) in message
    assert "api_key" in message
    assert "MINI_AGENT_API_KEY" in message


def test_example_config_is_valid() -> None:
    path = Path(__file__).resolve().parents[1] / "config.toml.example"
    cfg = load_config(path)
    assert cfg.api_key == ""
    assert cfg.base_url == Config.base_url
    assert cfg.model == Config.model
