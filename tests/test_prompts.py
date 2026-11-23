"""提示词资产：文件可定位、内容完整、没有未渲染的模板占位符。"""
from importlib.resources import files

from mini_agent.config import SYSTEM_PROMPT
from mini_agent.tools.subagent import CHILD_SYSTEM_PROMPT


def test_prompt_files_exist() -> None:
    base = files("mini_agent.prompts")
    for name in ("system.md", "subagent.md"):
        assert base.joinpath(name).is_file()


def test_system_prompt_content() -> None:
    assert SYSTEM_PROMPT.startswith("【PERSONA_LOAD】")
    assert "自动转后台" in SYSTEM_PROMPT
    assert "{" not in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.endswith("\n")


def test_subagent_prompt_content() -> None:
    assert CHILD_SYSTEM_PROMPT.startswith("你是一个被主代理派发的子代理")
    assert "返回格式" in CHILD_SYSTEM_PROMPT
    assert "Task" not in CHILD_SYSTEM_PROMPT
    assert CHILD_SYSTEM_PROMPT.endswith("\n")
