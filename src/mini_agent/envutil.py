"""子进程环境清理：run_command / Task 启动子进程时去掉密钥变量。

mini-agent 进程本身仍从环境变量或 .env 读取 API key。
这里只影响派出去的子进程，避免命令或孙进程直接继承密钥。
"""
from __future__ import annotations

import os

# 明确名单，避免漏掉不按常见后缀命名的密钥
_SECRET_EXACT = {
    "MINI_AGENT_API_KEY",
    "ARK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASH_SCOPE_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "MOONSHOT_API_KEY",
    "ZHIPU_API_KEY",
    "XAI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
}

_SECRET_SUFFIXES = (
    "_API_KEY",
    "_SECRET_ACCESS_KEY",
    "_SECRET_KEY",
    "_ACCESS_TOKEN",
    "_AUTH_TOKEN",
    "_PRIVATE_KEY",
    "_PASSWORD",
    "_PASSWD",
    "_SECRET",
    "_TOKEN",
)


def is_secret_env_name(name: str) -> bool:
    """变量名是否像密钥。PATH / HOME / MINI_AGENT_MAX_STEPS 等不会命中。"""
    upper = name.upper()
    if upper in _SECRET_EXACT:
        return True
    return any(upper.endswith(suffix) for suffix in _SECRET_SUFFIXES)


def sanitize_subprocess_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """复制一份环境并去掉密钥，供 run_command / Task 子进程使用。

    保留 PATH、HOME、虚拟环境、MINI_AGENT_MAX_STEPS 等运行所需变量。
    """
    source = os.environ if env is None else env
    cleaned: dict[str, str] = {}
    for key, value in source.items():
        if is_secret_env_name(key):
            continue
        cleaned[key] = value
    return cleaned
