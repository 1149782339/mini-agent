"""验证 run_command：环境变量继承、退出码、超时杀进程、cwd 约束。"""
from __future__ import annotations

import os
from pathlib import Path

from mini_agent.tools import ShellTools

ROOT = Path(__file__).resolve().parents[1]

# 故意设置一个自定义变量，确认子进程能看见
os.environ["MINI_TEST_VAR"] = "inherit-ok"

shell = ShellTools(ROOT, default_timeout=10)

print("=== 1. 环境变量继承 ===")
print(shell.run_command('echo "MINI_TEST_VAR=$MINI_TEST_VAR"; echo "HOME=$HOME"; echo "PATH前段=${PATH%%:*}"'))

print("\n=== 2. 退出码传递 ===")
print(shell.run_command("echo 正常输出; exit 3"))

print("\n=== 3. 超时被杀 ===")
print(shell.run_command("echo 开始; sleep 30; echo 不该到这里", timeout=2))

print("\n=== 4. cwd 约束 ===")
print(shell.run_command("pwd", cwd="src/mini_agent"))
print(shell.run_command("pwd", cwd="/etc"))

print("\n=== 5. 管道与 && ===")
print(shell.run_command("echo hello | tr a-z A-Z && echo 多命令OK"))
