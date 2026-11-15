"""离线自检：run_command 超时后是否真的杀掉了子进程树（不调用模型）。

子进程睡 5 秒后写 marker.txt，超时设 1 秒：
- 输出里应带 `[TIMEOUT...]` 标记；
- 进程树被正确清理时，marker.txt 永远不该出现（证明进程真的死了，
  而不是"被放弃但不回收"）。

Windows 上走 Job Object，POSIX 上走进程组；两边的判定标准完全一致。
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from mini_agent.tools import ShellTools, ToolRegistry

SLEEP = 5
WORK = Path(__file__).resolve().parents[1] / "sandbox" / "timeout_check"


def main() -> int:
    # bash 里反斜杠是转义符，Windows 路径必须换成正斜杠并加引号
    exe = f'"{sys.executable.replace(chr(92), "/")}"'
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    marker = WORK / "marker.txt"
    try:
        registry = ToolRegistry()
        for tool in ShellTools(WORK).tools():
            registry.register(tool)

        child = (
            f'{exe} -c "import time,pathlib;'
            f"time.sleep({SLEEP});pathlib.Path('marker.txt').write_text('alive')\""
        )
        result = registry.dispatch("run_command", {"command": child, "timeout": 1})
        print(result.replace("\n", " | ")[:400])

        ok_timeout = "TIMEOUT" in result and result.startswith("exit=")
        time.sleep(SLEEP + 1)  # 给残留进程足够时间暴露自己
        ok_cleanup = not marker.exists()

        print(f"超时被识别: {ok_timeout}")
        print(f"子进程被回收（marker.txt 未生成）: {ok_cleanup}")
        ok = ok_timeout and ok_cleanup
        print("TIMEOUT-CHECK:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
