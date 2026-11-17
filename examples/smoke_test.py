"""集成冒烟测试：真实调用模型，验证工具闭环 + 压缩。

需要可用的 API key（项目 .env 或 ~/.hermes/.env）。
不纳入默认的 `uv run pytest`；请单独运行：

    uv run python examples/smoke_test.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

from mini_agent.config import load_config
from mini_agent.loop import Agent
from mini_agent.tools import build_tools

SANDBOX = Path(__file__).resolve().parents[1] / "sandbox"


def build_agent():
    cfg = load_config()
    cfg.validate()
    cfg.workdir = SANDBOX
    SANDBOX.mkdir(parents=True, exist_ok=True)
    registry = build_tools(cfg)
    agent = Agent(cfg, registry, on_event=lambda e, p: print(f"[{e}] {str(p)[:200]}"))
    return cfg, agent, registry


def test_tool_loop():
    print("=" * 60)
    print("测试 1：工具闭环（Write → Edit → Read）")
    cfg, agent, registry = build_agent()
    shutil.rmtree(SANDBOX / "demo", ignore_errors=True)

    answer = agent.run(
        "请在 demo/hello.py 里写一个函数 greet(name)，返回 f'你好, {name}'，"
        "然后用 Edit 在 demo/hello.py 末尾追加一行 __main__ 调用示例。"
        "做完后用 Read 读一遍文件确认内容。"
    )
    print("\n最终回答:", answer)
    target = SANDBOX / "demo" / "hello.py"
    print("\n文件存在:", target.exists())
    if target.exists():
        print(target.read_text(encoding="utf-8"))
    return target.exists()


def test_compaction():
    print("=" * 60)
    print("测试 2：压缩（构造超长历史）")
    cfg, agent, registry = build_agent()
    # 把窗口调小，逼出压缩（要装下工具 schema 本身约 2.3K tokens）
    cfg.context_window = 8000
    cfg.compact_ratio = 0.5
    agent.ctx.reset()

    filler = "这是一段用于填满上下文的历史记录，内容无关紧要，只用来触发压缩。" * 12
    for i in range(14):
        agent.ctx.add({"role": "user", "content": f"第{i}条：{filler}"})
        agent.ctx.add({"role": "assistant", "content": f"收到第{i}条，继续。"})

    before = agent.ctx.size()
    print(f"压缩前: {before} tokens（阈值 {cfg.compact_threshold}）")
    report = agent.ctx.compact(agent.llm, registry.schemas())
    print(f"压缩报告: {report}")
    # 单次摘要可能仍超窗口（摘要本身占空间），主循环是每步重试压缩，
    # 测试对齐该行为：最多再压两次
    for _ in range(2):
        if agent.ctx.size(registry.schemas()) <= cfg.context_window:
            break
        report = agent.ctx.compact(agent.llm, registry.schemas(), force=True)
        print(f"二次压缩: {agent.ctx.size()} tokens")
    print(f"压缩后: {agent.ctx.size()} tokens，消息数 {len(agent.ctx.messages)}")
    problems = agent.ctx.validate(registry.schemas())
    print("校验问题:", problems or "无")

    # 压缩后还能继续对话
    answer = agent.run("刚才我们聊到第几条了？只回答编号。")
    print("压缩后对话:", answer)
    return report is not None and not problems


if __name__ == "__main__":
    ok1 = test_tool_loop()
    ok2 = test_compaction()
    print("=" * 60)
    print(f"工具闭环={'PASS' if ok1 else 'FAIL'}  压缩={'PASS' if ok2 else 'FAIL'}")
