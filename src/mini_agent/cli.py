"""命令行 REPL。"""
from __future__ import annotations

import argparse
import sys

from .config import load_config, SYSTEM_PROMPT
from .context import Context
from .client import LLM
from .loop import Agent
from .store import SessionStore
from .tools import ToolRegistry, build_tools
from .tools.shell import resolve_shell_name


COLORS = {
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "reset": "\033[0m",
}


def _c(text: str, color: str, enabled: bool = True) -> str:
    if not enabled:
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def _force_utf8_stdio() -> None:
    """标准流统一 UTF-8：REPL 打印和管道输出都不再依赖 locale 码页（cp936）。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or "").replace("-", "").lower() != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def make_registry(cfg, spill_dir=None) -> ToolRegistry:
    # 落盘目录跟着会话库走：会话库已经做过可写回退，这里就不用再踩一次主目录
    return build_tools(cfg, spill_dir=spill_dir)


class StreamPrinter:
    """流式输出器：思维链灰色实时吐字，正文正常色实时吐字。"""

    def __init__(self, color: bool = True, show_reasoning: bool = True):
        self.color = color
        self.show_reasoning = show_reasoning
        self.mode: str | None = None
        self.printed_content = False

    def _flush(self) -> None:
        if self.mode is not None:
            if self.mode == "progress":
                # 擦除进度行，避免残留刷屏
                print("\r" + " " * 44 + "\r", end="")
            else:
                print()
            self.mode = None

    def delta(self, kind: str, text: str) -> None:
        if kind == "reasoning":
            if not self.show_reasoning:
                return
            if self.mode != "reasoning":
                self._flush()
                self.mode = "reasoning"
            print(_c(text, "dim", self.color), end="", flush=True)
        elif kind == "progress":
            # 工具参数生成心跳（大文件 Write/Edit 的 JSON 有几千 token，
            # 流式期间终端死寂）：同行刷新，不换行。text = "工具名 字符数"
            if self.mode != "progress":
                self._flush()
                self.mode = "progress"
            name, _, count = text.rpartition(" ")
            label = f" {name} 的参数" if name else "工具参数"
            print(_c(f"\r  …正在生成{label} {count} 字符", "yellow", self.color),
                  end="", flush=True)
        else:
            if self.mode != "content":
                self._flush()
                self.mode = "content"
            self.printed_content = True
            print(text, end="", flush=True)

    def event(self, event: str, payload: dict) -> None:
        if event == "assistant":
            self._flush()
            return
        self._flush()
        if event == "tool_call":
            print(_c(f"  → {payload['name']} {payload['arguments'][:200]}", "cyan", self.color))
        elif event == "tool_result":
            preview = str(payload["result"]).replace("\n", " ")[:160]
            print(_c(f"    ← {preview}", "dim", self.color))
        elif event == "notice":
            # 重试 / 溢出恢复等提示（区别于思维的灰、进度的黄，用暗青）
            print(_c(f"  [{payload['message']}]", "cyan", self.color))
        elif event == "compaction":
            print(
                _c(
                    f"  [压缩] {payload['before_tokens']} → {payload['after_tokens']} tokens，"
                    f"丢弃 {payload['dropped_messages']} 条，摘要={payload['summarised']}",
                    "yellow",
                    self.color,
                )
            )
        elif event == "error":
            print(_c(f"  [错误] {payload['message']}", "red", self.color))


def main() -> None:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="mini-agent", description="迷你 agent：工具调用 + 上下文压缩 + SQLite 会话"
    )
    parser.add_argument("--resume", "-r", metavar="SESSION_ID",
                        help="按会话 ID 恢复（latest = 最近一个）")
    parser.add_argument("--continue", "-c", dest="cont", action="store_true",
                        help="继续最近一次会话")
    parser.add_argument("--list", "-l", action="store_true", help="列出已有会话")
    args = parser.parse_args()

    try:
        cfg = load_config()
        cfg.validate()
    except SystemExit as exc:
        print(exc)
        raise SystemExit(1)

    color = sys.stdout.isatty()
    store = SessionStore()

    if args.list:
        sessions = store.list_sessions()
        if not sessions:
            print("(没有会话)")
        for row in sessions:
            print(f"{row['id']}  {row['msg_count']:>4} 条  {row['updated_at']}  {row['title'][:40]}")
        store.close()
        return

    session_id: str | None = None
    restored = 0
    if args.resume:
        target = store.latest() if args.resume == "latest" else args.resume
        if not target or not store.exists(target):
            print(f"ERROR: 会话不存在 {args.resume!r}，用 --list 查看已有会话")
            raise SystemExit(1)
        session_id = target
        restored = store.message_count(session_id)
    elif args.cont:
        session_id = store.latest()
        if session_id:
            restored = store.message_count(session_id)

    if session_id is None:
        session_id = store.create(model=cfg.model, workdir=str(cfg.workdir))

    try:
        registry = make_registry(cfg, store.path.parent / "spill")
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    printer = StreamPrinter(color, cfg.show_reasoning)
    agent = Agent(
        cfg,
        registry,
        on_event=printer.event,
        on_delta=printer.delta,
        store=store,
        session_id=session_id,
    )

    print(_c(f"mini-agent · {cfg.model}", "green", color))
    print(_c(f"会话 ID: {session_id}" + (f"（恢复 {restored} 条消息）" if restored else "（新建）"), "cyan", color))
    print(_c(f"工作目录: {cfg.workdir}", "dim", color))
    print(_c(f"Shell: {resolve_shell_name(cfg.shell)}", "dim", color))
    print(_c(f"窗口: {cfg.context_window} tokens，压缩阈值 {cfg.compact_threshold}", "dim", color))
    print(_c("命令: /help /clear /compact /tokens /log /session /quit", "dim", color))
    print()

    while True:
        try:
            line = input(_c("你 › ", "green", color)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in {"/quit", "/exit", "/q"}:
            break
        if line == "/help":
            print("/clear 清空历史  /compact 立即压缩  /tokens 查看用量"
                  "  /log 压缩记录  /session 显示会话 ID  /quit 退出")
            continue
        if line == "/session":
            print(f"{session_id}（{store.message_count(session_id)} 条消息）")
            continue
        if line == "/clear":
            agent.ctx.reset()
            store.replace(session_id, agent.ctx.messages)
            print("历史已清空（数据库已同步）")
            continue
        if line == "/tokens":
            size = agent.ctx.size(registry.schemas())
            print(f"{size} tokens（窗口 {cfg.context_window}）")
            continue
        if line == "/log":
            if not agent.ctx.log:
                print("尚未压缩过")
            for i, r in enumerate(agent.ctx.log, 1):
                print(
                    f"{i}. {r.before_tokens} → {r.after_tokens}，丢弃 {r.dropped_messages} 条，"
                    f"摘要={r.summarised}，尝试={r.attempts}"
                )
            continue
        if line == "/compact":
            report = agent.ctx.compact(agent.llm, registry.schemas(), force=True)
            if report:
                store.replace(session_id, agent.ctx.messages)
                print(f"已压缩: {report.before_tokens} → {report.after_tokens}（已写入会话）")
            else:
                print("无需压缩（历史太短）")
            continue

        if store.message_count(session_id) == 0:
            store.set_title(session_id, line[:40])

        printer.printed_content = False
        try:
            answer = agent.run(line)
        except KeyboardInterrupt:
            print("\n（已中断本轮）")
            continue
        if answer and not printer.printed_content:
            print()
            print(answer)
        print()

    # Claude Code 同款退出提示：告诉用户怎么恢复本次会话
    print()
    print(_c("Resume this session with:", "dim", color))
    print(_c(f"mini-agent --resume {session_id}", "cyan", color))
    store.close()


if __name__ == "__main__":
    main()
