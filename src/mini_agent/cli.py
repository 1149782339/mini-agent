"""命令行 REPL。"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from pathlib import Path

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    _HAS_RICH = True
except ImportError:  # rich 是硬依赖，但保持降级 gracefully
    _HAS_RICH = False

from .config import load_config, SYSTEM_PROMPT
from .context import Context
from .client import LLM
from .loop import Agent
from .store import SessionStore, TaskStore
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


class SubagentPanels:
    """rich.Live 底部状态区：每个活跃子代理一块，原地刷新各自的最新输出。

    并行子代理的进度各自一块、原地刷新，避免交错刷屏。
    REPL 仍是行打印主体，只在底部开一个 transient 状态区：
    子代理全部结束后面板自动消失，最终 tool_result 行照常打印在上方。
    非 TTY（管道/重定向）自动禁用。
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled and _HAS_RICH and sys.stdout.isatty()
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._live: Live | None = None

    def update(self, payload: dict) -> None:
        with self._lock:
            self._tasks[payload["tag"]] = payload
            if not self._enabled:
                return
            if payload.get("status") == "running" and self._live is None:
                self._live = Live(
                    self._render(), console=Console(),
                    transient=True, refresh_per_second=8,
                )
                self._live.start()
            if self._live is not None:
                self._live.update(self._render())
            if self._live is not None and all(
                t.get("status") == "done" for t in self._tasks.values()
            ):
                self._live.stop()
                self._live = None

    def stop_all(self, reason: str = "中断") -> None:
        """REPL 中断/退出时收尾：运行中的标记为中断并关闭面板。"""
        with self._lock:
            for t in self._tasks.values():
                if t.get("status") == "running":
                    t["status"] = "done"
                    t["snippet"] = f"[{reason}]"
            if self._live is not None:
                self._live.stop()
                self._live = None

    def _render(self) -> Panel:
        rows = []
        for tag, t in self._tasks.items():
            if t.get("status") == "done":
                status, style = "完成", "green"
            else:
                status, style = "运行", "yellow"
            tokens = t.get("tokens") or 0
            tok = f"{tokens / 1000:.1f}K tok" if tokens else "- tok"
            snippet = t.get("snippet") or ""
            rows.append(f"[{style}]{status}[/{style}] {tag}  "
                        f"第{t.get('step', 0)}步 · {tok}  [dim]{snippet}[/dim]")
        return Panel(
            "\n".join(rows) if rows else "[dim]（无）[/dim]",
            title="子代理", border_style="dim", padding=(0, 1),
        )


def _force_utf8_stdio() -> None:
    """标准流统一 UTF-8：REPL 打印和管道输出都不再依赖 locale 码页（cp936）。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or "").replace("-", "").lower() != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def make_registry(cfg, spill_dir=None, task_store=None, session_id=None) -> ToolRegistry:
    # 落盘目录跟着会话库走：会话库已经做过可写回退，这里就不用再踩一次主目录
    return build_tools(
        cfg, spill_dir=spill_dir, task_store=task_store, session_id=session_id
    )


def run_print_mode(args, cfg, llm=None) -> int:
    """print 模式：跑单轮任务后退出（子代理的宿主形态，非交互）。

    stdout 在 --json 下**边跑边输出** JSONL 事件（message_end/result/error），
    供父进程（SubagentTool）解析——message_end 必须随步骤实时发出，
    攒到最后才吐等于让父进程的面板全程黑屏。
    llm 参数仅供测试注入假实现。
    """
    if args.model:
        cfg.model = args.model
    if getattr(args, "max_steps", None):
        cfg.max_steps = args.max_steps
    system_prompt = SYSTEM_PROMPT
    if args.append_system_prompt:
        try:
            extra = Path(args.append_system_prompt).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: 读取 system prompt 文件失败：{exc}")
            return 2
        if extra.strip():
            system_prompt = f"{system_prompt}\n\n{extra.strip()}"

    registry = build_tools(cfg)
    jobs = getattr(registry, "jobs", None)
    if jobs is not None:
        jobs.allow_notify = False
        jobs.allow_background_subagents = False
    # 防递归：print 模式永不提供 Task，子代理不能再派子代理
    registry.tools.pop("Task", None)
    if args.tools:
        wanted = {s.strip() for s in args.tools.split(",") if s.strip()}
        wanted.discard("Task")
        registry.tools = {k: v for k, v in registry.tools.items() if k in wanted}
        if not registry.tools:
            print("ERROR: --tools 白名单没有匹配到任何工具")
            return 2

    # -p / 脚本 / Task 子代理：默认开硬停（没人盯着）
    agent = Agent(cfg, registry, llm=llm, hard_stop_enabled=True)
    agent.ctx.messages = [{"role": "system", "content": system_prompt}]

    def _emit(event: dict) -> None:
        if args.json:
            print(json.dumps(event, ensure_ascii=False), flush=True)
        elif event.get("type") == "result":
            print(event.get("result") or "", flush=True)

    def _forward(event: str, payload: dict) -> None:
        # 每步模型响应落库后立即转发 message_end（payload 里无 usage，
        # 从刚写入的 ctx.messages 末位取——assistant 事件先于工具派发触发）
        if event != "assistant":
            return
        msg = agent.ctx.messages[-1] if agent.ctx.messages else {}
        if msg.get("role") != "assistant":
            return
        _emit({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": msg.get("content") or ""}],
            },
            "usage": msg.get("_usage"),
            "stopReason": msg.get("_finish_reason"),
        })

    # 长思维链期间的心跳：把流式 delta 节流成 thinking 事件，父面板显示
    # "思考中 N 字…"——否则首轮长思考几十秒，子代理看起来又是死的
    think_state = {"chars": 0, "last_emit": 0}

    def _delta(kind: str, text: str) -> None:
        if not args.json or not text:
            return
        think_state["chars"] += len(text)
        if think_state["chars"] - think_state["last_emit"] >= 500:
            think_state["last_emit"] = think_state["chars"]
            verb = "思考中" if kind == "reasoning" else "撰写中"
            _emit({"type": "thinking", "kind": kind,
                   "chars": think_state["chars"], "verb": verb})

    if not args.json:
        _force_utf8_stdio()
    agent.on_delta = _delta
    try:
        answer = agent.run(args.print_prompt, _event_override=_forward)

        if answer.startswith("ERROR:"):
            _emit({"type": "error", "message": answer})
            return 1
        _emit({"type": "result", "result": answer, "steps": None})
        return 0
    finally:
        if jobs is not None:
            jobs.cleanup_all()


class StreamPrinter:
    """流式输出器：思维链灰色实时吐字，正文正常色实时吐字。"""

    def __init__(self, color: bool = True, show_reasoning: bool = True):
        self.color = color
        self.show_reasoning = show_reasoning
        self.mode: str | None = None
        self.printed_content = False
        self.panels = SubagentPanels(enabled=color)

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
        elif event == "subagent_progress":
            # 子代理实时进度：进底部面板，原地刷新，不滚屏
            self.panels.update(payload)
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


class InteractiveSession:
    """交互式 REPL：用户输入和后台完成通知抢同一把调度锁。

    输入在独立线程读取（Windows 控制台不能靠 POSIX select 盯 stdin）。
    一轮进行中到达的完成通知或下一条用户输入会排队，结束后再处理。
    Shell 完成通知只汇报并记录；后台 Subagent 在空闲时会唤醒 Agent.run。
    忙碌期间由 Agent 在工具结果补齐后的下一 LLM step 前注入，重复事件会丢弃。
    退出或 Ctrl+C 离开 REPL 时清掉仍在跑的后台进程。
    """

    def __init__(self, agent, store, session_id: str, registry, printer, color: bool, jobs):
        self.agent = agent
        self.store = store
        self.session_id = session_id
        self.registry = registry
        self.printer = printer
        self.color = color
        self.jobs = jobs
        self.events: queue.Queue = queue.Queue()
        self._stop = threading.Event()

    def submit(self, kind: str, payload) -> None:
        self.events.put((kind, payload))

    def _on_task_complete(self, completion) -> None:
        self.submit("task_done", completion)

    def _input_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = input(_c("你 › ", "green", self.color)).strip()
            except EOFError:
                self.submit("eof", None)
                return
            except KeyboardInterrupt:
                self.submit("eof", None)
                return
            self.submit("user", line)

    def run(self) -> None:
        if self.jobs is not None:
            self.jobs.on_complete(self._on_task_complete)
        reader = threading.Thread(target=self._input_loop, name="repl-input", daemon=True)
        reader.start()
        try:
            self._dispatch_loop()
        finally:
            self._stop.set()
            if self.jobs is not None:
                self.jobs.cleanup_all()

    def _dispatch_loop(self) -> None:
        while True:
            kind, payload = self.events.get()
            if kind == "eof":
                return
            if kind == "user":
                if self._handle_user(payload) == "quit":
                    return
            elif kind == "task_done":
                self._handle_completion(payload)

    def _handle_user(self, line: str) -> str | None:
        if not line:
            return None
        if line in {"/quit", "/exit", "/q"}:
            return "quit"
        if line == "/help":
            print("/clear 清空历史  /compact 立即压缩  /tokens 查看用量"
                  "  /log 压缩记录  /session 显示会话 ID  /quit 退出")
            return None
        if line == "/session":
            view_n = self.store.view_message_count(self.session_id, system_prompt=SYSTEM_PROMPT)
            raw_n = self.store.message_count(self.session_id)
            if raw_n > view_n:
                print(f"{self.session_id}（当前视图 {view_n} 条，库中原文 {raw_n} 条）")
            else:
                print(f"{self.session_id}（{view_n} 条消息）")
            return None
        if line == "/clear":
            self.agent.ctx.reset()
            self.store.mark_clear(self.session_id)
            print("历史已清空（原文仍保留在会话库中，当前视图已重置）")
            return None
        if line == "/tokens":
            size = self.agent.ctx.size(self.registry.schemas())
            print(f"{size} tokens（窗口 {self.agent.cfg.context_window}）")
            return None
        if line == "/log":
            if not self.agent.ctx.log:
                print("尚未压缩过")
            for i, r in enumerate(self.agent.ctx.log, 1):
                print(
                    f"{i}. {r.before_tokens} → {r.after_tokens}，丢弃 {r.dropped_messages} 条，"
                    f"摘要={r.summarised}，尝试={r.attempts}"
                )
            return None
        if line == "/compact":
            report = self.agent.compact_now(force=True)
            if report:
                print(f"已压缩: {report.before_tokens} → {report.after_tokens}（已记录压缩，原文保留）")
            else:
                print("无需压缩（历史太短）")
            return None

        if self.store is not None and self.store.current_user_message_count(self.session_id, system_prompt=SYSTEM_PROMPT) == 0:
            self.store.set_title(self.session_id, line[:40])
        self._run_agent(line)
        return None

    def _handle_completion(self, completion) -> None:
        # 忙碌注入、显式 task_manage 或这里的空闲处理，只有一个能消费。
        if self.jobs is not None:
            completion = self.jobs.consume_notification(completion)
            if completion is None:
                return
        print()
        print(_c(f"后台任务 {completion.task_id} 已结束：", "yellow", self.color))
        print(completion.message)
        notice = completion.message or ""
        if completion.delivery == "wake":
            print(_c("结果已送回 Agent，继续处理当前任务。", "yellow", self.color))
            print()
            if self.agent is not None:
                self._run_agent(notice)
            return
        print(_c("结果已记录，下一步等你说了再做。", "yellow", self.color))
        print()
        if "不要自动开始下一步" not in notice:
            notice += "\n\n（以上是后台任务结果，已向用户汇报。请等待用户指示，不要自动开始下一步。）"
        if self.agent is not None:
            self.agent.record_event(notice)

    def _run_agent(self, text: str) -> None:
        self.printer.printed_content = False
        try:
            answer = self.agent.run(text)
        except KeyboardInterrupt:
            self.printer.panels.stop_all()
            print("\n（已中断本轮）")
            return
        if answer and not self.printer.printed_content:
            print()
            print(answer)
        print()


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
    parser.add_argument("-p", "--print", dest="print_prompt", metavar="PROMPT",
                        help="print 模式：跑单轮任务后退出（非交互，子代理的宿主形态）")
    parser.add_argument("--json", action="store_true",
                        help="print 模式下 stdout 按行输出 JSON 事件（message_end/result/error）")
    parser.add_argument("--no-session", action="store_true", help="不写会话文件")
    parser.add_argument("--append-system-prompt", metavar="FILE",
                        help="把文件内容追加到 system prompt 后")
    parser.add_argument("--model", metavar="MODEL", help="覆盖模型名")
    parser.add_argument("--config", metavar="FILE",
                        help="指定 config.toml（默认 ~/.config/mini-agent/config.toml）")
    parser.add_argument("--tools", metavar="列表", help="逗号分隔的工具白名单")
    parser.add_argument("--max-steps", type=int, metavar="N",
                        help="覆盖单轮最大工具步数（深调研任务可调大）")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        cfg.validate()
    except SystemExit as exc:
        print(exc)
        raise SystemExit(1)

    if args.model:
        cfg.model = args.model
    if args.max_steps:
        cfg.max_steps = args.max_steps

    if args.print_prompt is not None:
        if not args.print_prompt.strip():
            print("ERROR: -p 的任务文本不能为空")
            raise SystemExit(2)
        raise SystemExit(run_print_mode(args, cfg))

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
        restored = store.view_message_count(session_id, system_prompt=SYSTEM_PROMPT)
    elif args.cont:
        session_id = store.latest()
        if session_id:
            restored = store.view_message_count(session_id, system_prompt=SYSTEM_PROMPT)

    if session_id is None:
        session_id = store.create(model=cfg.model, workdir=str(cfg.workdir))

    try:
        registry = make_registry(
            cfg,
            store.path.parent / "spill",
            task_store=TaskStore(store.path),
            session_id=session_id,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    printer = StreamPrinter(color, cfg.show_reasoning)
    # 交互式 CLI：默认只警告、不硬停，避免误伤正在盯着的调试
    agent = Agent(
        cfg,
        registry,
        on_event=printer.event,
        on_delta=printer.delta,
        store=store,
        session_id=session_id,
        hard_stop_enabled=False,
    )

    print(_c(f"mini-agent · {cfg.model}", "green", color))
    print(_c(f"会话 ID: {session_id}" + (f"（恢复 {restored} 条消息）" if restored else "（新建）"), "cyan", color))
    print(_c(f"工作目录: {cfg.workdir}", "dim", color))
    print(_c(f"Shell: {resolve_shell_name(cfg.shell)}", "dim", color))
    print(_c(f"窗口: {cfg.context_window} tokens，压缩阈值 {cfg.compact_threshold}", "dim", color))
    print(_c("命令: /help /clear /compact /tokens /log /session /quit", "dim", color))
    print()

    jobs = getattr(registry, "jobs", None)
    session = InteractiveSession(
        agent, store, session_id, registry, printer, color, jobs
    )
    try:
        session.run()
    finally:
        # 告诉用户怎么恢复本次会话
        printer.panels.stop_all()
        print()
        print(_c("Resume this session with:", "dim", color))
        print(_c(f"mini-agent --resume {session_id}", "cyan", color))
        store.close()


if __name__ == "__main__":
    main()
