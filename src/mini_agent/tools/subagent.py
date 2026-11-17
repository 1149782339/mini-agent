"""Task 工具：子代理委派。子代理是独立进程，内核不改。

- 用 sys.executable -m mini_agent 拉起宿主自己，print 模式跑单轮任务
- 子代理零继承父上下文：交接物只有"子代理 system prompt（临时文件）+
  任务文本（argv 末位）"，想要多少上下文由父模型写进任务文本
- 父子通信：子进程 stdout 每行一个 JSON 事件（message_end / result / error），
  父进程取最后一条 assistant 文本作为工具结果

防递归：print 模式下永不注册 Task（cli.run_print_mode 里强制摘除），
子代理不能再派子代理。

步骤级进度：子进程 stdout 的 message_end / thinking 事件上报到父终端面板
（步数、token、一行摘要）。还没有 agents/*.md 自定义定义、parallel/chain，
中止靠 Ctrl+C 杀掉子进程。
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolError, ToolOutput, ValidationResult, semantic_int
from ..envutil import sanitize_subprocess_env
from .shell import WindowsJob, kill_process_tree

# 子代理的 system prompt：明确身份、工作方式和返回格式
CHILD_SYSTEM_PROMPT = """\
你是一个被主代理派发的子代理，在用户的开发机上独立执行一个任务。

工作方式：
- 用 Read/Ls/Glob/Grep 了解现状，用 Edit/Write 修改文件，删除用 RM，移动或重命名用 MV，用 run_command 验证。不要在 shell 里直接执行 rm 或 mv。
- 做完必须实际验证（跑测试 / 运行 / 语法检查），不要声称没做过的验证。
- 不擅长的细节宁可再读文件确认，不要臆造路径和内容。

返回格式（最后一条消息，即你的返回值）：
- 第一段：任务完成情况的结论（做了什么、验证了什么、结果如何）。
- 随后按需列出：改动的文件路径清单、关键发现、遗留问题。
- 不要寒暄，不要复述任务，内容控制在 500 字内，细节留给文件。\
"""

# 子代理默认工具白名单（Task 已在 print 模式被摘除，双重保险不列它）
CHILD_TOOLS = "Read,Edit,Write,RM,MV,Glob,Grep,Ls,run_command"

MAX_RESULT_CHARS = 50000    # 单任务结果字符上限
DEFAULT_TIMEOUT = 900       # 子代理最长运行秒数
CHILD_MAX_STEPS = 150       # 子代理步数预算：深调研（读十几个文件）40 步不够，
                            # 进程是独立 load_config()，经环境变量覆盖给它

# 子进程 stdout 里出现这个 = 子代理被步数上限截断，结果不完整
_STEP_LIMIT_NOTICE = "达到最大工具步数"

# 活跃子进程登记：主线程 KeyboardInterrupt 时统一 kill，不留孤儿。
# 值是 (进程, Job)。Windows 上 Job 能带走子代理再拉起的 run_command。
_ACTIVE_PROCS: dict[int, tuple[subprocess.Popen, object | None]] = {}
_TAG_COUNTER = itertools.count(1)


class ProcessTimeout(Exception):
    """子进程在截止时间前没有结束。"""


def _kill_proc(proc: subprocess.Popen, job: object | None) -> None:
    """杀掉进程以及它拉起的子孙。优先 Job Object，否则杀进程树。"""
    try:
        if job is not None and getattr(job, "active", False):
            job.kill()  # type: ignore[attr-defined]
        else:
            kill_process_tree(proc)
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=3)
    except Exception:
        pass


def terminate_active_children() -> None:
    """杀掉所有在跑的子代理进程（loop 的 KeyboardInterrupt 路径调用）。"""
    for proc, job in list(_ACTIVE_PROCS.values()):
        _kill_proc(proc, job)


def drain_lines(stream, timeout: float, on_line) -> bool:
    """在 timeout 秒内把流读到 EOF。

    返回 True 表示读完。返回 False 表示时间到了流还没结束——调用方负责杀进程。
    读在后台线程里进行，所以超时不依赖子进程自己关闭 stdout。
    """
    if stream is None:
        return True
    done = threading.Event()

    def _reader() -> None:
        try:
            for line in iter(stream.readline, ""):
                on_line(line)
        finally:
            done.set()

    threading.Thread(target=_reader, daemon=True).start()
    return done.wait(timeout)


def collect_stdout(proc: subprocess.Popen, timeout: float, on_line, job: object | None = None) -> list[str]:
    """读完子进程 stdout。超过 timeout 秒就杀掉整棵进程树并抛 ProcessTimeout。

    进程在时限内自己结束时，即使墙钟已经接近 timeout，也保留已经读到的输出。
    """
    lines: list[str] = []
    deadline = time.monotonic() + timeout

    def _on(line: str) -> None:
        lines.append(line)
        on_line(line)

    try:
        remaining = max(0.0, deadline - time.monotonic())
        finished = drain_lines(proc.stdout, remaining, _on)
        if not finished:
            _kill_proc(proc, job)
            raise ProcessTimeout()
        if proc.poll() is None:
            remaining = deadline - time.monotonic()
            try:
                proc.wait(timeout=max(0.1, remaining))
            except subprocess.TimeoutExpired:
                _kill_proc(proc, job)
                raise ProcessTimeout() from None
    except KeyboardInterrupt:
        _kill_proc(proc, job)
        raise
    return lines


# ---------- 纯函数（可单测） ----------

def build_child_argv(task: str, prompt_file: str, model: str = "",
                     tools: str = CHILD_TOOLS) -> list[str]:
    """构造子进程 argv。"""
    argv = [sys.executable, "-m", "mini_agent",
            "-p", task, "--json", "--no-session",
            "--append-system-prompt", prompt_file]
    if tools.strip():
        argv += ["--tools", tools.strip()]
    if model.strip():
        argv += ["--model", model.strip()]
    return argv


def build_child_env(
    base: dict[str, str] | None = None,
    max_steps: int = CHILD_MAX_STEPS,
) -> dict[str, str]:
    """构造子代理进程的环境。

    剥离父进程里的 API key 等密钥。子进程启动时由 load_config() 自行从
    项目 .env 或 ~/.hermes/.env 读取密钥，不靠继承父进程环境。
    MINI_AGENT_MAX_STEPS 仍由父进程显式写入，覆盖子代理步数预算。
    """
    env = sanitize_subprocess_env(base)
    env["MINI_AGENT_MAX_STEPS"] = str(max_steps)
    return env


def parse_json_events(stdout_text: str) -> dict[str, Any]:
    """解析子进程 stdout 的 JSONL 事件流。

    取最后一条 message_end 的 assistant 文本作为最终输出。
    返回 {final, assistant_count, last_stop, error, result}。
    """
    final = ""
    assistant_count = 0
    last_stop: str | None = None
    error: str | None = None
    result: str | None = None
    for line in stdout_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # 非 JSON 行（如库警告）跳过
        etype = event.get("type")
        if etype == "message_end":
            msg = event.get("message") or {}
            if msg.get("role") == "assistant":
                assistant_count += 1
                last_stop = event.get("stopReason")
                for part in msg.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        final = part["text"]  # 后到的覆盖：取最后一条 assistant 的首个 text
                        break
        elif etype == "result":
            result = str(event.get("result") or "")
        elif etype == "error":
            error = str(event.get("message") or "")
    return {
        "final": result if result is not None else final,
        "assistant_count": assistant_count,
        "last_stop": last_stop,
        "error": error,
    }


def describe_event(event: dict[str, Any]) -> tuple[bool, int, str]:
    """从 message_end 事件提取进度三元组：(是否assistant步, token总量, 单行摘要)。

    摘要去换行截 40 字符，作为"子代理此刻在干嘛"的展示。
    """
    msg = event.get("message") or {}
    if event.get("type") != "message_end" or msg.get("role") != "assistant":
        return False, 0, ""
    usage = event.get("usage") or {}
    tokens = int(usage.get("total") or 0)
    snippet = ""
    for part in msg.get("content") or []:
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            snippet = " ".join(str(part["text"]).split())[:40]
            break
    return True, tokens, snippet


def truncate_result(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    """保头，绝不在行中间切断。"""
    if len(text) <= limit:
        return text
    head = text[:limit]
    if "\n" in head:
        head = head[: head.rfind("\n")]
    return head + f"\n[子代理输出超过 {limit} 字符已截断]"


# ---------- 工具本体 ----------

class SubagentTool(Tool):
    name = "Task"
    description = (
        "派发一个子代理独立执行多步骤任务（开放式搜索、代码审查、独立实现等）。\n"
        "子代理是独立进程：看不到本会话的任何对话内容，请在 prompt 里写清它需要的"
        "全部上下文（目标、相关路径、约束、已知结论）。它会自己读写文件、运行命令、"
        "验证结果，最后把结论作为本工具的返回值。\n"
        "- 适合：可以独立闭环、不需要你逐步指挥的任务；多个互不依赖的任务可以在"
        "一次响应里发多个 Task 调用，它们会并行执行。\n"
        "- 不适合：一两步就能做完的事（直接自己做更快）。\n"
        "- 任务要自包含：子代理无法向你追问；步数预算约 150 步，过大的调研请"
        "按子系统/目录拆开派发。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "完整、自包含的任务描述（含所需上下文）",
            },
            "timeout": {
                "type": "integer",
                "description": f"最长运行秒数，默认 {DEFAULT_TIMEOUT}，最大 3600",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    def is_concurrency_safe(self) -> bool:
        return True  # 子代理是独立进程天然隔离；并行派发是它的核心用法

    def validate_input(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return ValidationResult.fail("prompt 必须是非空字符串", 1)
        return ValidationResult.pass_()

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        task = args["prompt"].strip()
        timeout = semantic_int(args.get("timeout"), DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT
        timeout = max(60, min(int(timeout), 3600))
        tag = f"子代理#{next(_TAG_COUNTER)}"
        hook = getattr(ctx, "event_hook", None)  # 可能为 None（测试/无注入场景）

        def _report(step: int, tokens: int, snippet: str, status: str,
                    exit_code: int | None = None) -> None:
            if hook is not None:
                hook("subagent_progress", {
                    "tag": tag, "step": step, "tokens": tokens,
                    "snippet": snippet, "status": status, "exit": exit_code,
                })

        debug = os.environ.get("MINI_AGENT_DEBUG", "").strip() in {"1", "true"}
        _t0 = time.monotonic()

        def _dbg(marker: str) -> None:
            if debug:
                sys.stderr.write(
                    f"[subagent-debug] {tag} {marker} @{time.time():.3f} "
                    f"(+{time.monotonic() - _t0:.2f}s)\n"
                )

        # system prompt 走临时文件，经 --append-system-prompt 传给子进程
        tmp = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".md", prefix="mini_agent_sub_", delete=False
        )
        try:
            tmp.write(CHILD_SYSTEM_PROMPT)
            tmp.close()
            argv = build_child_argv(task, tmp.name)
            # 子进程是独立 load_config()：步数预算经环境变量给它（深调研默认 150 步）。
            # 密钥不从父进程环境继承，由子进程自己读项目 .env / ~/.hermes/.env。
            child_env = build_child_env(os.environ, CHILD_MAX_STEPS)
            _dbg("tmpfile-ready")
            popen_kwargs: dict[str, Any] = dict(
                cwd=str(ctx.workdir),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,            # 流式逐行读需要文本模式
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                # 独立进程组：POSIX 上 killpg 不会误伤父代理
                start_new_session=True,
            )
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            try:
                proc = subprocess.Popen(argv, **popen_kwargs)
            except OSError as exc:
                raise ToolError(f"无法启动子代理进程：{exc}", 20) from None
            _dbg("popen-done")

            job = WindowsJob() if WindowsJob is not None else None
            if job is not None:
                job.attach(proc)
            _ACTIVE_PROCS[id(proc)] = (proc, job)
            _dbg("pre-drain")
            # stderr 并发排空：PIPE 容量有限（Windows 64KB），子进程 stderr
            # 写满了没进程读会整体冻死。边跑边抽干，结束时汇总
            stderr_chunks: list[str] = []
            if proc.stderr is not None:
                def _drain(stream=proc.stderr, sink=stderr_chunks):
                    for chunk in iter(stream.readline, ""):
                        sink.append(chunk)
                threading.Thread(target=_drain, daemon=True).start()

            _report(0, 0, "子进程已启动", "running")  # 面板立刻出现，不等首个响应

            step = 0
            tokens = 0

            def _on_line(line: str) -> None:
                nonlocal step, tokens
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    return
                if event.get("type") == "thinking":
                    chars = int(event.get("chars") or 0)
                    _report(step, tokens, f"思考中 {chars} 字…", "running")
                    return
                is_step, tok, snippet = describe_event(event)
                if is_step:
                    step += 1
                    tokens = tok or tokens
                    _report(step, tokens, snippet, "running")

            try:
                # 读 stdout 有截止时间：子进程不退出也会被杀掉，
                # 而不是等它自己关闭管道之后才发现已经超时。
                try:
                    lines = collect_stdout(proc, timeout, _on_line, job)
                except ProcessTimeout:
                    raise ToolError(
                        f"子代理运行超过 {timeout}s 被杀。任务太大请拆小后重试。", 21
                    ) from None
                _report(step, tokens, "", "done", proc.returncode)
            finally:
                _ACTIVE_PROCS.pop(id(proc), None)
                if job is not None:
                    job.close()
            err = "".join(stderr_chunks)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        out = "".join(lines)
        parsed = parse_json_events(out)

        if parsed["error"] and not parsed["final"]:
            raise ToolError(
                f"子代理失败：{parsed['error']}（stderr 尾部：{err.strip()[-300:]}）", 22
            )
        if proc.returncode != 0 and not parsed["final"]:
            raise ToolError(
                f"子代理退出码 {proc.returncode}（stderr 尾部：{err.strip()[-300:]}）", 23
            )

        final = truncate_result(parsed["final"] or "(子代理未返回文本结果)")
        truncated_steps = _STEP_LIMIT_NOTICE in final
        if truncated_steps:
            # 结果不完整要显式告诉父代理（它据此缩小范围重派，而不是当作完整结论）
            final += (
                f"\n\n[注意] 子代理被步数上限（{CHILD_MAX_STEPS} 步）截断，以上不完整；"
                "请把任务拆小或缩小范围后重新派发。"
            )
        meta = {"type": "subagent", "tag": tag, "steps": parsed["assistant_count"],
                "stop": parsed["last_stop"], "exit": proc.returncode,
                "truncated": truncated_steps}
        return ToolOutput(final, meta)
