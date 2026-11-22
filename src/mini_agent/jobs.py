"""进程内统一后台任务注册表。

Shell 与 Subagent 共用任务 ID、状态、输出、取消和完成通知。
本模块只做 process-local 生命周期；不持久化任务，也不跨重启恢复。
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from .store import project_data_dir
from .tools.base import ToolError, decode_output

# 与 tools.shell 保持一致；避免顶层互相 import。
MAX_BUFFER = 40000
HARD_TIMEOUT = 300
TASK_HARD_TIMEOUT = 3600
KILL_WAIT = 5

Action = Literal["list", "poll", "log", "wait", "kill"]
TaskAction = Literal["list", "get", "output", "wait", "cancel"]
Delivery = Literal["none", "record", "wake"]

# wait 默认别无限堵死这一轮工具调用
DEFAULT_WAIT = 30


def clip_output(text: str, tag: str = "job") -> str:
    """超长输出：完整内容落盘 spill/，上下文里只保留末尾。"""
    if len(text) <= MAX_BUFFER:
        return text
    try:
        spill_dir = project_data_dir() / "spill"
        spill_dir.mkdir(parents=True, exist_ok=True)
        path = spill_dir / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.txt"
        path.write_text(text, encoding="utf-8")
        note = (
            f"[完整输出 {len(text)} 字符已保存到 {path}，"
            f"以下为末尾 {MAX_BUFFER} 字符]\n"
        )
    except OSError:
        note = f"[输出共 {len(text)} 字符，以下为末尾 {MAX_BUFFER} 字符]\n"
    return note + text[-MAX_BUFFER:]


class TaskLimitError(RuntimeError):
    """同一 owner 的活跃任务已达到准入上限。"""


@dataclass
class JobCompletion:
    job_id: str
    command: str
    cwd: str
    exit_code: int | None
    elapsed: float
    stdout: str
    stderr: str
    message: str
    kind: str = "shell"
    delivery: Delivery = "record"
    notice_id: str = ""
    result: str = ""
    error: str = ""

    @property
    def task_id(self) -> str:
        return self.job_id


@dataclass
class Job:
    job_id: str
    command: str
    cwd: Path
    proc: object | None  # subprocess.Popen；纯 Python runner 可以为 None
    win_job: object | None
    started: float
    notify_on_complete: bool = False
    upgraded_from_timeout: bool = False
    kind: str = "shell"
    owner_session_id: str | None = None
    delivery: Delivery = "record"
    notice_id: str = ""
    notification_state: str = "none"  # none -> pending -> consumed
    result: str = ""
    error: str = ""
    error_code: int | None = None
    meta: dict = field(default_factory=dict)
    stdout_parts: list[str] = field(default_factory=list)
    stderr_parts: list[str] = field(default_factory=list)
    exit_code: int | None = None
    finished_at: float | None = None
    output_lock: threading.Lock = field(default_factory=threading.Lock)
    done: threading.Event = field(default_factory=threading.Event)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    last_viewed_len: int = -1
    stale_views: int = 0
    log_path: Path | None = None

    @property
    def task_id(self) -> str:
        return self.job_id

    def output_len(self) -> int:
        with self.output_lock:
            return sum(len(p) for p in self.stdout_parts) + sum(len(p) for p in self.stderr_parts)

    def snapshot(self) -> tuple[str, str]:
        with self.output_lock:
            return "".join(self.stdout_parts), "".join(self.stderr_parts)

    def append(self, kind: str, text: str) -> None:
        if not text:
            return
        with self.output_lock:
            if kind == "stderr":
                self.stderr_parts.append(text)
            else:
                self.stdout_parts.append(text)

    def running(self) -> bool:
        return not self.done.is_set()

    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started

    @property
    def status(self) -> str:
        if self.running():
            return "running"
        if self.cancel_event.is_set():
            return "cancelled"
        if self.error:
            return "failed"
        if self.kind == "shell":
            return f"exited={self.exit_code}"
        return "completed"


class TaskRegistry:
    def __init__(self, allow_notify: bool = True) -> None:
        self.allow_notify = allow_notify
        self.allow_background_subagents = True
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._listeners: list[Callable[[JobCompletion], None]] = []

    def on_complete(self, fn: Callable[[JobCompletion], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def running_jobs(self) -> list[Job]:
        return [j for j in self.all_jobs() if j.running()]

    def _next_id(self) -> str:
        self._seq += 1
        return f"job_{self._seq:02d}"

    def add_running(
        self,
        proc,
        command: str,
        cwd: Path,
        win_job: object | None = None,
        notify_on_complete: bool = False,
        upgraded_from_timeout: bool = False,
        initial_stdout: bytes | str | None = None,
        initial_stderr: bytes | str | None = None,
    ) -> Job:
        """登记一个已经启动、仍在跑的 Shell 进程，并开始收集输出。"""
        with self._lock:
            job_id = self._next_id()
            job = Job(
                job_id=job_id,
                command=command,
                cwd=cwd,
                proc=proc,
                win_job=win_job,
                started=time.monotonic(),
                notify_on_complete=bool(notify_on_complete) and self.allow_notify,
                upgraded_from_timeout=upgraded_from_timeout,
                kind="shell",
                delivery="record",
                notice_id=f"{job_id}:complete",
            )
            self._jobs[job_id] = job
        if initial_stdout:
            job.append("stdout", decode_output(initial_stdout))
        if initial_stderr:
            job.append("stderr", decode_output(initial_stderr))
        self._flush_log(job)
        threading.Thread(target=self._supervise, args=(job,), name=f"job-{job.job_id}", daemon=True).start()
        return job

    def add_task(
        self,
        runner: Callable[[Job], object],
        *,
        command: str,
        cwd: Path,
        kind: str = "subagent",
        owner_session_id: str | None = None,
        delivery: Delivery = "wake",
        max_active: int | None = None,
    ) -> Job:
        """原子完成准入与登记，再由后台线程执行 runner。

        登记本身就是活跃名额预留；runner 无论成功失败都会进入终态并释放名额。
        """
        with self._lock:
            if max_active is not None:
                active = sum(
                    j.running() and j.kind == kind and j.owner_session_id == owner_session_id
                    for j in self._jobs.values()
                )
                if active >= max_active:
                    raise TaskLimitError(
                        f"owner={owner_session_id or 'default'} 的活跃 {kind} 已达上限 "
                        f"{max_active}，请等待完成或先取消任务"
                    )
            job_id = self._next_id()
            notify = delivery != "none" and self.allow_notify
            job = Job(
                job_id=job_id,
                command=command,
                cwd=cwd,
                proc=None,
                win_job=None,
                started=time.monotonic(),
                notify_on_complete=notify,
                kind=kind,
                owner_session_id=owner_session_id,
                delivery=delivery,
                notice_id=f"{job_id}:complete",
            )
            self._jobs[job_id] = job
        try:
            threading.Thread(
                target=self._run_task,
                args=(job, runner),
                name=f"task-{job.job_id}",
                daemon=True,
            ).start()
        except Exception:
            # 极端情况下线程创建失败，回滚预留，不留下永远 running 的记录。
            with self._lock:
                self._jobs.pop(job.job_id, None)
            raise
        return job

    def _run_task(self, job: Job, runner: Callable[[Job], object]) -> None:
        try:
            output = runner(job)
            job.result = str(getattr(output, "text", output) or "")
            job.meta = dict(getattr(output, "meta", {}) or {})
        except ToolError as exc:
            job.error = str(exc)
            job.error_code = exc.error_code
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            proc = job.proc
            if proc is not None:
                job.exit_code = getattr(proc, "returncode", None)
            if job.exit_code is None:
                job.exit_code = 1 if job.error else 0
            job.finished_at = time.monotonic()
            self._flush_log(job)
            self._publish(job)

    def _supervise(self, job: Job) -> None:
        proc = job.proc
        pumps: list[threading.Thread] = []
        if getattr(proc, "stdout", None) is not None:
            pumps.append(threading.Thread(
                target=self._pump, args=(job, proc.stdout, "stdout"), daemon=True
            ))
        if getattr(proc, "stderr", None) is not None:
            pumps.append(threading.Thread(
                target=self._pump, args=(job, proc.stderr, "stderr"), daemon=True
            ))
        for t in pumps:
            t.start()
        try:
            proc.wait()
        except Exception:
            pass
        for t in pumps:
            t.join(timeout=2)
        job.exit_code = proc.returncode
        job.finished_at = time.monotonic()
        self._flush_log(job)
        self._publish(job)

    def _publish(self, job: Job) -> None:
        """只发布一次固定 notice_id；消费者再原子竞争消费权。"""
        with self._lock:
            if job.notify_on_complete and self.allow_notify:
                job.notification_state = "pending"
            listeners = list(self._listeners) if job.notification_state == "pending" else []
        completion = self._completion(job)
        # 先把通知放进各消费者队列，再唤醒 done 等待者。这样快速完成时，
        # Agent 在工具返回后的 drain 不会跑在通知入队之前。
        for fn in listeners:
            try:
                fn(completion)
            except Exception:
                pass
        job.done.set()

    def consume_notification(self, completion: JobCompletion | str) -> JobCompletion | None:
        """消费一条完成通知。显式控制、忙碌注入、空闲唤醒只会有一个成功。"""
        original = completion if isinstance(completion, JobCompletion) else None
        if original is not None:
            job_id = original.job_id
            notice_id = original.notice_id
        else:
            job_id = str(completion)
            notice_id = ""
        with self._lock:
            job = self._jobs.get(job_id)
            if (
                job is None
                or job.notification_state != "pending"
                or (notice_id and notice_id != job.notice_id)
            ):
                return None
            job.notification_state = "consumed"
        # 发布时生成的 completion 已是终态快照；即使消费者在 done.set 前运行，
        # 也不能重新生成成 running 报告。
        return original if original is not None else self._completion(job)

    def _consume_job_notification(self, job: Job) -> None:
        self.consume_notification(job.job_id)

    def _pump(self, job: Job, pipe, kind: str) -> None:
        try:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                job.append(kind, decode_output(chunk))
        except (OSError, ValueError):
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _flush_log(self, job: Job) -> None:
        out, err = job.snapshot()
        text = self._raw_output(out, err)
        if job.kind != "shell":
            parts = [] if text == "(无输出)" else [text]
            if job.result:
                parts.append("--- result ---\n" + job.result.rstrip())
            if job.error:
                parts.append("--- error ---\n" + job.error.rstrip())
            text = "\n".join(parts) if parts else "(无输出)"
        try:
            spill_dir = project_data_dir() / "spill"
            spill_dir.mkdir(parents=True, exist_ok=True)
            path = spill_dir / f"{job.job_id}.log"
            path.write_text(text, encoding="utf-8")
            job.log_path = path
        except OSError:
            pass

    @staticmethod
    def _raw_output(out: str, err: str) -> str:
        parts: list[str] = []
        if out.strip():
            parts.append("--- stdout ---\n" + out.rstrip())
        if err.strip():
            parts.append("--- stderr ---\n" + err.rstrip())
        if not parts:
            return "(无输出)"
        return "\n".join(parts)

    def _completion(self, job: Job) -> JobCompletion:
        out, err = job.snapshot()
        message = format_job_report(
            job,
            headline="[后台任务完成通知]",
            out=out,
            err=err,
        )
        return JobCompletion(
            job_id=job.job_id,
            command=job.command,
            cwd=str(job.cwd),
            exit_code=job.exit_code,
            elapsed=job.elapsed(),
            stdout=out,
            stderr=err,
            message=message,
            kind=job.kind,
            delivery=job.delivery,
            notice_id=job.notice_id,
            result=job.result,
            error=job.error,
        )

    def format_started(self, job: Job, *, notify_requested: bool, upgraded: bool = False) -> str:
        notify = job.notify_on_complete
        if upgraded:
            head = f"[TIMEOUT，已升级为后台任务 {job.job_id}，进程仍在运行，没有被杀掉]"
        else:
            head = f"[已在后台启动 {job.job_id}]"
        notes = [
            head,
            f"command={job.command}",
            f"cwd={job.cwd}  pid={getattr(job.proc, 'pid', '?')}",
        ]
        if notify:
            notes.append("已打开完成通知：任务结束后会把退出码和输出汇报给你，然后停下，不会自动开始下一步。请不要反复 poll。")
        elif notify_requested and not self.allow_notify:
            notes.append(
                "当前是非交互模式，完成通知已关闭（退出后没有人接手）。"
                "请用 process_manage 的 wait/poll 查看；进程退出时仍在跑的任务会被清理。"
            )
        else:
            notes.append("未打开完成通知。可用 process_manage 的 poll/log/wait/kill 查看或取消。")
        out, err = job.snapshot()
        body = self._clip_pair(out, err)
        if body != "(无输出)":
            notes.append(body)
        return "\n".join(notes)

    def _clip_pair(self, out: str, err: str) -> str:
        parts: list[str] = []
        if out.strip():
            parts.append("--- stdout ---\n" + clip_output(out.rstrip(), "stdout"))
        if err.strip():
            parts.append("--- stderr ---\n" + clip_output(err.rstrip(), "stderr"))
        return "\n".join(parts) if parts else "(无输出)"

    def _mark_view(self, job: Job) -> bool:
        """连续两次查看都没有新输出时，打开完成通知。返回是否刚触发。"""
        if not job.running():
            return False
        current = job.output_len()
        if job.last_viewed_len >= 0 and current == job.last_viewed_len:
            job.stale_views += 1
        else:
            job.stale_views = 0
        job.last_viewed_len = current
        if job.stale_views >= 2 and not job.notify_on_complete:
            if self.allow_notify:
                job.notify_on_complete = True
                return True
        return False

    def manage(self, action: str, job_id: str | None = None, timeout: int | None = None) -> str:
        """旧 Shell 控制接口，输出与 action 名称保持兼容。"""
        action = (action or "").strip().lower()
        if action not in {"list", "poll", "log", "wait", "kill"}:
            return "ERROR: action 必须是 list、poll、log、wait、kill 之一"
        if action == "list":
            return self._list()
        if not job_id:
            return f"ERROR: {action} 需要 job_id"
        job = self.get(job_id)
        if job is None:
            return f"ERROR: 找不到任务 {job_id}"
        if action == "poll":
            return self._poll(job)
        if action == "log":
            return self._log(job)
        if action == "wait":
            limit = DEFAULT_WAIT if timeout is None else int(timeout)
            limit = max(1, min(limit, HARD_TIMEOUT))
            finished = job.done.wait(timeout=limit)
            if finished:
                self._consume_job_notification(job)
                return self._report(job, headline="[已结束]", switched=False)
            switched = self._mark_view(job)
            return self._report(job, headline=f"[仍在运行，已等待 {limit}s]", switched=switched)
        return self._kill(job)

    def task_manage(
        self,
        action: str,
        task_id: str | None = None,
        timeout: int | None = None,
        owner_session_id: str | None = None,
    ) -> str:
        """统一控制接口：get 不消费通知，终态 output/wait/cancel 会消费。"""
        action = (action or "").strip().lower()
        if action not in {"list", "get", "output", "wait", "cancel"}:
            return "ERROR: action 必须是 list、get、output、wait、cancel 之一"
        if action == "list":
            return self._task_list(owner_session_id)
        if not task_id:
            return f"ERROR: {action} 需要 task_id"
        job = self.get(task_id)
        if job is None or (
            owner_session_id is not None
            and job.owner_session_id not in {None, owner_session_id}
        ):
            return f"ERROR: 找不到任务 {task_id}"
        if action == "get":
            return self._report(job, headline=None, switched=False)
        if action == "output":
            if not job.running():
                self._consume_job_notification(job)
            return self._report(job, headline=None, switched=False)
        if action == "wait":
            limit = DEFAULT_WAIT if timeout is None else int(timeout)
            limit = max(1, min(limit, TASK_HARD_TIMEOUT))
            finished = job.done.wait(timeout=limit)
            if finished:
                self._consume_job_notification(job)
                return self._report(job, headline="[已结束]", switched=False)
            return self._report(job, headline=f"[仍在运行，已等待 {limit}s；任务未被取消]", switched=False)
        return self._kill(job)

    def _list(self) -> str:
        jobs = self.all_jobs()
        if not jobs:
            return "(没有后台任务)"
        lines = []
        for job in jobs:
            status = "running" if job.running() else f"exited={job.exit_code}"
            notify = "notify=on" if job.notify_on_complete else "notify=off"
            cmd = job.command if len(job.command) <= 80 else job.command[:77] + "..."
            lines.append(
                f"{job.job_id}  {status}  {notify}  time={job.elapsed():.2f}s  "
                f"pid={getattr(job.proc, 'pid', '?')}  {cmd}"
            )
        return "\n".join(lines)

    def _task_list(self, owner_session_id: str | None = None) -> str:
        jobs = [
            j for j in self.all_jobs()
            if owner_session_id is None
            or j.owner_session_id in {None, owner_session_id}
        ]
        if not jobs:
            return "(没有任务)"
        lines = []
        for job in jobs:
            cmd = job.command if len(job.command) <= 80 else job.command[:77] + "..."
            lines.append(
                f"{job.job_id}  kind={job.kind}  status={job.status}  "
                f"notice={job.notification_state}  time={job.elapsed():.2f}s  {cmd}"
            )
        return "\n".join(lines)

    def _poll(self, job: Job) -> str:
        switched = self._mark_view(job)
        return self._report(job, headline=None, switched=switched)

    def _log(self, job: Job) -> str:
        self._flush_log(job)
        out, err = job.snapshot()
        loc = f"log={job.log_path}" if job.log_path else "log=(未能落盘)"
        return self._report(job, headline=loc, switched=False, out=out, err=err)

    def _kill(self, job: Job) -> str:
        if not job.running():
            self._consume_job_notification(job)
            return self._report(job, headline="[已经结束，无需再杀]", switched=False)
        job.cancel_event.set()
        proc = job.proc
        win_job = job.win_job
        if proc is not None:
            try:
                if win_job is not None and getattr(win_job, "active", False):
                    win_job.kill()
                else:
                    from .tools.shell import kill_process_tree
                    kill_process_tree(proc)
            except Exception:
                try:
                    proc.kill()
                except OSError:
                    pass
        job.done.wait(timeout=KILL_WAIT)
        if job.running() and proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
            job.done.wait(timeout=KILL_WAIT)
        if not job.running():
            self._consume_job_notification(job)
        return self._report(job, headline="[已发出终止]", switched=False)

    def _report(
        self,
        job: Job,
        headline: str | None,
        switched: bool,
        out: str | None = None,
        err: str | None = None,
    ) -> str:
        if out is None or err is None:
            out, err = job.snapshot()
        extra = ""
        if switched:
            extra = (
                "连续两次查看都没有新输出，已改为完成通知。"
                "请不要再 poll/wait，任务结束后会把结果汇报给你，然后停下。"
            )
        return format_job_report(job, headline=headline, out=out, err=err, extra=extra)

    def cleanup_all(self) -> int:
        """杀掉仍在跑的后台任务并回收句柄。返回杀掉的数量。"""
        running = self.running_jobs()
        for job in running:
            self._kill(job)
        for job in self.all_jobs():
            win_job = job.win_job
            if win_job is not None:
                try:
                    win_job.close()
                except Exception:
                    pass
                job.win_job = None
        return len(running)


# 旧名称保留给 ShellTools 和外部调用；实际实现已是统一任务注册表。
JobRegistry = TaskRegistry


def format_job_report(
    job: Job,
    headline: str | None,
    out: str,
    err: str,
    extra: str = "",
) -> str:
    parts = []
    if headline:
        parts.append(headline)
    if job.kind == "shell":
        status = "running" if job.running() else f"exit={job.exit_code}"
        parts.append(
            f"job_id={job.job_id}  {status}  time={job.elapsed():.2f}s  "
            f"cwd={job.cwd}  notify={'on' if job.notify_on_complete else 'off'}"
        )
        parts.append(f"command={job.command}")
        if extra:
            parts.append(extra)
        if out.strip():
            parts.append("--- stdout ---\n" + clip_output(out.rstrip(), f"{job.job_id}_stdout"))
        if err.strip():
            parts.append("--- stderr ---\n" + clip_output(err.rstrip(), f"{job.job_id}_stderr"))
        if not out.strip() and not err.strip():
            parts.append("(无输出)")
        return "\n".join(parts)

    parts.append(
        f"task_id={job.job_id}  kind={job.kind}  status={job.status}  "
        f"time={job.elapsed():.2f}s  cwd={job.cwd}  notice={job.notification_state}"
    )
    parts.append(f"command={job.command}")
    if extra:
        parts.append(extra)
    if job.result:
        parts.append("--- result ---\n" + clip_output(job.result.rstrip(), f"{job.job_id}_result"))
    if job.error:
        parts.append(f"error={job.error}")
    if err.strip():
        parts.append("--- stderr ---\n" + clip_output(err.rstrip(), f"{job.job_id}_stderr"))
    if not job.result and not job.error and not err.strip():
        if out.strip():
            parts.append("--- stdout ---\n" + clip_output(out.rstrip(), f"{job.job_id}_stdout"))
        else:
            parts.append("(无输出)")
    return "\n".join(parts)
