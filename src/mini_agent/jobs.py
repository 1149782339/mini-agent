"""后台命令注册表：瘦身版，只覆盖启动、查询、完成通知和退出清理。

不做的事：watch_patterns、heartbeat、跨会话/跨重启收据、网关自唤醒、
把子代理进程移交给父会话。-p 等非交互模式默认关闭完成通知，
因为退出后没有人接手。
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from .store import project_data_dir
from .tools.base import decode_output

# 与 tools.shell 保持一致；避免顶层互相 import。
MAX_BUFFER = 40000
HARD_TIMEOUT = 300
KILL_WAIT = 5

Action = Literal["list", "poll", "log", "wait", "kill"]

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


@dataclass
class Job:
    job_id: str
    command: str
    cwd: Path
    proc: object  # subprocess.Popen
    win_job: object | None
    started: float
    notify_on_complete: bool = False
    upgraded_from_timeout: bool = False
    stdout_parts: list[str] = field(default_factory=list)
    stderr_parts: list[str] = field(default_factory=list)
    exit_code: int | None = None
    finished_at: float | None = None
    output_lock: threading.Lock = field(default_factory=threading.Lock)
    done: threading.Event = field(default_factory=threading.Event)
    last_viewed_len: int = -1
    stale_views: int = 0
    log_path: Path | None = None

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


class JobRegistry:
    def __init__(self, allow_notify: bool = True) -> None:
        self.allow_notify = allow_notify
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._listeners: list[Callable[[JobCompletion], None]] = []

    def on_complete(self, fn: Callable[[JobCompletion], None]) -> None:
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
        """登记一个已经启动、仍在跑的进程，并开始收集输出。"""
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
            )
            self._jobs[job_id] = job
        if initial_stdout:
            job.append("stdout", decode_output(initial_stdout))
        if initial_stderr:
            job.append("stderr", decode_output(initial_stderr))
        self._flush_log(job)
        threading.Thread(target=self._supervise, args=(job,), name=f"job-{job.job_id}", daemon=True).start()
        return job

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
        job.done.set()
        if job.notify_on_complete and self.allow_notify:
            completion = self._completion(job)
            for fn in list(self._listeners):
                try:
                    fn(completion)
                except Exception:
                    pass

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
        )

    def format_started(self, job: Job, *, notify_requested: bool, upgraded: bool = False) -> str:
        notify = job.notify_on_complete
        if upgraded:
            head = (
                f"[TIMEOUT，已升级为后台任务 {job.job_id}，进程仍在运行，没有被杀掉]"
            )
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
            return self._wait(job, limit)
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

    def _poll(self, job: Job) -> str:
        switched = self._mark_view(job)
        return self._report(job, headline=None, switched=switched)

    def _log(self, job: Job) -> str:
        self._flush_log(job)
        out, err = job.snapshot()
        loc = f"log={job.log_path}" if job.log_path else "log=(未能落盘)"
        return self._report(job, headline=loc, switched=False, out=out, err=err)

    def _wait(self, job: Job, limit: int) -> str:
        finished = job.done.wait(timeout=limit)
        if not finished:
            switched = self._mark_view(job)
            return self._report(
                job,
                headline=f"[仍在运行，已等待 {limit}s]",
                switched=switched,
            )
        return self._report(job, headline="[已结束]", switched=False)

    def _kill(self, job: Job) -> str:
        if not job.running():
            return self._report(job, headline="[已经结束，无需再杀]", switched=False)
        proc = job.proc
        win_job = job.win_job
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
        if job.running():
            try:
                proc.kill()
            except OSError:
                pass
            job.done.wait(timeout=KILL_WAIT)
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


def format_job_report(
    job: Job,
    headline: str | None,
    out: str,
    err: str,
    extra: str = "",
) -> str:
    status = "running" if job.running() else f"exit={job.exit_code}"
    parts = []
    if headline:
        parts.append(headline)
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
