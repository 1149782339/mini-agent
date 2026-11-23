"""统一后台任务注册表。

Shell 与 Subagent 共用任务 ID、状态、输出、取消和完成通知。
可选 TaskStore 持久化任务快照；重启后恢复终态，运行中的旧任务标记为 lost。
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from .store import TaskStore, project_data_dir
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


def clip_output(text: str, tag: str = "task") -> str:
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
class TaskCompletion:
    task_id: str
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


@dataclass
class Task:
    task_id: str
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
    lost: bool = False
    timed_out: bool = False
    deadline: float | None = None  # 前台转后台后仍要遵守的总运行时限
    kill_fn: Callable[[], None] | None = None
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
        if self.finished_at is None and self.running():
            return "running"
        if self.lost:
            return "lost"
        if self.timed_out:
            return "timeout"
        if self.cancel_event.is_set():
            return "cancelled"
        if self.error:
            return "failed"
        if self.kind == "shell":
            return f"exited={self.exit_code}"
        return "completed"


class TaskRegistry:
    def __init__(
        self,
        allow_notify: bool = True,
        task_store: TaskStore | None = None,
        owner_session_id: str | None = None,
    ) -> None:
        self.allow_notify = allow_notify
        self.allow_background_subagents = True
        self.task_store = task_store
        self.owner_session_id = owner_session_id
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._listeners: list[Callable[[TaskCompletion], None]] = []
        self._recovered_pending: set[str] = set()
        if self.task_store is not None:
            self._restore()

    def on_complete(self, fn: Callable[[TaskCompletion], None]) -> None:
        with self._lock:
            self._listeners.append(fn)
            recovered = [
                self._tasks[task_id] for task_id in self._recovered_pending
                if self._tasks[task_id].notification_state == "pending"
            ]
        # 每个监听器都收到同一 notice_id，最终仍由 consume_notification 原子去重。
        for task in recovered:
            try:
                fn(self._completion(task))
            except Exception:
                pass

    def _restore(self) -> None:
        """恢复快照；上次进程留下的 running 不可能继续执行，只能标记 lost。"""
        for data in self.task_store.load_tasks():
            task_id = str(data["task_id"])
            suffix = task_id.rsplit("_", 1)[-1]
            if suffix.isdigit():
                self._seq = max(self._seq, int(suffix))
            if data.get("status") == "running":
                data["status"] = "lost"
                data["lost"] = True
                data["error"] = "进程重启前任务仍在运行，执行上下文已丢失；未自动重跑"
                data["notification_state"] = (
                    "pending"
                    if data.get("notify_on_complete") and data.get("delivery") != "none"
                    else "none"
                )
            elapsed = max(0.0, float(data.get("elapsed") or 0.0))
            now = time.monotonic()
            task = Task(
                task_id=task_id,
                command=str(data.get("command") or ""),
                cwd=Path(data.get("cwd") or "."),
                proc=None,
                win_job=None,
                started=now - elapsed,
                notify_on_complete=bool(data.get("notify_on_complete")),
                upgraded_from_timeout=bool(data.get("upgraded_from_timeout")),
                kind=str(data.get("kind") or "shell"),
                owner_session_id=data.get("owner_session_id"),
                delivery=data.get("delivery") or "record",
                notice_id=str(data.get("notice_id") or f"{task_id}:complete"),
                notification_state=str(data.get("notification_state") or "none"),
                result=str(data.get("result") or ""),
                error=str(data.get("error") or ""),
                error_code=data.get("error_code"),
                lost=bool(data.get("lost")),
                timed_out=bool(data.get("timed_out")),
                meta=dict(data.get("meta") or {}),
                stdout_parts=[str(data.get("stdout") or "")],
                stderr_parts=[str(data.get("stderr") or "")],
                exit_code=data.get("exit_code"),
                finished_at=now,
            )
            if data.get("status") == "cancelled":
                task.cancel_event.set()
            task.done.set()
            self._tasks[task_id] = task
            if data.get("status") == "lost":
                self._persist(task)
            if (
                self.allow_notify
                and task.notification_state == "pending"
                and self.owner_session_id in {None, task.owner_session_id}
            ):
                self._recovered_pending.add(task_id)

    @staticmethod
    def _storage_status(task: Task) -> str:
        if task.finished_at is None:
            return "running"
        if task.lost:
            return "lost"
        if task.cancel_event.is_set():
            return "cancelled"
        if task.error:
            return "failed"
        return "completed"

    def _persist(self, task: Task) -> None:
        if self.task_store is None:
            return
        out, err = task.snapshot()
        snapshot = {
            "task_id": task.task_id,
            "command": task.command,
            "cwd": str(task.cwd),
            "kind": task.kind,
            "owner_session_id": task.owner_session_id,
            "delivery": task.delivery,
            "notify_on_complete": task.notify_on_complete,
            "notification_state": task.notification_state,
            "notice_id": task.notice_id,
            "status": self._storage_status(task),
            "exit_code": task.exit_code,
            "result": task.result,
            "error": task.error,
            "error_code": task.error_code,
            "lost": task.lost,
            "timed_out": task.timed_out,
            "meta": task.meta,
            "stdout": out,
            "stderr": err,
            "elapsed": task.elapsed(),
            "upgraded_from_timeout": task.upgraded_from_timeout,
        }
        try:
            self.task_store.save_task(snapshot)
        except (OSError, sqlite3.Error):
            # 持久化失败不能中断已经运行的进程；本次生命周期仍保持 process-local。
            pass

    def persist(self, task: Task) -> None:
        """供 Shell 超时升级为后台任务后同步通知开关。"""
        self._persist(task)

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def all_tasks(self) -> list[Task]:
        with self._lock:
            return list(self._tasks.values())

    def running_tasks(self) -> list[Task]:
        return [j for j in self.all_tasks() if j.running()]

    def _next_id(self) -> str:
        self._seq += 1
        return f"task_{self._seq:02d}"

    def add_running(
        self,
        proc,
        command: str,
        cwd: Path,
        win_job: object | None = None,
        notify_on_complete: bool = False,
        upgraded_from_timeout: bool = False,
        kill_fn: Callable[[], None] | None = None,
        initial_stdout: bytes | str | None = None,
        initial_stderr: bytes | str | None = None,
    ) -> Task:
        """登记一个已经启动、仍在跑的 Shell 进程，并开始收集输出。"""
        with self._lock:
            task_id = self._next_id()
            notify = bool(notify_on_complete) and self.allow_notify
            task = Task(
                task_id=task_id,
                command=command,
                cwd=cwd,
                proc=proc,
                win_job=win_job,
                started=time.monotonic(),
                notify_on_complete=notify,
                upgraded_from_timeout=upgraded_from_timeout,
                kill_fn=kill_fn,
                kind="shell",
                # 开了通知就唤醒 Agent 继续处理；否则只记录，等用户下次说话。
                delivery="wake" if notify else "record",
                notice_id=f"{task_id}:complete",
            )
            self._tasks[task_id] = task
        if initial_stdout:
            task.append("stdout", decode_output(initial_stdout))
        if initial_stderr:
            task.append("stderr", decode_output(initial_stderr))
        self._flush_log(task)
        self._persist(task)
        threading.Thread(target=self._supervise, args=(task,), name=f"task-{task.task_id}", daemon=True).start()
        return task

    def add_task(
        self,
        runner: Callable[[Task], object],
        *,
        command: str,
        cwd: Path,
        kind: str = "subagent",
        owner_session_id: str | None = None,
        delivery: Delivery = "wake",
        max_active: int | None = None,
    ) -> Task:
        """原子完成准入与登记，再由后台线程执行 runner。

        登记本身就是活跃名额预留；runner 无论成功失败都会进入终态并释放名额。
        """
        with self._lock:
            if max_active is not None:
                active = sum(
                    j.running() and j.kind == kind and j.owner_session_id == owner_session_id
                    for j in self._tasks.values()
                )
                if active >= max_active:
                    raise TaskLimitError(
                        f"owner={owner_session_id or 'default'} 的活跃 {kind} 已达上限 "
                        f"{max_active}，请等待完成或先取消任务"
                    )
            task_id = self._next_id()
            notify = delivery != "none" and self.allow_notify
            task = Task(
                task_id=task_id,
                command=command,
                cwd=cwd,
                proc=None,
                win_job=None,
                started=time.monotonic(),
                notify_on_complete=notify,
                kind=kind,
                owner_session_id=owner_session_id,
                delivery=delivery,
                notice_id=f"{task_id}:complete",
            )
            self._tasks[task_id] = task
        self._persist(task)
        try:
            threading.Thread(
                target=self._run_task,
                args=(task, runner),
                name=f"task-{task.task_id}",
                daemon=True,
            ).start()
        except Exception:
            # 极端情况下线程创建失败，回滚预留，不留下永远 running 的记录。
            with self._lock:
                self._tasks.pop(task.task_id, None)
            if self.task_store is not None:
                try:
                    self.task_store.delete_task(task.task_id)
                except (OSError, sqlite3.Error):
                    pass
            raise
        return task

    def _run_task(self, task: Task, runner: Callable[[Task], object]) -> None:
        try:
            output = runner(task)
            task.result = str(getattr(output, "text", output) or "")
            task.meta = dict(getattr(output, "meta", {}) or {})
        except ToolError as exc:
            task.error = str(exc)
            task.error_code = exc.error_code
        except Exception as exc:
            task.error = f"{type(exc).__name__}: {exc}"
        finally:
            proc = task.proc
            if proc is not None:
                task.exit_code = getattr(proc, "returncode", None)
            if task.exit_code is None:
                task.exit_code = 1 if task.error else 0
            task.finished_at = time.monotonic()
            self._flush_log(task)
            self._publish(task)

    def _supervise(self, task: Task) -> None:
        proc = task.proc
        pumps: list[threading.Thread] = []
        if getattr(proc, "stdout", None) is not None:
            pumps.append(threading.Thread(
                target=self._pump, args=(task, proc.stdout, "stdout"), daemon=True
            ))
        if getattr(proc, "stderr", None) is not None:
            pumps.append(threading.Thread(
                target=self._pump, args=(task, proc.stderr, "stderr"), daemon=True
            ))
        for t in pumps:
            t.start()
        # 轮询而不是无限 wait：前台转后台后 deadline 才被设置，需要能中途生效。
        try:
            while True:
                if task.deadline is not None and time.monotonic() >= task.deadline:
                    task.timed_out = True
                    if task.kill_fn is not None:
                        task.kill_fn()
                    else:
                        proc.kill()
                    try:
                        proc.wait(timeout=KILL_WAIT)
                    except Exception:
                        pass
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
        except Exception:
            pass
        for t in pumps:
            t.join(timeout=2)
        task.exit_code = proc.returncode
        task.finished_at = time.monotonic()
        self._flush_log(task)
        self._publish(task)

    def _publish(self, task: Task) -> None:
        """只发布一次固定 notice_id；消费者再原子竞争消费权。"""
        with self._lock:
            if task.notify_on_complete and self.allow_notify:
                task.notification_state = "pending"
            listeners = list(self._listeners) if task.notification_state == "pending" else []
        completion = self._completion(task)
        # 终态与 pending 先落库，再入队；崩溃发生在入队前也能在下次启动补投。
        self._persist(task)
        # 先把通知放进各消费者队列，再唤醒 done 等待者。这样快速完成时，
        # Agent 在工具返回后的 drain 不会跑在通知入队之前。
        for fn in listeners:
            try:
                fn(completion)
            except Exception:
                pass
        task.done.set()

    def consume_notification(self, completion: TaskCompletion | str) -> TaskCompletion | None:
        """消费一条完成通知。显式控制、忙碌注入、空闲唤醒只会有一个成功。"""
        original = completion if isinstance(completion, TaskCompletion) else None
        if original is not None:
            task_id = original.task_id
            notice_id = original.notice_id
        else:
            task_id = str(completion)
            notice_id = ""
        with self._lock:
            task = self._tasks.get(task_id)
            if (
                task is None
                or task.notification_state != "pending"
                or (notice_id and notice_id != task.notice_id)
            ):
                return None
            task.notification_state = "consumed"
        self._persist(task)
        # 发布时生成的 completion 已是终态快照；即使消费者在 done.set 前运行，
        # 也不能重新生成成 running 报告。
        return original if original is not None else self._completion(task)

    def _consume_task_notification(self, task: Task) -> None:
        self.consume_notification(task.task_id)

    def _pump(self, task: Task, pipe, kind: str) -> None:
        # read1：有多少拿多少，别等缓冲区攒满——否则转后台那一刻拿不到已有输出。
        read = getattr(pipe, "read1", None) or pipe.read
        try:
            while True:
                chunk = read(4096)
                if not chunk:
                    break
                task.append(kind, decode_output(chunk))
        except (OSError, ValueError):
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _flush_log(self, task: Task) -> None:
        out, err = task.snapshot()
        text = self._raw_output(out, err)
        if task.kind != "shell":
            parts = [] if text == "(无输出)" else [text]
            if task.result:
                parts.append("--- result ---\n" + task.result.rstrip())
            if task.error:
                parts.append("--- error ---\n" + task.error.rstrip())
            text = "\n".join(parts) if parts else "(无输出)"
        try:
            spill_dir = project_data_dir() / "spill"
            spill_dir.mkdir(parents=True, exist_ok=True)
            path = spill_dir / f"{task.task_id}.log"
            path.write_text(text, encoding="utf-8")
            task.log_path = path
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

    def _completion(self, task: Task) -> TaskCompletion:
        out, err = task.snapshot()
        message = format_task_report(
            task,
            headline="[后台任务完成通知]",
            out=out,
            err=err,
        )
        return TaskCompletion(
            task_id=task.task_id,
            command=task.command,
            cwd=str(task.cwd),
            exit_code=task.exit_code,
            elapsed=task.elapsed(),
            stdout=out,
            stderr=err,
            message=message,
            kind=task.kind,
            delivery=task.delivery,
            notice_id=task.notice_id,
            result=task.result,
            error=task.error,
        )

    def format_started(self, task: Task, *, notify_requested: bool, upgraded: bool = False) -> str:
        notify = task.notify_on_complete
        if upgraded:
            if task.deadline is None:
                head = (
                    f"[前台阻塞预算已用完，已升级为后台任务 {task.task_id}，"
                    "进程仍在运行，没有被杀掉]"
                )
            else:
                remaining = max(0.0, task.deadline - time.monotonic())
                head = (
                    f"[前台阻塞预算已用完，已升级为后台任务 {task.task_id}，"
                    f"进程仍在运行；最长再跑 {remaining:.0f}s 后终止]"
                )
        else:
            head = f"[已在后台启动 {task.task_id}]"
        notes = [
            head,
            f"command={task.command}",
            f"cwd={task.cwd}  pid={getattr(task.proc, 'pid', '?')}",
        ]
        if notify:
            notes.append(
                "已打开完成通知：任务结束后会唤醒 Agent，把退出码和输出送回来继续处理。"
                "不要重复执行同一命令，也不要反复 poll。"
            )
        elif notify_requested and not self.allow_notify:
            notes.append(
                "当前是非交互模式，完成通知已关闭（退出后没有人接手）。"
                "请用 process_manage 的 wait/poll 查看；进程退出时仍在跑的任务会被清理。"
            )
        else:
            notes.append("未打开完成通知。可用 process_manage 的 poll/log/wait/kill 查看或取消。")
        out, err = task.snapshot()
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

    def _mark_view(self, task: Task) -> bool:
        """连续两次查看都没有新输出时，打开完成通知。返回是否刚触发。"""
        if not task.running():
            return False
        current = task.output_len()
        if task.last_viewed_len >= 0 and current == task.last_viewed_len:
            task.stale_views += 1
        else:
            task.stale_views = 0
        task.last_viewed_len = current
        if task.stale_views >= 2 and not task.notify_on_complete:
            if self.allow_notify:
                task.notify_on_complete = True
                self._persist(task)
                return True
        return False

    def manage(self, action: str, task_id: str | None = None, timeout: int | None = None) -> str:
        """旧 Shell 控制接口，输出与 action 名称保持兼容。"""
        action = (action or "").strip().lower()
        if action not in {"list", "poll", "log", "wait", "kill"}:
            return "ERROR: action 必须是 list、poll、log、wait、kill 之一"
        if action == "list":
            return self._list()
        if not task_id:
            return f"ERROR: {action} 需要 task_id"
        task = self.get(task_id)
        if task is None:
            return f"ERROR: 找不到任务 {task_id}"
        if action == "poll":
            return self._poll(task)
        if action == "log":
            return self._log(task)
        if action == "wait":
            limit = DEFAULT_WAIT if timeout is None else int(timeout)
            limit = max(1, min(limit, HARD_TIMEOUT))
            finished = task.done.wait(timeout=limit)
            if finished:
                self._consume_task_notification(task)
                return self._report(task, headline="[已结束]", switched=False)
            switched = self._mark_view(task)
            return self._report(task, headline=f"[仍在运行，已等待 {limit}s]", switched=switched)
        return self._kill(task)

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
        task = self.get(task_id)
        if task is None or (
            owner_session_id is not None
            and task.owner_session_id not in {None, owner_session_id}
        ):
            return f"ERROR: 找不到任务 {task_id}"
        if action == "get":
            return self._report(task, headline=None, switched=False)
        if action == "output":
            if not task.running():
                self._consume_task_notification(task)
            return self._report(task, headline=None, switched=False)
        if action == "wait":
            limit = DEFAULT_WAIT if timeout is None else int(timeout)
            limit = max(1, min(limit, TASK_HARD_TIMEOUT))
            finished = task.done.wait(timeout=limit)
            if finished:
                self._consume_task_notification(task)
                return self._report(task, headline="[已结束]", switched=False)
            return self._report(task, headline=f"[仍在运行，已等待 {limit}s；任务未被取消]", switched=False)
        return self._kill(task)

    def _list(self) -> str:
        tasks = self.all_tasks()
        if not tasks:
            return "(没有后台任务)"
        lines = []
        for task in tasks:
            status = "running" if task.running() else ("lost" if task.lost else f"exited={task.exit_code}")
            notify = "notify=on" if task.notify_on_complete else "notify=off"
            cmd = task.command if len(task.command) <= 80 else task.command[:77] + "..."
            lines.append(
                f"{task.task_id}  {status}  {notify}  time={task.elapsed():.2f}s  "
                f"pid={getattr(task.proc, 'pid', '?')}  {cmd}"
            )
        return "\n".join(lines)

    def _task_list(self, owner_session_id: str | None = None) -> str:
        tasks = [
            j for j in self.all_tasks()
            if owner_session_id is None
            or j.owner_session_id in {None, owner_session_id}
        ]
        if not tasks:
            return "(没有任务)"
        lines = []
        for task in tasks:
            cmd = task.command if len(task.command) <= 80 else task.command[:77] + "..."
            lines.append(
                f"{task.task_id}  kind={task.kind}  status={task.status}  "
                f"notice={task.notification_state}  time={task.elapsed():.2f}s  {cmd}"
            )
        return "\n".join(lines)

    def _poll(self, task: Task) -> str:
        switched = self._mark_view(task)
        return self._report(task, headline=None, switched=switched)

    def _log(self, task: Task) -> str:
        self._flush_log(task)
        out, err = task.snapshot()
        loc = f"log={task.log_path}" if task.log_path else "log=(未能落盘)"
        return self._report(task, headline=loc, switched=False, out=out, err=err)

    def _kill(self, task: Task) -> str:
        if not task.running():
            self._consume_task_notification(task)
            return self._report(task, headline="[已经结束，无需再杀]", switched=False)
        task.cancel_event.set()
        proc = task.proc
        win_job = task.win_job
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
        task.done.wait(timeout=KILL_WAIT)
        if task.running() and proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
            task.done.wait(timeout=KILL_WAIT)
        if not task.running():
            self._consume_task_notification(task)
        return self._report(task, headline="[已发出终止]", switched=False)

    def _report(
        self,
        task: Task,
        headline: str | None,
        switched: bool,
        out: str | None = None,
        err: str | None = None,
    ) -> str:
        if out is None or err is None:
            out, err = task.snapshot()
        extra = ""
        if switched:
            extra = (
                "连续两次查看都没有新输出，已改为完成通知。"
                "请不要再 poll/wait，任务结束后会把结果汇报给你，然后停下。"
            )
        return format_task_report(task, headline=headline, out=out, err=err, extra=extra)

    def cleanup_all(self) -> int:
        """杀掉仍在跑的后台任务并回收句柄。返回杀掉的数量。"""
        running = self.running_tasks()
        for task in running:
            self._kill(task)
        for task in self.all_tasks():
            win_job = task.win_job
            if win_job is not None:
                try:
                    win_job.close()
                except Exception:
                    pass
                task.win_job = None
        return len(running)


# 旧名称保留给外部调用；实际实现已是统一任务注册表。
Job = Task
JobRegistry = TaskRegistry


def format_task_report(
    task: Task,
    headline: str | None,
    out: str,
    err: str,
    extra: str = "",
) -> str:
    parts = []
    if headline:
        parts.append(headline)
    if task.kind == "shell":
        # 通知早于 done.set() 入队，不能用 running() 判断，否则会显示成 running。
        status = "running" if task.finished_at is None else (
            "lost" if task.lost else ("timeout" if task.timed_out else f"exit={task.exit_code}")
        )
        parts.append(
            f"task_id={task.task_id}  {status}  time={task.elapsed():.2f}s  "
            f"cwd={task.cwd}  notify={'on' if task.notify_on_complete else 'off'}"
        )
        parts.append(f"command={task.command}")
        if extra:
            parts.append(extra)
        if task.lost and task.error:
            parts.append(f"error={task.error}")
        if out.strip():
            parts.append("--- stdout ---\n" + clip_output(out.rstrip(), f"{task.task_id}_stdout"))
        if err.strip():
            parts.append("--- stderr ---\n" + clip_output(err.rstrip(), f"{task.task_id}_stderr"))
        if not out.strip() and not err.strip():
            parts.append("(无输出)")
        return "\n".join(parts)

    parts.append(
        f"task_id={task.task_id}  kind={task.kind}  status={task.status}  "
        f"time={task.elapsed():.2f}s  cwd={task.cwd}"
    )
    parts.append(f"command={task.command}")
    if extra:
        parts.append(extra)
    if task.result:
        parts.append("--- result ---\n" + clip_output(task.result.rstrip(), f"{task.task_id}_result"))
    if task.error:
        parts.append(f"error={task.error}")
    if out.strip() and not task.result:
        parts.append("--- stdout ---\n" + clip_output(out.rstrip(), f"{task.task_id}_stdout"))
    if err.strip():
        parts.append("--- stderr ---\n" + clip_output(err.rstrip(), f"{task.task_id}_stderr"))
    if not task.result and not task.error and not out.strip() and not err.strip():
        parts.append("(无输出)")
    return "\n".join(parts)
