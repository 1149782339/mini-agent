"""后台命令、完成通知、process_manage 与超时升级。"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from mini_agent.cli import InteractiveSession
from mini_agent.jobs import TaskCompletion, TaskLimitError, TaskRegistry
from mini_agent.store import TaskStore
from mini_agent.tools.shell import ShellTools


def test_background_returns_immediately(tmp_path: Path) -> None:
    jobs = TaskRegistry(allow_notify=True)
    shell = ShellTools(tmp_path, default_timeout=10, jobs=jobs)
    started = time.monotonic()
    out = shell.run_command("sleep 2; echo done-bg", background=True)
    elapsed = time.monotonic() - started
    try:
        assert "task_01" in out
        assert "已在后台启动" in out
        assert elapsed < 1.2
        listed = shell.process_manage("list")
        assert "task_01" in listed
        assert "running" in listed
    finally:
        jobs.cleanup_all()


def test_notify_fires_on_complete(tmp_path: Path) -> None:
    jobs = TaskRegistry(allow_notify=True)
    seen: list[TaskCompletion] = []
    jobs.on_complete(seen.append)
    shell = ShellTools(tmp_path, default_timeout=10, jobs=jobs)
    out = shell.run_command("echo hello-bg", background=True, notify_on_complete=True)
    assert "已打开完成通知" in out
    deadline = time.monotonic() + 5
    while not seen and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        assert seen, "完成通知没有触发"
        assert seen[0].exit_code == 0
        assert "hello-bg" in seen[0].message
        assert "exit=0" in seen[0].message
        assert "running" not in seen[0].message
        assert seen[0].task_id == "task_01"
    finally:
        jobs.cleanup_all()


def test_process_manage_wait_and_kill(tmp_path: Path) -> None:
    jobs = TaskRegistry(allow_notify=True)
    shell = ShellTools(tmp_path, default_timeout=10, jobs=jobs)
    try:
        shell.run_command("echo wait-me", background=True)
        waited = shell.process_manage("wait", task_id="task_01", timeout=5)
        assert "exit=0" in waited
        assert "wait-me" in waited
        log = shell.process_manage("log", task_id="task_01")
        assert "wait-me" in log

        shell.run_command("sleep 30", background=True)
        killed = shell.process_manage("kill", task_id="task_02")
        assert "已发出终止" in killed or "exit=" in killed
        # 给回收一点时间
        deadline = time.monotonic() + 3
        while jobs.get("task_02") and jobs.get("task_02").running() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert jobs.get("task_02") is not None
        assert not jobs.get("task_02").running()
    finally:
        jobs.cleanup_all()


def test_stale_polls_enable_notify(tmp_path: Path) -> None:
    jobs = TaskRegistry(allow_notify=True)
    shell = ShellTools(tmp_path, default_timeout=10, jobs=jobs)
    try:
        shell.run_command("sleep 8", background=True)
        first = shell.process_manage("poll", task_id="task_01")
        second = shell.process_manage("poll", task_id="task_01")
        third = shell.process_manage("poll", task_id="task_01")
        assert "改为完成通知" not in first
        assert "改为完成通知" not in second
        assert "改为完成通知" in third
        assert jobs.get("task_01").notify_on_complete
    finally:
        jobs.cleanup_all()


def test_timeout_upgrades_to_background(tmp_path: Path) -> None:
    work = tmp_path / "upgrade"
    work.mkdir()
    marker = work / "marker.txt"
    jobs = TaskRegistry(allow_notify=True)
    shell = ShellTools(work, default_timeout=10, jobs=jobs)
    command = (
        "python3 -c \"import time, pathlib; time.sleep(2); "
        "pathlib.Path('marker.txt').write_text('alive')\""
    )
    try:
        started = time.monotonic()
        out = shell.run_command(command, timeout=1)
        elapsed = time.monotonic() - started
        assert "升级为后台" in out
        assert "task_01" in out
        assert elapsed < 3
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert marker.exists(), "超时后进程被杀掉了，标记文件没有出现"
    finally:
        jobs.cleanup_all()


def test_foreground_block_budget_backgrounds_without_killing(tmp_path: Path) -> None:
    jobs = TaskRegistry(allow_notify=True)
    shell = ShellTools(
        tmp_path, default_timeout=30, foreground_block_budget=0.4, jobs=jobs
    )
    try:
        started = time.monotonic()
        out = shell.run_command("echo early; sleep 5; echo late", timeout=30)
        elapsed = time.monotonic() - started
        assert "已升级为后台" in out
        assert "task_01" in out
        assert "early" in out  # 转后台时要把已有输出带回来
        assert elapsed < 3
        task = jobs.get("task_01")
        assert task.running()
        assert task.done.wait(timeout=8), "转后台后进程被杀了"
        assert task.exit_code == 0
        assert "late" in task.snapshot()[0]
    finally:
        jobs.cleanup_all()


def test_block_until_zero_backgrounds_immediately(tmp_path: Path) -> None:
    jobs = TaskRegistry(allow_notify=True)
    shell = ShellTools(tmp_path, foreground_block_budget=15, jobs=jobs)
    try:
        started = time.monotonic()
        out = shell.run_command("sleep 2", block_until=0, timeout=10)
        assert time.monotonic() - started < 1
        assert "task_01" in out
        assert jobs.get("task_01").running()
    finally:
        jobs.cleanup_all()


def test_upgraded_task_killed_at_total_timeout(tmp_path: Path) -> None:
    work = tmp_path / "deadline"
    work.mkdir()
    marker = work / "late.txt"
    jobs = TaskRegistry(allow_notify=True)
    shell = ShellTools(
        work, default_timeout=30, foreground_block_budget=0.3, jobs=jobs
    )
    command = (
        "python3 -c \"import time, pathlib; time.sleep(5); "
        "pathlib.Path('late.txt').write_text('alive')\""
    )
    try:
        started = time.monotonic()
        out = shell.run_command(command, timeout=2)
        assert time.monotonic() - started < 3
        assert "已升级为后台" in out
        task = jobs.get("task_01")
        assert task.done.wait(timeout=6)
        assert task.timed_out
        assert task.status == "timeout"
        assert not marker.exists(), "总时限到点后进程仍在写文件"
    finally:
        jobs.cleanup_all()


def test_timeout_still_kills_when_notify_disabled(tmp_path: Path) -> None:
    work = tmp_path / "timeout_check"
    work.mkdir()
    marker = work / "marker.txt"
    jobs = TaskRegistry(allow_notify=False)
    shell = ShellTools(work, default_timeout=10, jobs=jobs)
    command = (
        "python3 -c \"import time, pathlib; time.sleep(5); "
        "pathlib.Path('marker.txt').write_text('alive')\""
    )
    started = time.monotonic()
    result = shell.run_command(command, timeout=1)
    elapsed = time.monotonic() - started
    time.sleep(1.5)
    try:
        assert "TIMEOUT" in result
        assert "已杀进程树" in result
        assert not marker.exists(), "非交互超时后进程仍写了标记文件"
        assert elapsed < 8
    finally:
        jobs.cleanup_all()


def test_background_still_blocks_rm(tmp_path: Path) -> None:
    jobs = TaskRegistry()
    shell = ShellTools(tmp_path, jobs=jobs)
    out = shell.run_command("rm -rf /tmp/whatever", background=True)
    assert out.startswith("ERROR:")
    assert "RM 工具" in out
    assert jobs.all_tasks() == []


def test_print_mode_disables_notify(tmp_path: Path) -> None:
    jobs = TaskRegistry(allow_notify=False)
    seen: list[TaskCompletion] = []
    jobs.on_complete(seen.append)
    shell = ShellTools(tmp_path, default_timeout=10, jobs=jobs)
    out = shell.run_command("echo no-notify", background=True, notify_on_complete=True)
    assert "完成通知已关闭" in out or "非交互" in out
    waited = shell.process_manage("wait", task_id="task_01", timeout=5)
    assert "exit=0" in waited
    time.sleep(0.2)
    assert seen == []
    jobs.cleanup_all()


def test_cleanup_all_kills_running_tasks(tmp_path: Path) -> None:
    jobs = TaskRegistry()
    shell = ShellTools(tmp_path, default_timeout=10, jobs=jobs)
    shell.run_command("sleep 30", background=True)
    assert jobs.running_tasks()
    killed = jobs.cleanup_all()
    assert killed == 1
    deadline = time.monotonic() + 3
    while jobs.running_tasks() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not jobs.running_tasks()


class _FakeAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.events: list[str] = []
        self._entered = threading.Event()
        self._release = threading.Event()
        self.ctx = type("C", (), {"reset": lambda self: None, "log": [], "size": lambda self, s: 0})()
        self.cfg = type("G", (), {"context_window": 1000})()

    def run(self, text: str) -> str:
        self._entered.set()
        self._release.wait(timeout=2)
        self.calls.append(text)
        return "ok"

    def record_event(self, text: str) -> None:
        self.events.append(text)

    def compact_now(self, force: bool = True):
        return None


class _FakePrinter:
    printed_content = False
    panels = type("P", (), {"stop_all": lambda self, reason="中断": None})()


def test_completion_is_queued_during_a_run(tmp_path: Path) -> None:
    agent = _FakeAgent()
    session = InteractiveSession(
        agent, store=None, session_id="s", registry=None,
        printer=_FakePrinter(), color=False, jobs=None,
    )
    thread = threading.Thread(target=session._dispatch_loop, daemon=True)
    thread.start()
    try:
        session.submit("user", "先做这件事")
        assert agent._entered.wait(timeout=2)
        session.submit(
            "task_done",
            TaskCompletion(
                task_id="task_01",
                command="echo x",
                cwd=str(tmp_path),
                exit_code=0,
                elapsed=0.1,
                stdout="x",
                stderr="",
                message="[后台任务完成通知]\ntask_id=task_01  exit=0",
            ),
        )
        session.submit("user", "再做下一件")
        agent._release.set()
        deadline = time.monotonic() + 3
        while (len(agent.calls) < 2 or len(agent.events) < 1) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert agent.calls == ["先做这件事", "再做下一件"]
        assert any("后台任务完成通知" in e for e in agent.events)
        assert all("后台任务完成通知" not in c for c in agent.calls)
    finally:
        session.submit("eof", None)
        thread.join(timeout=2)


def test_task_notice_has_only_one_consumer(tmp_path: Path) -> None:
    jobs = TaskRegistry()
    seen: list[TaskCompletion] = []
    jobs.on_complete(seen.append)
    task = jobs.add_task(
        lambda _task: "task-result",
        command="demo",
        cwd=tmp_path,
        owner_session_id="owner-a",
        delivery="wake",
    )
    assert task.done.wait(timeout=2)
    assert len(seen) == 1

    # get 是无副作用读取，不能抢走自动通知。
    assert "task-result" in jobs.task_manage(
        "get", task.task_id, owner_session_id="owner-a"
    )
    assert task.notification_state == "pending"

    barrier = threading.Barrier(5)
    winners: list[TaskCompletion | None] = []

    def compete() -> None:
        barrier.wait()
        winners.append(jobs.consume_notification(seen[0]))

    threads = [threading.Thread(target=compete) for _ in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)
    assert sum(item is not None for item in winners) == 1
    assert jobs.consume_notification(seen[0]) is None
    # 消费通知不会删除任务结果，之后仍可显式读取。
    assert "task-result" in jobs.task_manage("output", task.task_id)


@pytest.mark.parametrize("action", ["output", "wait", "cancel"])
def test_terminal_task_controls_consume_notice(action: str, tmp_path: Path) -> None:
    jobs = TaskRegistry()
    seen: list[TaskCompletion] = []
    jobs.on_complete(seen.append)
    task = jobs.add_task(lambda _task: "done", command="demo", cwd=tmp_path)
    assert task.done.wait(timeout=2)
    assert len(seen) == 1
    report = jobs.task_manage(action, task.task_id, timeout=2)
    assert "done" in report
    assert task.notification_state == "consumed"
    assert jobs.consume_notification(seen[0]) is None


def test_task_wait_timeout_does_not_cancel(tmp_path: Path) -> None:
    jobs = TaskRegistry()
    release = threading.Event()
    task = jobs.add_task(
        lambda _task: release.wait(timeout=5) and "finished",
        command="slow",
        cwd=tmp_path,
    )
    try:
        report = jobs.task_manage("wait", task.task_id, timeout=1)
        assert "任务未被取消" in report
        assert task.running()
    finally:
        release.set()
    assert task.done.wait(timeout=2)
    assert "finished" in jobs.task_manage("output", task.task_id)


def test_subagent_admission_is_per_owner_and_released(tmp_path: Path) -> None:
    jobs = TaskRegistry()
    release = threading.Event()

    def runner(_task) -> str:
        release.wait(timeout=5)
        return "ok"

    first = jobs.add_task(
        runner, command="a1", cwd=tmp_path,
        owner_session_id="a", max_active=1,
    )
    with pytest.raises(TaskLimitError):
        jobs.add_task(
            runner, command="a2", cwd=tmp_path,
            owner_session_id="a", max_active=1,
        )
    other = jobs.add_task(
        runner, command="b1", cwd=tmp_path,
        owner_session_id="b", max_active=1,
    )
    release.set()
    assert first.done.wait(timeout=2)
    assert other.done.wait(timeout=2)
    # 终态释放名额，同 owner 可以再次启动。
    third = jobs.add_task(
        lambda _task: "new", command="a3", cwd=tmp_path,
        owner_session_id="a", max_active=1,
    )
    assert third.done.wait(timeout=2)


def test_cancel_running_task_consumes_queued_notice(tmp_path: Path) -> None:
    jobs = TaskRegistry()
    seen: list[TaskCompletion] = []
    jobs.on_complete(seen.append)
    started = threading.Event()

    def runner(task) -> str:
        started.set()
        task.cancel_event.wait(timeout=5)
        return "stopped"

    task = jobs.add_task(runner, command="cancel-me", cwd=tmp_path, delivery="wake")
    assert started.wait(timeout=2)
    report = jobs.task_manage("cancel", task.task_id)
    assert "cancelled" in report
    assert not task.running()
    deadline = time.monotonic() + 2
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen
    assert jobs.consume_notification(seen[0]) is None


def test_idle_subagent_completion_wakes_agent_once(tmp_path: Path) -> None:
    jobs = TaskRegistry()
    seen: list[TaskCompletion] = []
    jobs.on_complete(seen.append)
    agent = _FakeAgent()
    agent._release.set()
    session = InteractiveSession(
        agent, store=None, session_id="s", registry=None,
        printer=_FakePrinter(), color=False, jobs=jobs,
    )
    task = jobs.add_task(
        lambda _task: "idle-result", command="idle", cwd=tmp_path,
        kind="subagent", delivery="wake",
    )
    assert task.done.wait(timeout=2)
    assert len(seen) == 1
    session._handle_completion(seen[0])
    session._handle_completion(seen[0])  # 重复队列事件必须 no-op
    assert len(agent.calls) == 1
    assert "idle-result" in agent.calls[0]
    assert agent.events == []


def test_completed_task_result_and_notice_resume(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "sessions.db")
    jobs = TaskRegistry(task_store=store, owner_session_id="s")
    task = jobs.add_task(
        lambda _task: "durable-result",
        command="persist-me",
        cwd=tmp_path,
        owner_session_id="s",
        delivery="wake",
    )
    assert task.done.wait(timeout=2)

    resumed = TaskRegistry(task_store=store, owner_session_id="s")
    restored = resumed.get(task.task_id)
    assert restored is not None
    assert not restored.running()
    assert restored.status == "completed"
    assert "durable-result" in resumed.task_manage("get", task.task_id)

    seen: list[TaskCompletion] = []
    duplicate: list[TaskCompletion] = []
    resumed.on_complete(seen.append)
    resumed.on_complete(duplicate.append)
    assert len(seen) == len(duplicate) == 1
    assert seen[0].notice_id == duplicate[0].notice_id == task.notice_id
    assert resumed.consume_notification(seen[0]) is not None
    assert resumed.consume_notification(duplicate[0]) is None

    # consumed 也持久化；再次重启不会重复唤醒，任务 ID 也不会复用。
    restarted = TaskRegistry(task_store=store, owner_session_id="s")
    replayed: list[TaskCompletion] = []
    restarted.on_complete(replayed.append)
    assert replayed == []
    new_task = restarted.add_task(
        lambda _task: "next",
        command="next",
        cwd=tmp_path,
        owner_session_id="s",
        delivery="wake",
    )
    assert new_task.task_id == "task_02"
    assert new_task.done.wait(timeout=2)


@pytest.mark.parametrize("kind", ["shell", "subagent"])
def test_running_snapshot_recovers_as_lost(kind: str, tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "sessions.db")
    # 直接构造崩溃时可能留下的最后快照：进程已消失，来不及写终态。
    store.save_task({
        "task_id": "task_07",
        "command": "unfinished",
        "cwd": str(tmp_path),
        "kind": kind,
        "owner_session_id": "s",
        "delivery": "wake",
        "notify_on_complete": True,
        "notification_state": "none",
        "notice_id": "task_07:complete",
        "status": "running",
        "stdout": "partial output",
        "stderr": "",
        "elapsed": 12.5,
    })

    resumed = TaskRegistry(task_store=store, owner_session_id="s")
    restored = resumed.get("task_07")
    assert restored is not None
    assert restored.status == "lost"
    assert not restored.running()
    assert restored.proc is None
    report = resumed.task_manage("get", "task_07")
    assert "lost" in report
    assert "未自动重跑" in report
    assert "partial output" in report
    assert store.load_tasks()[0]["status"] == "lost"

    seen: list[TaskCompletion] = []
    resumed.on_complete(seen.append)
    assert len(seen) == 1
    assert seen[0].error.endswith("未自动重跑")


def test_recovered_notice_is_limited_to_owner_session(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "sessions.db")
    jobs = TaskRegistry(task_store=store, owner_session_id="owner-a")
    task = jobs.add_task(
        lambda _task: "private-result",
        command="private",
        cwd=tmp_path,
        owner_session_id="owner-a",
        delivery="wake",
    )
    assert task.done.wait(timeout=2)

    other = TaskRegistry(task_store=store, owner_session_id="owner-b")
    other_seen: list[TaskCompletion] = []
    other.on_complete(other_seen.append)
    assert other_seen == []
    assert "找不到任务" in other.task_manage(
        "get", task.task_id, owner_session_id="owner-b"
    )

    owner = TaskRegistry(task_store=store, owner_session_id="owner-a")
    owner_seen: list[TaskCompletion] = []
    owner.on_complete(owner_seen.append)
    assert len(owner_seen) == 1
    assert owner_seen[0].result == "private-result"
