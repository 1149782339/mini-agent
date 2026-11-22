"""后台命令、完成通知、process_manage 与超时升级。"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from mini_agent.cli import InteractiveSession
from mini_agent.jobs import JobCompletion, JobRegistry, TaskLimitError, TaskRegistry
from mini_agent.tools.shell import ShellTools


def test_background_returns_immediately(tmp_path: Path) -> None:
    jobs = JobRegistry(allow_notify=True)
    shell = ShellTools(tmp_path, default_timeout=10, jobs=jobs)
    started = time.monotonic()
    out = shell.run_command("sleep 2; echo done-bg", background=True)
    elapsed = time.monotonic() - started
    try:
        assert "job_01" in out
        assert "已在后台启动" in out
        assert elapsed < 1.2
        listed = shell.process_manage("list")
        assert "job_01" in listed
        assert "running" in listed
    finally:
        jobs.cleanup_all()


def test_notify_fires_on_complete(tmp_path: Path) -> None:
    jobs = JobRegistry(allow_notify=True)
    seen: list[JobCompletion] = []
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
        assert seen[0].job_id == "job_01"
    finally:
        jobs.cleanup_all()


def test_process_manage_wait_and_kill(tmp_path: Path) -> None:
    jobs = JobRegistry(allow_notify=True)
    shell = ShellTools(tmp_path, default_timeout=10, jobs=jobs)
    try:
        shell.run_command("echo wait-me", background=True)
        waited = shell.process_manage("wait", job_id="job_01", timeout=5)
        assert "exit=0" in waited
        assert "wait-me" in waited
        log = shell.process_manage("log", job_id="job_01")
        assert "wait-me" in log

        shell.run_command("sleep 30", background=True)
        killed = shell.process_manage("kill", job_id="job_02")
        assert "已发出终止" in killed or "exit=" in killed
        # 给回收一点时间
        deadline = time.monotonic() + 3
        while jobs.get("job_02") and jobs.get("job_02").running() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert jobs.get("job_02") is not None
        assert not jobs.get("job_02").running()
    finally:
        jobs.cleanup_all()


def test_stale_polls_enable_notify(tmp_path: Path) -> None:
    jobs = JobRegistry(allow_notify=True)
    shell = ShellTools(tmp_path, default_timeout=10, jobs=jobs)
    try:
        shell.run_command("sleep 8", background=True)
        first = shell.process_manage("poll", job_id="job_01")
        second = shell.process_manage("poll", job_id="job_01")
        third = shell.process_manage("poll", job_id="job_01")
        assert "改为完成通知" not in first
        assert "改为完成通知" not in second
        assert "改为完成通知" in third
        assert jobs.get("job_01").notify_on_complete
    finally:
        jobs.cleanup_all()


def test_timeout_upgrades_to_background(tmp_path: Path) -> None:
    work = tmp_path / "upgrade"
    work.mkdir()
    marker = work / "marker.txt"
    jobs = JobRegistry(allow_notify=True)
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
        assert "job_01" in out
        assert elapsed < 3
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert marker.exists(), "超时后进程被杀掉了，标记文件没有出现"
    finally:
        jobs.cleanup_all()


def test_timeout_still_kills_when_notify_disabled(tmp_path: Path) -> None:
    work = tmp_path / "timeout_check"
    work.mkdir()
    marker = work / "marker.txt"
    jobs = JobRegistry(allow_notify=False)
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
    jobs = JobRegistry()
    shell = ShellTools(tmp_path, jobs=jobs)
    out = shell.run_command("rm -rf /tmp/whatever", background=True)
    assert out.startswith("ERROR:")
    assert "RM 工具" in out
    assert jobs.all_jobs() == []


def test_print_mode_disables_notify(tmp_path: Path) -> None:
    jobs = JobRegistry(allow_notify=False)
    seen: list[JobCompletion] = []
    jobs.on_complete(seen.append)
    shell = ShellTools(tmp_path, default_timeout=10, jobs=jobs)
    out = shell.run_command("echo no-notify", background=True, notify_on_complete=True)
    assert "完成通知已关闭" in out or "非交互" in out
    waited = shell.process_manage("wait", job_id="job_01", timeout=5)
    assert "exit=0" in waited
    time.sleep(0.2)
    assert seen == []
    jobs.cleanup_all()


def test_cleanup_all_kills_running_jobs(tmp_path: Path) -> None:
    jobs = JobRegistry()
    shell = ShellTools(tmp_path, default_timeout=10, jobs=jobs)
    shell.run_command("sleep 30", background=True)
    assert jobs.running_jobs()
    killed = jobs.cleanup_all()
    assert killed == 1
    deadline = time.monotonic() + 3
    while jobs.running_jobs() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not jobs.running_jobs()


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
            "job_done",
            JobCompletion(
                job_id="job_01",
                command="echo x",
                cwd=str(tmp_path),
                exit_code=0,
                elapsed=0.1,
                stdout="x",
                stderr="",
                message="[后台任务完成通知]\njob_id=job_01  exit=0",
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
    seen: list[JobCompletion] = []
    jobs.on_complete(seen.append)
    job = jobs.add_task(
        lambda _job: "task-result",
        command="demo",
        cwd=tmp_path,
        owner_session_id="owner-a",
        delivery="wake",
    )
    assert job.done.wait(timeout=2)
    assert len(seen) == 1

    # get 是无副作用读取，不能抢走自动通知。
    assert "task-result" in jobs.task_manage(
        "get", job.job_id, owner_session_id="owner-a"
    )
    assert job.notification_state == "pending"

    barrier = threading.Barrier(5)
    winners: list[JobCompletion | None] = []

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
    assert "task-result" in jobs.task_manage("output", job.job_id)


@pytest.mark.parametrize("action", ["output", "wait", "cancel"])
def test_terminal_task_controls_consume_notice(action: str, tmp_path: Path) -> None:
    jobs = TaskRegistry()
    seen: list[JobCompletion] = []
    jobs.on_complete(seen.append)
    job = jobs.add_task(lambda _job: "done", command="demo", cwd=tmp_path)
    assert job.done.wait(timeout=2)
    assert len(seen) == 1
    report = jobs.task_manage(action, job.job_id, timeout=2)
    assert "done" in report
    assert job.notification_state == "consumed"
    assert jobs.consume_notification(seen[0]) is None


def test_task_wait_timeout_does_not_cancel(tmp_path: Path) -> None:
    jobs = TaskRegistry()
    release = threading.Event()
    job = jobs.add_task(
        lambda _job: release.wait(timeout=5) and "finished",
        command="slow",
        cwd=tmp_path,
    )
    try:
        report = jobs.task_manage("wait", job.job_id, timeout=1)
        assert "任务未被取消" in report
        assert job.running()
    finally:
        release.set()
    assert job.done.wait(timeout=2)
    assert "finished" in jobs.task_manage("output", job.job_id)


def test_subagent_admission_is_per_owner_and_released(tmp_path: Path) -> None:
    jobs = TaskRegistry()
    release = threading.Event()

    def runner(_job) -> str:
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
        lambda _job: "new", command="a3", cwd=tmp_path,
        owner_session_id="a", max_active=1,
    )
    assert third.done.wait(timeout=2)


def test_cancel_running_task_consumes_queued_notice(tmp_path: Path) -> None:
    jobs = TaskRegistry()
    seen: list[JobCompletion] = []
    jobs.on_complete(seen.append)
    started = threading.Event()

    def runner(job) -> str:
        started.set()
        job.cancel_event.wait(timeout=5)
        return "stopped"

    job = jobs.add_task(runner, command="cancel-me", cwd=tmp_path, delivery="wake")
    assert started.wait(timeout=2)
    report = jobs.task_manage("cancel", job.job_id)
    assert "cancelled" in report
    assert not job.running()
    deadline = time.monotonic() + 2
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen
    assert jobs.consume_notification(seen[0]) is None


def test_idle_subagent_completion_wakes_agent_once(tmp_path: Path) -> None:
    jobs = TaskRegistry()
    seen: list[JobCompletion] = []
    jobs.on_complete(seen.append)
    agent = _FakeAgent()
    agent._release.set()
    session = InteractiveSession(
        agent, store=None, session_id="s", registry=None,
        printer=_FakePrinter(), color=False, jobs=jobs,
    )
    job = jobs.add_task(
        lambda _job: "idle-result", command="idle", cwd=tmp_path,
        kind="subagent", delivery="wake",
    )
    assert job.done.wait(timeout=2)
    assert len(seen) == 1
    session._handle_completion(seen[0])
    session._handle_completion(seen[0])  # 重复队列事件必须 no-op
    assert len(agent.calls) == 1
    assert "idle-result" in agent.calls[0]
    assert agent.events == []
