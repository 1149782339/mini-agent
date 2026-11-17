"""Shell 执行工具：继承当前进程的运行环境（已去掉密钥），支持超时和进程树清理。

设计取舍：
- 继承 PATH、HOME 等运行变量，但剥离 MINI_AGENT_API_KEY / ARK_API_KEY 等密钥；
- Windows 上优先用 Git Bash（`bash.exe -lc`，自带 grep/sed/awk/find 等），
  找不到 Git Bash 才退回 cmd.exe；POSIX 上用 `sh -c`；
- 命令大体不受限（能跑编译、测试、git），但 cwd 必须落在 workdir 内；
- 直接调用 rm/mv（含 /bin/rm、/usr/bin/mv）会被拒绝，请改用 RM/MV 工具。
  这是安全带，拦不住 python / find -delete / git rm 等绕法；
- 超时杀整棵进程树：POSIX 杀进程组；Windows 优先用 Job Object，
  taskkill 作为兜底，避免子进程变成孤儿；
- stdin 关闭，杜绝交互式命令把主循环挂死。
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from .base import FnTool, decode_output
from ..envutil import sanitize_subprocess_env
from ..store import project_data_dir

DEFAULT_TIMEOUT = 60
HARD_TIMEOUT = 300
MAX_BUFFER = 40000  # 单流缓冲上限，超出由注册表再截断/落盘
KILL_WAIT = 5        # 杀掉进程树后等待回收输出的秒数


def resolve_shell_name(shell: str = "auto") -> str:
    """启动横幅用：不构造 ShellTools，只解析实际会用的 shell 名称。"""
    if os.name != "nt":
        return "sh"
    if (shell or "auto").strip().lower() == "cmd":
        return "cmd"
    bash = find_git_bash()
    return f"gitbash ({bash})" if bash else "cmd"


def find_git_bash() -> Path | None:
    """定位 Git for Windows 的 bash.exe。

    注意不是 git-bash.exe：那个是打开新终端窗口的启动器，不能用于非交互执行。
    解析顺序：MINI_AGENT_BASH / GIT_BASH 环境变量 → 从 PATH 上的 git.exe 推导
    （git\\..\\..\\bin\\bash.exe）→ 扫常见安装位置。
    """
    explicit = os.environ.get("MINI_AGENT_BASH") or os.environ.get("GIT_BASH")
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.is_file() else None

    # 跟着 PATH 上的 git.exe 推导 bash.exe：官方安装布局（cmd\bin\git.exe）
    # 下都成立，绿色安装/自定义路径也能找到。
    git = shutil.which("git")
    if git:
        candidate = (Path(git) / ".." / ".." / "bin" / "bash.exe").resolve()
        if candidate.is_file():
            return candidate

    roots = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("LOCALAPPDATA"),
    ]
    suffixes = (r"Git\bin\bash.exe", r"Programs\Git\bin\bash.exe")
    for root in roots:
        if not root:
            continue
        for suffix in suffixes:
            candidate = Path(root) / suffix
            if candidate.is_file():
                return candidate
    return None


# 模型偶发 cmd 习惯的 `2>nul`：Git Bash 会创建字面文件 nul（Windows 保留设备名，
# 极难删除且搞坏 git add/clone）。统一改写成 /dev/null。匹配 `>nul` `2>nul`
# `&>nul` `>>nul`（大小写不敏感），不匹配 `>null` `>nul.txt`；不解析引号，
# `echo ">nul"` 会被误改但无害。
_NUL_REDIRECT_RE = re.compile(r"(\d?&?>+\s*)[Nn][Uu][Ll](?=\s|$|[|&;)\n])")


def rewrite_windows_null_redirect(command: str) -> str:
    """把 cmd 风格的 `>nul` 重定向改写成 POSIX 的 `>/dev/null`。"""
    return _NUL_REDIRECT_RE.sub(r"\1/dev/null", command)


# 直接调用 rm/mv 的安全带（不是锁）：只拦命令位置上的 rm/mv 及其绝对路径，
# 不拦 git rm、find -delete、python os.remove、unlink，也不解析被引号包住的文本。
_BLOCKED_BINARIES = {"rm", "mv"}
_SHELL_SPLIT_RE = re.compile(r"(?:&&|\|\||[;|&\n`]|\$\()")
_WRAPPERS = {"command", "exec", "time", "nohup", "nice", "env", "sudo"}
_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _mask_quoted(command: str) -> str:
    """把引号内的内容换成空格，避免 echo 'rm file' 被误判为调用 rm。"""
    out: list[str] = []
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch == "'":
            j = i + 1
            while j < n and command[j] != "'":
                j += 1
            out.append(" " * (min(j, n - 1) - i + 1))
            i = j + 1
            continue
        if ch == '"':
            j = i + 1
            while j < n:
                if command[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if command[j] == '"':
                    break
                j += 1
            out.append(" " * (min(j, n - 1) - i + 1))
            i = j + 1
            continue
        if ch == "\\" and i + 1 < n:
            out.append("  ")
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _command_name(token: str) -> str:
    name = token.strip().strip("'\"")
    name = os.path.basename(name.replace("\\", "/").rstrip("/")).lstrip("\\")
    if os.name == "nt" and name.lower().endswith(".exe"):
        name = name[:-4]
        return name.lower()
    return name


def direct_rm_mv_invocation(command: str) -> str | None:
    """若命令在命令位置直接调用 rm/mv（含 /bin/rm、/usr/bin/mv），返回该词；否则 None。"""
    masked = _mask_quoted(command or "")
    for part in _SHELL_SPLIT_RE.split(masked):
        tokens = part.split()
        i = 0
        while i < len(tokens) and _ASSIGN_RE.match(tokens[i]):
            i += 1
        while i < len(tokens) and _command_name(tokens[i]).lower() in _WRAPPERS:
            i += 1
            while i < len(tokens) and tokens[i].startswith("-"):
                i += 1
        if i >= len(tokens):
            continue
        name = _command_name(tokens[i])
        key = name.lower() if os.name == "nt" else name
        if key in _BLOCKED_BINARIES:
            return name
    return None


def _rm_mv_blocked_message(name: str) -> str:
    tool = name.lower()
    if tool == "rm":
        return (
            "ERROR: 禁止在 shell 里直接执行 rm（包括 /bin/rm、/usr/bin/rm）。"
            "请改用 RM 工具：路径必须是工作目录内的字面绝对路径；"
            "删除目录需显式 recursive=true。"
        )
    return (
        "ERROR: 禁止在 shell 里直接执行 mv（包括 /bin/mv、/usr/bin/mv）。"
        "请改用 MV 工具：路径必须是工作目录内的字面绝对路径；"
        "目标已存在时拒绝覆盖。"
    )


if os.name == "nt":  # pragma: no cover - 仅 Windows
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KILL_ON_JOB_CLOSE = 0x2000
    _JOB_EXTENDED = 9
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    class _BasicLimit(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _ExtendedLimit(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimit),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class WindowsJob:
        """把命令的整棵进程树收进 Job Object，超时一次性杀掉。

        taskkill 在受限环境（AppContainer 沙箱）里会被拒绝（错误 5），
        Job Object 是内核机制，不需要额外的进程权限，因此更可靠。
        句柄带 KILL_ON_JOB_CLOSE，关闭时残留子孙进程也会被清理。
        """

        def __init__(self) -> None:
            self._job: int | None = None
            self._proc_handle: int | None = None

        @property
        def active(self) -> bool:
            return self._job is not None

        def attach(self, proc: subprocess.Popen) -> None:
            try:
                job = _k32.CreateJobObjectW(None, None)
                if not job:
                    return
                info = _ExtendedLimit()
                info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
                if not _k32.SetInformationJobObject(
                    job, _JOB_EXTENDED, ctypes.byref(info), ctypes.sizeof(info)
                ):
                    _k32.CloseHandle(job)
                    return
                handle = _k32.OpenProcess(
                    _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, proc.pid
                )
                if not handle or not _k32.AssignProcessToJobObject(job, handle):
                    if handle:
                        _k32.CloseHandle(handle)
                    _k32.CloseHandle(job)
                    return
                self._job, self._proc_handle = job, handle
            except OSError:
                self._job = self._proc_handle = None

        def kill(self) -> None:
            if self._job:
                _k32.TerminateJobObject(self._job, 1)

        def close(self) -> None:
            for handle in (self._proc_handle, self._job):
                if handle:
                    _k32.CloseHandle(handle)
            self._job = self._proc_handle = None

else:
    WindowsJob = None


# 正在跑的 shell：中断时按这张表杀整棵进程树，不留下半截命令。
_ACTIVE_SHELLS: dict[int, tuple[subprocess.Popen, object | None]] = {}
_SHELL_GUARD = threading.Lock()


def register_shell(proc: subprocess.Popen, job: object | None) -> None:
    with _SHELL_GUARD:
        _ACTIVE_SHELLS[id(proc)] = (proc, job)


def unregister_shell(proc: subprocess.Popen) -> None:
    with _SHELL_GUARD:
        _ACTIVE_SHELLS.pop(id(proc), None)


def terminate_active_shells() -> None:
    """杀掉所有还在跑的 run_command 进程树。"""
    with _SHELL_GUARD:
        items = list(_ACTIVE_SHELLS.values())
    for proc, job in items:
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


def kill_process_tree(proc: subprocess.Popen) -> None:
    """兜底清理：POSIX 杀进程组；Windows 用 taskkill 再退化为 kill 子进程。

    Windows 正常路径是 Job Object（见 WindowsJob）；这里只在 Job 创建或
    绑定失败时使用，因此不保证一定连孙进程一起清掉。
    """
    if os.name == "nt":
        # taskkill /T 连带子孙进程，/F 强制；CREATE_NO_WINDOW 避免闪黑框
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        try:
            proc.kill()
        except OSError:
            pass


class ShellTools:
    def __init__(
        self,
        workdir: Path,
        default_timeout: int = DEFAULT_TIMEOUT,
        hard_timeout: int = HARD_TIMEOUT,
        env: dict[str, str] | None = None,
        shell: str = "auto",
        jobs=None,
    ):
        self.workdir = workdir.resolve()
        self.default_timeout = default_timeout
        self.hard_timeout = hard_timeout
        # 延迟导入，避免与 jobs 循环引用
        if jobs is None:
            from ..jobs import JobRegistry
            jobs = JobRegistry()
        self.jobs = jobs
        # 拷贝运行环境，但去掉 API key 等密钥，避免命令直接读到凭证
        self.env = sanitize_subprocess_env(os.environ if env is None else env)
        # 统一 UTF-8：让子进程（尤其 python）也按 UTF-8 输出，解码端不用猜编码
        self.env["PYTHONIOENCODING"] = "utf-8"
        self.env["PYTHONUTF8"] = "1"

        self.shell = (shell or "auto").strip().lower()
        if self.shell in {"bash", "gitbash", "git-bash"}:
            self.shell = "gitbash"
        if self.shell not in {"auto", "gitbash", "cmd", "posix"}:
            raise RuntimeError(f"未知 shell 配置 {shell!r}，可用：auto / gitbash / cmd")

        self.bash: Path | None = None
        if self.shell in {"auto", "gitbash"} and os.name == "nt":
            self.bash = find_git_bash()
            if self.bash is None and self.shell == "gitbash":
                raise RuntimeError(
                    "没有找到 Git Bash。请安装 Git for Windows，"
                    "或用 MINI_AGENT_BASH 指定 bash.exe 路径，或设 MINI_AGENT_SHELL=cmd"
                )

    @property
    def shell_name(self) -> str:
        """简短名，出现在 run_command 的输出头里。"""
        if self.bash is not None:
            return "gitbash"
        return "cmd" if os.name == "nt" else "sh"

    # ---------- 内部 ----------
    def _resolve_cwd(self, cwd: str | None) -> Path:
        if not cwd:
            return self.workdir
        raw = Path(cwd).expanduser()
        target = (raw if raw.is_absolute() else self.workdir / raw).resolve()
        try:
            target.relative_to(self.workdir)
        except ValueError:
            raise PermissionError(f"cwd 越界：{target} 不在 {self.workdir} 内") from None
        return target

    def _clip(self, text: str, tag: str = "out") -> str:
        """超长输出：完整内容落盘 spill/，上下文里只保留末尾。

        保尾不保头——错误信息几乎总在输出末尾，旧版保头会把关键报错剪掉。
        """
        if len(text) <= MAX_BUFFER:
            return text
        try:
            spill_dir = project_data_dir() / "spill"
            spill_dir.mkdir(parents=True, exist_ok=True)
            path = spill_dir / f"bash_{tag}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.txt"
            path.write_text(text, encoding="utf-8")
            note = (
                f"[完整输出 {len(text)} 字符已保存到 {path}，"
                f"以下为末尾 {MAX_BUFFER} 字符]\n"
            )
        except OSError:
            note = f"[输出共 {len(text)} 字符，以下为末尾 {MAX_BUFFER} 字符]\n"
        return note + text[-MAX_BUFFER:]

    # ---------- 工具实现 ----------
    def run_command(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        background: bool | str = False,
        notify_on_complete: bool | str = False,
    ) -> str:
        from .base import semantic_bool
        command = (command or "").strip()
        if not command:
            return "ERROR: command 不能为空"
        blocked = direct_rm_mv_invocation(command)
        if blocked:
            return _rm_mv_blocked_message(blocked)
        command = rewrite_windows_null_redirect(command)
        background = semantic_bool(background, False)
        notify_requested = semantic_bool(notify_on_complete, False)

        try:
            workdir = self._resolve_cwd(cwd)
        except PermissionError as exc:
            return f"ERROR: {exc}"
        if not workdir.exists():
            return f"ERROR: cwd 不存在 {workdir}"

        proc, job_obj = self._spawn(command, workdir)
        job = self.jobs.add_running(
            proc,
            command=command,
            cwd=workdir,
            win_job=job_obj,
            notify_on_complete=notify_requested if background else False,
        )
        if background:
            return self.jobs.format_started(job, notify_requested=notify_requested)

        # 前台：仍受 timeout / hard_timeout 约束，但超时后交互模式升级为后台。
        limit = int(timeout) if timeout else self.default_timeout
        limit = max(1, min(limit, self.hard_timeout))
        register_shell(proc, job_obj)
        started = time.monotonic()
        try:
            finished = job.done.wait(timeout=limit)
        except KeyboardInterrupt:
            if job_obj is not None and getattr(job_obj, "active", False):
                job_obj.kill()
            else:
                kill_process_tree(proc)
            raise
        finally:
            unregister_shell(proc)

        if finished:
            out, err = job.snapshot()
            elapsed = time.monotonic() - started
            parts = [
                f"exit={job.exit_code}  time={elapsed:.2f}s  cwd={workdir}  shell={self.shell_name}"
            ]
            if out.strip():
                parts.append("--- stdout ---\n" + self._clip(out.rstrip(), "stdout"))
            if err.strip():
                parts.append("--- stderr ---\n" + self._clip(err.rstrip(), "stderr"))
            if not out.strip() and not err.strip():
                parts.append("(无输出)")
            if job_obj is not None:
                job_obj.close()
                job.win_job = None
            return "\n".join(parts)

        # 超时
        if self.jobs.allow_notify:
            job.notify_on_complete = True
            job.upgraded_from_timeout = True
            return self.jobs.format_started(job, notify_requested=True, upgraded=True)

        self.jobs.manage("kill", job.job_id)
        out, err = job.snapshot()
        elapsed = time.monotonic() - started
        parts = [
            f"exit={job.exit_code}  time={elapsed:.2f}s  cwd={workdir}  shell={self.shell_name}"
            "  [TIMEOUT，已杀进程树]"
        ]
        if out.strip():
            parts.append("--- stdout ---\n" + self._clip(out.rstrip(), "stdout"))
        if err.strip():
            parts.append("--- stderr ---\n" + self._clip(err.rstrip(), "stderr"))
        if not out.strip() and not err.strip():
            parts.append("(无输出)")
        return "\n".join(parts)

    def process_manage(
        self,
        action: str,
        job_id: str | None = None,
        timeout: int | None = None,
    ) -> str:
        return self.jobs.manage(action, job_id=job_id, timeout=timeout)

    def _spawn(self, command: str, workdir: Path):
        """启动命令：不通信、不关闭，调用方负责等待或交给注册表。"""
        if self.bash is not None:
            # Git Bash：-l 载入 profile（拿到 /usr/bin 的 grep/sed/awk），-c 执行命令
            argv: list[str] | str = [str(self.bash), "-lc", command]
            use_shell = False
        else:
            argv = command
            use_shell = True
        proc = subprocess.Popen(
            argv,
            shell=use_shell,
            cwd=str(workdir),
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            # 子进程不新建控制台窗口
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        job_obj = WindowsJob() if WindowsJob is not None else None
        if job_obj is not None:
            job_obj.attach(proc)
        return proc, job_obj

    # ---------- 注册 ----------

    def tools(self) -> list[FnTool]:
        return [
            FnTool(
                name="run_command",
                description=(
                    "执行 shell 命令并返回 exit code、耗时、stdout、stderr。"
                    "Windows 上默认跑在 Git Bash（标准 bash 语法：$VAR、; 分隔、$(...)、"
                    "管道和重定向齐全，grep/sed/awk/find 都可用）；找不到 Git Bash 时退回 cmd.exe。"
                    "Linux/macOS 上是 sh。继承 PATH/HOME 等运行环境，但不传递 API key 等密钥。"
                    "用于跑测试、编译、安装依赖、查看进程。非交互式，stdin 已关闭，"
                    "不要用它跑 vim/top 之类交互程序。"
                    "长时间任务（下载、编译）请设 background=true，立刻返回 job_id，不要同步死等；"
                    "需要做完后自动继续时再加 notify_on_complete=true。"
                    "前台命令触达超时时，交互模式下会升级为后台并打开完成通知，而不是杀掉。"
                    "不要在这里直接执行 rm 或 mv（包括 /bin/rm、/usr/bin/mv），请改用 RM 或 MV 工具；"
                    "路径必须是工作目录内的字面绝对路径。这是安全带，不拦 python / find -delete / git rm 等绕法。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "要执行的 shell 命令，支持管道和 &&",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "前台超时秒数，默认 60，最大 300。后台任务不受此硬杀限制。",
                            "default": DEFAULT_TIMEOUT,
                        },
                        "cwd": {
                            "type": "string",
                            "description": "工作目录（相对或绝对，必须在工作根内），默认工作根",
                        },
                        "background": {
                            "type": "boolean",
                            "description": "true 时立即返回 job_id，进程在后台继续跑，不受 300 秒硬杀限制",
                            "default": False,
                        },
                        "notify_on_complete": {
                            "type": "boolean",
                            "description": "true 时任务结束后用合成消息叫醒交互式主代理。-p 等非交互模式会忽略此开关。",
                            "default": False,
                        },
                    },
                    "required": ["command"],
                },
                fn=self.run_command,
            ),
            FnTool(
                name="process_manage",
                description=(
                    "管理后台命令：list 列出全部任务；poll 查看状态和新输出；"
                    "log 读取已落盘的完整输出（上下文里仍会截断）；"
                    "wait 等待结束或直到 timeout（默认 30 秒，最大 300）；"
                    "kill 终止进程树。若已打开完成通知，不要反复 poll。"
                    "同一任务连续两次 poll 或 wait 且没有新输出时，会自动改为完成通知。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["list", "poll", "log", "wait", "kill"],
                            "description": "list / poll / log / wait / kill",
                        },
                        "job_id": {
                            "type": "string",
                            "description": "list 以外的操作需要任务 id，例如 job_01",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "仅 wait 使用，最长等待秒数，默认 30，最大 300",
                        },
                    },
                    "required": ["action"],
                },
                fn=self.process_manage,
            ),
        ]
