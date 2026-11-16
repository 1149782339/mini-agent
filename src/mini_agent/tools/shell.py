"""Shell 执行工具：继承当前进程的完整环境变量，支持超时和进程树清理。

设计取舍：
- 继承 os.environ（用户要求），因此 agent 的子进程能看到 PATH、HOME、API key 等；
- Windows 上优先用 Git Bash（`bash.exe -lc`，自带 grep/sed/awk/find 等），
  找不到 Git Bash 才退回 cmd.exe；POSIX 上用 `sh -c`；
- 命令不受限（能跑编译、测试、git），但 cwd 必须落在 workdir 内；
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
    # 下都成立，绿色安装/自定义路径也能找到。借自 Claude Code 的 findGitBashPath
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
# 极难删除且搞坏 git add/clone，见 anthropics/claude-code#4928）。统一改写成
# /dev/null。正则借自 Claude Code：匹配 `>nul` `2>nul` `&>nul` `>>nul`（大小写
# 不敏感），不匹配 `>null` `>nul.txt`；不解析引号，`echo ">nul"` 会被误改但无害。
_NUL_REDIRECT_RE = re.compile(r"(\d?&?>+\s*)[Nn][Uu][Ll](?=\s|$|[|&;)\n])")


def rewrite_windows_null_redirect(command: str) -> str:
    """把 cmd 风格的 `>nul` 重定向改写成 POSIX 的 `>/dev/null`。"""
    return _NUL_REDIRECT_RE.sub(r"\1/dev/null", command)


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
    ):
        self.workdir = workdir.resolve()
        self.default_timeout = default_timeout
        self.hard_timeout = hard_timeout
        # 显式拷贝一份，保证子进程拿到的是完整用户环境
        self.env = dict(os.environ if env is None else env)
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
        """超长输出：完整内容落盘 spill/，上下文里只保留末尾（Pi truncateTail）。

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
    def run_command(self, command: str, timeout: int | None = None, cwd: str | None = None) -> str:
        command = (command or "").strip()
        if not command:
            return "ERROR: command 不能为空"
        command = rewrite_windows_null_redirect(command)

        limit = int(timeout) if timeout else self.default_timeout
        limit = max(1, min(limit, self.hard_timeout))
        try:
            workdir = self._resolve_cwd(cwd)
        except PermissionError as exc:
            return f"ERROR: {exc}"
        if not workdir.exists():
            return f"ERROR: cwd 不存在 {workdir}"

        started = time.monotonic()
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
            # 等价 Claude Code 的 windowsHide：子进程不新建控制台窗口
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        job = WindowsJob() if WindowsJob is not None else None
        if job is not None:
            job.attach(proc)
        register_shell(proc, job)

        timed_out = False
        try:
            try:
                out_raw, err_raw = proc.communicate(timeout=limit)
            except KeyboardInterrupt:
                if job is not None and job.active:
                    job.kill()
                else:
                    kill_process_tree(proc)
                raise
            except subprocess.TimeoutExpired:
                timed_out = True
                if job is not None and job.active:
                    job.kill()
                else:
                    kill_process_tree(proc)
                try:
                    out_raw, err_raw = proc.communicate(timeout=KILL_WAIT)
                except subprocess.TimeoutExpired as exc:
                    # 极端情况：孙进程仍占着管道，放弃等待，保留已读到的输出
                    out_raw, err_raw = exc.stdout or b"", exc.stderr or b""
        finally:
            unregister_shell(proc)
            if job is not None:
                job.close()

        out = decode_output(out_raw)
        err = decode_output(err_raw)

        elapsed = time.monotonic() - started
        code = proc.returncode
        parts = [
            f"exit={code}  time={elapsed:.2f}s  cwd={workdir}  shell={self.shell_name}"
            + ("  [TIMEOUT，已杀进程树]" if timed_out else "")
        ]
        if out.strip():
            parts.append("--- stdout ---\n" + self._clip(out.rstrip(), "stdout"))
        if err.strip():
            parts.append("--- stderr ---\n" + self._clip(err.rstrip(), "stderr"))
        if not out.strip() and not err.strip():
            parts.append("(无输出)")
        return "\n".join(parts)

    # ---------- 注册 ----------
    def tools(self) -> list[FnTool]:
        return [
            FnTool(
                name="run_command",
                description=(
                    "执行 shell 命令并返回 exit code、耗时、stdout、stderr。"
                    "Windows 上默认跑在 Git Bash（标准 bash 语法：$VAR、; 分隔、$(...)、"
                    "管道和重定向齐全，grep/sed/awk/find 都可用）；找不到 Git Bash 时退回 cmd.exe。"
                    "Linux/macOS 上是 sh。继承当前进程环境变量（PATH/HOME 等）。用于跑测试、"
                    "编译、安装依赖、查看进程。非交互式，stdin 已关闭，不要用它跑 vim/top 之类交互程序。"
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
                            "description": "超时秒数，默认 60，最大 300",
                            "default": DEFAULT_TIMEOUT,
                        },
                        "cwd": {
                            "type": "string",
                            "description": "工作目录（相对或绝对，必须在工作根内），默认工作根",
                        },
                    },
                    "required": ["command"],
                },
                fn=self.run_command,
            )
        ]
