"""文件工具共享件：读状态缓存 + 路径/编码工具。

读状态是整个文件工具族的状态机（一状态三用）：
- Read 登记读取过的内容（范围、mtime、编码、行尾）
- Edit/Write 以它为前置条件（read-before-write），并做 staleness 闸
- Read 自身用它做 file_unchanged 去重（同范围同 mtime 直接回 stub）

编码约定：磁盘字节按 BOM 检测 utf-16-le/be，其余一律 UTF-8
（errors=replace，不猜 GBK）；内存中文本统一 CRLF→LF，写盘时按原行尾还原。
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .base import ToolContext, ToolError

MAX_FILE_BYTES = 1 << 30  # Edit/Write 的防 OOM 上限（1 GiB）
MAX_OUTPUT_SIZE = 256 * 1024  # Read 整读字节帽

CACHE_MAX_ENTRIES = 100
CACHE_MAX_BYTES = 25 * 1024 * 1024


# ---------- 路径 ----------

def is_unc(raw: str) -> bool:
    """UNC 路径（\\\\server 或 //server）。

    对 UNC 路径跳过一切文件系统操作——Windows 的 stat/exists
    会触发 SMB 连接，造成 NTLM 凭据泄漏。这里只做字符串判断，零 I/O。
    """
    return raw.startswith("\\\\") or raw.startswith("//")


def resolve_in_workdir(workdir: Path, raw: str) -> Path:
    """展开 ~ / 相对路径并约束在 workdir 内（软链接经 resolve() 后同样受约束）。

    只用于写路径（Edit/Write）；读/搜索工具用 resolve_anywhere。
    """
    if not raw or not str(raw).strip():
        raise ToolError("path 不能为空", error_code=1)
    p = Path(raw).expanduser()
    target = (p if p.is_absolute() else workdir / p).resolve()
    try:
        target.relative_to(workdir)
    except ValueError:
        raise ToolError(
            f"越界：{target} 不在工作目录 {workdir} 内", error_code=2
        ) from None
    return target


def resolve_anywhere(workdir: Path, raw: str) -> Path:
    """读/搜索工具的路径解析：相对路径基于 workdir，绝对路径任意位置放行。

    刻意不做范围限制——只读操作允许看机器上任何地方（Read/Ls/Glob/Grep）。
    UNC 仍由调用方在 validate 阶段拒绝（防 NTLM 凭据泄漏，与范围无关）。
    软链接照常 resolve()，读的是真实目标；写工具仍走 resolve_in_workdir。
    """
    if not raw or not str(raw).strip():
        raise ToolError("path 不能为空", error_code=1)
    p = Path(raw).expanduser()
    return (p if p.is_absolute() else workdir / p).resolve()


def find_similar_file(missing: Path) -> Path | None:
    """ENOENT 时的 "Did you mean X?"：同目录下同名不同扩展名的文件。"""
    stem = missing.stem
    try:
        for sibling in missing.parent.iterdir():
            if sibling.is_file() and sibling.stem == stem and sibling != missing:
                return sibling
    except OSError:
        pass
    return None


# ---------- 读状态缓存 ----------

@dataclass
class FileState:
    content: str        # CRLF 归一化后的全文（Read 视图 / Edit 匹配基准）
    mtime_ns: int
    offset: int | None  # Read 登记的范围；Edit/Write 回写时置 None，
    limit: int | None   # 用于区分"模型亲眼读过的"还是"工具回写的"（去重只对前者生效）
    is_partial_view: bool = False  # partial 视图禁止 Edit/Write
    encoding: str = "utf-8"        # 磁盘编码，写回时保留
    eol: str = "\n"                # 磁盘原始行尾，写回时还原


class FileStateCache:
    """LRU + 总量上限的读状态缓存；键统一 normcase/normpath。"""

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES,
                 max_bytes: int = CACHE_MAX_BYTES):
        self._entries: OrderedDict[str, FileState] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._bytes = 0
        self._lock = threading.Lock()  # 并行派发下多个工具可能同时读写

    @staticmethod
    def _key(path: Path) -> str:
        return os.path.normcase(os.path.normpath(str(path)))

    def get(self, path: Path) -> FileState | None:
        key = self._key(path)
        with self._lock:
            state = self._entries.get(key)
            if state is not None:
                self._entries.move_to_end(key)
            return state

    def set(self, path: Path, state: FileState) -> None:
        key = self._key(path)
        encoded_len = len(state.content.encode("utf-8"))
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= len(old.content.encode("utf-8"))
            self._entries[key] = state
            self._bytes += encoded_len
            while (
                len(self._entries) > self._max_entries or self._bytes > self._max_bytes
            ) and self._entries:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= len(evicted.content.encode("utf-8"))

    def delete(self, path: Path) -> None:
        with self._lock:
            self._entries.pop(self._key(path), None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0


# ---------- 磁盘文本读写（编码 + 行尾保真） ----------

_BOMS = (
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
    (b"\xef\xbb\xbf", "utf-8"),
)


def detect_encoding(raw: bytes) -> tuple[str, bytes]:
    """返回 (编码, 去掉 BOM 后的字节)。无 BOM 一律 utf-8。

    utf-8 带 BOM 时返回 "utf-8-sig"：该编解码器解码剥 BOM、编码自动加 BOM，
    保证写回时 BOM 可还原（utf-16/32 由 write_disk_text 的 BOM 表补回）。
    """
    for bom, enc in _BOMS:
        if raw.startswith(bom):
            if enc == "utf-8":
                enc = "utf-8-sig"
            return enc, raw[len(bom):]
    return "utf-8", raw


def eol_of(text: str) -> str:
    """磁盘原始行尾：CRLF 数量超过孤 LF 的半数即视为 CRLF 文件。"""
    crlf = text.count("\r\n")
    lone_lf = text.count("\n") - crlf
    return "\r\n" if crlf > lone_lf else "\n"


def read_disk_text(path: Path) -> tuple[str, str, str]:
    """读盘 → (CRLF 归一化的文本, 编码, 原始行尾)。

    不猜 GBK：非 BOM 文件一律 UTF-8 errors=replace，解码失败替换字符，
    不按系统码页回退。
    """
    raw = path.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raise ToolError(
            f"文件过大（{len(raw)} 字节，上限 {MAX_FILE_BYTES >> 30} GiB）", error_code=3
        )
    encoding, body = detect_encoding(raw)
    text = body.decode(encoding, errors="replace")
    eol = eol_of(text)
    return text.replace("\r\n", "\n"), encoding, eol


def write_disk_text(path: Path, normalized: str, encoding: str, eol: str) -> None:
    """LF 归一化文本 → 按原编码/行尾写盘（Edit 保真；Write 传 eol="\\n" 原样）。

    BOM 还原：utf-16/32 的 *-le/*-be 编码器不加 BOM，需要手动补回
    （utf-8-sig 编码器自身会加，不重复处理）。
    """
    text = normalized.replace("\n", eol) if eol != "\n" else normalized
    bom = _BOM_BY_ENCODING.get(encoding, b"")
    path.write_bytes(bom + text.encode(encoding, errors="replace"))


_BOM_BY_ENCODING = {
    "utf-16-le": b"\xff\xfe",
    "utf-16-be": b"\xfe\xff",
    "utf-32-le": b"\xff\xfe\x00\x00",
    "utf-32-be": b"\x00\x00\xfe\xff",
}


# ---------- 文件工具上下文 ----------

class FileToolContext(ToolContext):
    """文件工具的上下文：workdir + 读状态缓存 + 读取限额。"""

    def __init__(
        self,
        workdir: Path,
        state: FileStateCache | None = None,
        max_read_tokens: int = 25000,
        max_file_bytes: int = MAX_OUTPUT_SIZE,
    ):
        super().__init__(workdir)
        self.state = state or FileStateCache()
        self.max_read_tokens = max_read_tokens
        self.max_file_bytes = max_file_bytes
        # 同一路径的 Edit/Write 互斥。调度器已把它们当成屏障，这把锁挡住
        # 校验和写盘之间的窗口，避免后写完的那次盖掉前一次。
        self._path_locks: dict[str, threading.Lock] = {}
        self._path_locks_guard = threading.Lock()

    @contextmanager
    def exclusive_path(self, path: Path):
        key = self.state._key(path)
        with self._path_locks_guard:
            lock = self._path_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._path_locks[key] = lock
        with lock:
            yield

    def stat_mtime_ns(self, path: Path) -> int:
        """mtime 用纳秒整数比对，避免毫秒取整的精度坑。"""
        return path.stat().st_mtime_ns

    def check_fresh(self, path: Path) -> FileState:
        """staleness 双重检查：

        磁盘 mtime 与缓存一致 → 新鲜；不一致时若缓存是全量读，回退比对
        内存内容（Windows 云同步/杀软会造成 mtime 假阳性），内容相同则
        顺手修正缓存 mtime 并视为新鲜；否则抛 "Read it again"。
        """
        state = self.state.get(path)
        if state is None:
            raise ToolError(
                f"必须先 Read {path.name} 再修改（read-before-write）",
                error_code=5,
            )
        if state.is_partial_view:
            raise ToolError(
                f"{path.name} 只有部分视图，请先用 Read 完整读取", error_code=6
            )
        try:
            mtime_ns = self.stat_mtime_ns(path)
        except OSError:
            mtime_ns = -1
        if mtime_ns != state.mtime_ns:
            if state.offset == 1 and state.limit is None:
                try:
                    disk, _, _ = read_disk_text(path)
                except OSError:
                    disk = None
                if disk is not None and disk == state.content:
                    state.mtime_ns = mtime_ns  # mtime 假阳性，修正后放行
                    return state
            raise ToolError(
                f"文件在读取后被修改过（{path.name}），请重新 Read 再试",
                error_code=7,
            )
        return state
