"""SQLite 会话持久化。

一个会话在 messages 表里占多行（按 seq 排序），而不是一个会话一个文件：
- 对话消息只追加，不因压缩而删除原文；
- 压缩结果写入 session_events（摘要 + 保留起点），发给模型的是由此重建的视图；
- /clear 同样只追加一条清空记录，原文仍在库里。

默认库位置 <项目根>/.mini_agent/sessions.db（跟着项目走，不污染家目录），
可用 MINI_AGENT_DB 覆盖；项目内不可写时退回 ~/.mini-agent/sessions.db。
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .tokens import content_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT '',
    workdir     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE(session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);
CREATE TABLE IF NOT EXISTS session_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind             TEXT NOT NULL,
    first_kept_seq   INTEGER,
    at_seq           INTEGER,
    payload          TEXT,
    before_tokens    INTEGER,
    after_tokens     INTEGER,
    dropped_messages INTEGER,
    summarised       INTEGER,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON session_events(session_id, id);
CREATE TABLE IF NOT EXISTS tasks (
    task_id            TEXT PRIMARY KEY,
    owner_session_id   TEXT,
    status             TEXT NOT NULL,
    notification_state TEXT NOT NULL,
    payload            TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(owner_session_id, status);
"""

TASK_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id            TEXT PRIMARY KEY,
    owner_session_id   TEXT,
    status             TEXT NOT NULL,
    notification_state TEXT NOT NULL,
    payload            TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(owner_session_id, status);
"""

COMPACTION_MARKER = "[COMPACTION]"

DATA_DIRNAME = ".mini_agent"   # 项目内数据目录（会话库 + spill）
HOME_DIRNAME = ".mini-agent"   # 家目录惯例位置，只在项目内不可写时兜底


def project_root() -> Path:
    """从本文件向上找 pyproject.toml 定位项目根；装在 site-packages 时退回 cwd。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def project_data_dir() -> Path:
    """项目内数据目录：<项目根>/.mini_agent（会话库与 spill 都放这里）。"""
    return project_root() / DATA_DIRNAME


def default_db_path() -> Path:
    env = os.environ.get("MINI_AGENT_DB")
    if env:
        return Path(env).expanduser()
    return project_root() / DATA_DIRNAME / "sessions.db"


def _is_writable(directory: Path) -> bool:
    """真写一个空文件再删掉——比 is_dir()/access() 判断可靠（Windows 上尤其）。"""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".write-probe-{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def resolve_db_path() -> tuple[Path, str | None]:
    """会话库路径：默认项目内 .mini_agent/，写了 MINI_AGENT_DB 就完全听它的。

    项目内不可写（只读检出/受限环境）时退回 ~/.mini-agent/，并给出提示。
    """
    if os.environ.get("MINI_AGENT_DB"):
        return default_db_path(), None
    wanted = default_db_path()
    if _is_writable(wanted.parent):
        return wanted, None
    fallback = Path.home() / HOME_DIRNAME / "sessions.db"
    if _is_writable(fallback.parent):
        return fallback, f"项目内数据目录 {wanted.parent} 不可写，会话库改用 {fallback}"
    return wanted, None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _payload_json(message: dict) -> str:
    """写入库的消息 JSON。_seq 只是行号缓存，不入库；_usage 等私有字段要保留。"""
    clean = {k: v for k, v in message.items() if k != "_seq"}
    return json.dumps(clean, ensure_ascii=False)


def _is_marker(message: dict) -> bool:
    return content_text(message.get("content")).startswith(COMPACTION_MARKER)


def _system_message(system_prompt: str) -> dict:
    return {"role": "system", "content": system_prompt}


class TaskStore:
    """后台任务快照；每次操作使用独立连接，允许 worker 线程安全写入。

    SessionStore 仍只由主线程使用。这里不保存会话消息，也不直接写通知历史，
    只保存任务终态、输出与通知消费状态，供下次进程启动时恢复。
    """

    def __init__(self, path: Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connection() as conn:
            conn.executescript(TASK_SCHEMA)

    @contextmanager
    def _connection(self):
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=5)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 5000")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def save_task(self, snapshot: dict) -> None:
        payload = json.dumps(snapshot, ensure_ascii=False, default=str)
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO tasks
                   (task_id, owner_session_id, status, notification_state, payload, updated_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(task_id) DO UPDATE SET
                       owner_session_id=excluded.owner_session_id,
                       status=excluded.status,
                       notification_state=excluded.notification_state,
                       payload=excluded.payload,
                       updated_at=excluded.updated_at""",
                (
                    snapshot["task_id"],
                    snapshot.get("owner_session_id"),
                    snapshot["status"],
                    snapshot["notification_state"],
                    payload,
                    _now(),
                ),
            )

    def delete_task(self, task_id: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))

    def load_tasks(self) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute("SELECT payload FROM tasks ORDER BY rowid").fetchall()
        return [json.loads(row["payload"]) for row in rows]


class SessionStore:
    def __init__(self, path: Path | None = None):
        if path is None:
            self.path, note = resolve_db_path()
            if note:
                print(f"[mini-agent] {note}", file=sys.stderr)
        else:
            self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------- 会话 ----------
    def create(
        self, title: str = "", model: str = "", workdir: str = "", session_id: str | None = None
    ) -> str:
        sid = session_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now = _now()
        self.conn.execute(
            "INSERT INTO sessions (id, title, model, workdir, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (sid, title, model, workdir, now, now),
        )
        self.conn.commit()
        return sid

    def set_title(self, session_id: str, title: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), session_id),
        )
        self.conn.commit()

    def touch(self, session_id: str) -> None:
        self.conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))
        self.conn.commit()

    def exists(self, session_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row is not None

    def latest(self) -> str | None:
        row = self.conn.execute(
            "SELECT id FROM sessions ORDER BY updated_at DESC, created_at DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

    def find_by_title(self, title: str) -> str | None:
        row = self.conn.execute(
            "SELECT id FROM sessions WHERE title = ? ORDER BY updated_at DESC LIMIT 1", (title,)
        ).fetchone()
        return row["id"] if row else None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """SELECT s.id, s.title, s.model, s.workdir, s.created_at, s.updated_at,
                      (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS msg_count
               FROM sessions s ORDER BY s.updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- 消息（只追加）----------
    def _max_seq(self, session_id: str) -> int:
        return int(
            self.conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
        )

    def append(self, session_id: str, message: dict) -> int:
        seq = self._max_seq(session_id) + 1
        self.conn.execute(
            "INSERT INTO messages (session_id, seq, role, payload, created_at) VALUES (?,?,?,?,?)",
            (session_id, seq, message.get("role", ""), _payload_json(message), _now()),
        )
        self.conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))
        self.conn.commit()
        message["_seq"] = seq
        return seq

    def load_entries(self, session_id: str) -> list[tuple[int, dict]]:
        """按 seq 返回全部原文，并在消息上附 _seq（不写回库）。"""
        rows = self.conn.execute(
            "SELECT seq, payload FROM messages WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
        entries: list[tuple[int, dict]] = []
        for row in rows:
            msg = json.loads(row["payload"])
            msg["_seq"] = row["seq"]
            entries.append((row["seq"], msg))
        return entries

    def load(self, session_id: str) -> list[dict]:
        """全部原文（含压缩前的历史），不是发给模型的视图。"""
        return [msg for _, msg in self.load_entries(session_id)]

    def message_count(self, session_id: str) -> int:
        """库里保存的原文条数（含压缩前与清空前的记录）。"""
        return int(
            self.conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        )

    # ---------- 压缩 / 清空事件 ----------
    def _events(self, session_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM session_events WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()

    def _latest_clear(self, session_id: str) -> sqlite3.Row | None:
        events = self._events(session_id)
        clear = None
        for ev in events:
            if ev["kind"] == "clear":
                clear = ev
        return clear

    def mark_clear(self, session_id: str) -> None:
        """清空当前视图：原文保留，之后的 load_view 从这条记录之后开始。"""
        at_seq = self._max_seq(session_id)
        with self.conn:
            self.conn.execute(
                """INSERT INTO session_events
                   (session_id, kind, first_kept_seq, at_seq, payload, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (session_id, "clear", None, at_seq, "", _now()),
            )
            self.conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id)
            )

    def record_compaction(
        self,
        session_id: str,
        view: list[dict],
        *,
        before_tokens: int = 0,
        after_tokens: int = 0,
        dropped_messages: int = 0,
        summarised: bool | None = None,
    ) -> None:
        """记录一次压缩：保存摘要消息与保留起点，不删除 messages 里的原文。"""
        marker = None
        first_kept: int | None = None
        for msg in view:
            if msg.get("role") == "system":
                continue
            if _is_marker(msg):
                marker = {k: v for k, v in msg.items() if k != "_seq"}
                continue
            seq = msg.get("_seq")
            if isinstance(seq, int) and (first_kept is None or seq < first_kept):
                first_kept = seq
        if first_kept is None:
            first_kept = self._max_seq(session_id) + 1
        # 以最终视图为准：摘要若在后续机械裁剪里被丢掉，就不要再标成 summarised
        if marker is not None:
            summarised = True
        elif summarised is None:
            summarised = False
        with self.conn:
            self.conn.execute(
                """INSERT INTO session_events
                   (session_id, kind, first_kept_seq, at_seq, payload,
                    before_tokens, after_tokens, dropped_messages, summarised, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id,
                    "compaction",
                    first_kept,
                    None,
                    json.dumps(marker, ensure_ascii=False) if marker else "",
                    before_tokens,
                    after_tokens,
                    dropped_messages,
                    1 if summarised else 0,
                    _now(),
                ),
            )
            self.conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id)
            )

    def load_view(self, session_id: str, system_prompt: str = "") -> list[dict]:
        """发给模型（以及 resume 进内存）的历史视图：摘要 + 压缩后仍保留的原文。"""
        entries = self.load_entries(session_id)
        events = self._events(session_id)

        after_seq = 0
        last_clear_id = 0
        for ev in events:
            if ev["kind"] == "clear":
                after_seq = ev["at_seq"] or 0
                last_clear_id = ev["id"]

        visible = [(seq, msg) for seq, msg in entries if seq > after_seq]
        compaction = None
        for ev in events:
            if ev["id"] > last_clear_id and ev["kind"] == "compaction":
                compaction = ev

        system = None
        for _, msg in entries:
            if msg.get("role") == "system":
                system = {k: v for k, v in msg.items()}
                break
        if system is None:
            system = _system_message(system_prompt)

        if compaction is None:
            rest = [msg for _, msg in visible if msg.get("role") != "system"]
            return [system, *rest]

        first_kept = compaction["first_kept_seq"] or 0
        kept = [msg for seq, msg in visible if seq >= first_kept and msg.get("role") != "system"]
        marker = None
        raw_payload = compaction["payload"] or ""
        if raw_payload and compaction["summarised"]:
            try:
                loaded = json.loads(raw_payload)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                marker = loaded
        if marker is not None:
            return [system, marker, *kept]
        return [system, *kept]

    def view_message_count(self, session_id: str, system_prompt: str = "") -> int:
        """当前模型视图的消息条数（含 system）。"""
        return len(self.load_view(session_id, system_prompt=system_prompt))

    def current_user_message_count(self, session_id: str, system_prompt: str = "") -> int:
        """当前视图里的 user 消息数，用来判断是否该写入会话标题。"""
        return sum(
            1
            for msg in self.load_view(session_id, system_prompt=system_prompt)
            if msg.get("role") == "user" and not _is_marker(msg)
        )
