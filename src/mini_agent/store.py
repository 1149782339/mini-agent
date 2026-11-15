"""SQLite 会话持久化。

一个会话在 messages 表里占多行（按 seq 排序），而不是一个会话一个文件：
- 追加写只 insert 一行，不重写整库；
- 压缩后整体替换（delete + 批量 insert），保证库里和内存一致；
- 跨会话可查询、可统计。

默认库位置 <项目根>/.mini_agent/sessions.db（跟着项目走，不污染家目录），
可用 MINI_AGENT_DB 覆盖；项目内不可写时退回 ~/.mini-agent/sessions.db。
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

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
"""


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

    # ---------- 消息 ----------
    def append(self, session_id: str, message: dict) -> int:
        seq = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        self.conn.execute(
            "INSERT INTO messages (session_id, seq, role, payload, created_at) VALUES (?,?,?,?,?)",
            (session_id, seq, message.get("role", ""), json.dumps(message, ensure_ascii=False), _now()),
        )
        self.conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))
        self.conn.commit()
        return seq

    def replace(self, session_id: str, messages: list[dict]) -> None:
        """压缩后整体覆盖，保证库与内存一致。"""
        with self.conn:
            self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            for i, msg in enumerate(messages, start=1):
                self.conn.execute(
                    "INSERT INTO messages (session_id, seq, role, payload, created_at) VALUES (?,?,?,?,?)",
                    (session_id, i, msg.get("role", ""), json.dumps(msg, ensure_ascii=False), _now()),
                )
            self.conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))

    def load(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT payload FROM messages WHERE session_id = ? ORDER BY seq", (session_id,)
        ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def message_count(self, session_id: str) -> int:
        return int(
            self.conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        )
