"""Local SQLite store. Nothing here ever leaves the machine.

Scope is exactly what you approved: which apps and Chrome tabs are open, their
titles and URLs, and when. Page *contents* are not read or stored — the schema
has no column for them, so enabling it would be a visible, deliberate change.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from ..config import VAR_DIR

DB_PATH = VAR_DIR / "jarvis.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS capabilities (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    granted      INTEGER NOT NULL DEFAULT 0,
    granted_at   REAL,
    revoked_at   REAL,
    use_count    INTEGER NOT NULL DEFAULT 0
);

-- Passive observation: apps, window titles, Chrome tabs.
CREATE TABLE IF NOT EXISTS observations (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    kind    TEXT NOT NULL,          -- 'app' | 'tab' | 'window'
    app     TEXT,
    title   TEXT,
    url     TEXT,
    active  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_obs_ts   ON observations(ts DESC);
CREATE INDEX IF NOT EXISTS idx_obs_kind ON observations(kind, ts DESC);

-- Things JARVIS learned about you, from onboarding or from you saying them.
CREATE TABLE IF NOT EXISTS facts (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    source     TEXT NOT NULL,       -- 'onboarding' | 'inferred' | 'told'
    confidence REAL NOT NULL DEFAULT 1.0,
    updated_at REAL NOT NULL
);

-- Every exchange, for recall and for tuning the local/cloud router.
CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    transcript  TEXT,
    response    TEXT,
    route       TEXT,               -- 'local' | 'cloud'
    tools       TEXT,               -- json array of tool ids used
    latency_ms  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_int_ts ON interactions(ts DESC);

-- Rate-limits autonomous speech so "proactive" doesn't become "constant".
CREATE TABLE IF NOT EXISTS proactive_log (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts   REAL NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pro_ts ON proactive_log(ts DESC);
"""


class Memory:
    def __init__(self, path: Path = DB_PATH):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---- observations ----------------------------------------------------

    def observe(self, kind: str, *, app: str | None = None, title: str | None = None,
                url: str | None = None, active: bool = False) -> None:
        self._conn.execute(
            "INSERT INTO observations (ts, kind, app, title, url, active)"
            " VALUES (?,?,?,?,?,?)",
            (time.time(), kind, app, title, url, int(active)),
        )
        self._conn.commit()

    def recent_tabs(self, limit: int = 40) -> list[sqlite3.Row]:
        """Most recently seen unique tabs. Backs 'open the tab about X'."""
        return self._conn.execute(
            "SELECT url, title, MAX(ts) AS ts FROM observations"
            " WHERE kind='tab' AND url IS NOT NULL"
            " GROUP BY url ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def search_tabs(self, query: str, limit: int = 10) -> list[sqlite3.Row]:
        like = f"%{query}%"
        return self._conn.execute(
            "SELECT url, title, MAX(ts) AS ts FROM observations"
            " WHERE kind='tab' AND (title LIKE ? OR url LIKE ?)"
            " GROUP BY url ORDER BY ts DESC LIMIT ?",
            (like, like, limit),
        ).fetchall()

    def prune(self, older_than_days: int) -> int:
        cutoff = time.time() - older_than_days * 86400
        cur = self._conn.execute("DELETE FROM observations WHERE ts < ?", (cutoff,))
        self._conn.commit()
        return cur.rowcount

    # ---- facts -----------------------------------------------------------

    def remember(self, key: str, value: str, source: str = "told",
                 confidence: float = 1.0) -> None:
        self._conn.execute(
            "INSERT INTO facts (key, value, source, confidence, updated_at)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
            " source=excluded.source, confidence=excluded.confidence,"
            " updated_at=excluded.updated_at",
            (key, value, source, confidence, time.time()),
        )
        self._conn.commit()

    def recall(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def all_facts(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self._conn.execute("SELECT key, value FROM facts")}

    # ---- interactions ----------------------------------------------------

    def log_interaction(self, transcript: str, response: str, route: str,
                        tools: Iterable[str], latency_ms: int) -> None:
        self._conn.execute(
            "INSERT INTO interactions (ts, transcript, response, route, tools, latency_ms)"
            " VALUES (?,?,?,?,?,?)",
            (time.time(), transcript, response, route, json.dumps(list(tools)), latency_ms),
        )
        self._conn.commit()

    def recent_interactions(self, limit: int = 6) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT transcript, response FROM interactions ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()

    # ---- proactive -------------------------------------------------------

    def log_proactive(self, kind: str, text: str) -> None:
        self._conn.execute(
            "INSERT INTO proactive_log (ts, kind, text) VALUES (?,?,?)",
            (time.time(), kind, text),
        )
        self._conn.commit()

    def proactive_count_since(self, seconds_ago: float) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM proactive_log WHERE ts > ?",
            (time.time() - seconds_ago,),
        ).fetchone()
        return row["n"]

    def said_recently(self, kind: str, within_s: float) -> bool:
        """Stops it repeating the same nudge every tick."""
        row = self._conn.execute(
            "SELECT 1 FROM proactive_log WHERE kind=? AND ts > ? LIMIT 1",
            (kind, time.time() - within_s),
        ).fetchone()
        return row is not None

    # ---- raw -------------------------------------------------------------

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur.fetchall()

    def close(self) -> None:
        self._conn.close()
