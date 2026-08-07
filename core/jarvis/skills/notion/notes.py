"""Note crawling and search.

The problem this solves: Notion's own search API indexes **titles only**. Ask
it for "the squeeze theorem" and it finds nothing unless a page is literally
named that — even if the phrase appears in a dozen note bodies. Your notes have
no tags, no topics and no relations either; the only metadata is a title.

So JARVIS crawls the tree once, extracts the text of every note, and keeps a
local SQLite FTS5 index. Search is then instant and offline, and it finds notes
by what's *in* them rather than what they're called.

The crawl is incremental — pages are skipped when their last_edited_time hasn't
moved since the last pass, so a re-index costs one API call per unit rather
than one per note.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from ...config import VAR_DIR
from .blocks import BREADCRUMB_BLOCKS, block_text, page_url, title_of
from .client import NOTE_ROOTS, NotionClient

log = logging.getLogger("jarvis.notes")

INDEX_PATH = VAR_DIR / "notes.db"
MAX_DEPTH = 6  # deep enough for your structure, shallow enough to stay bounded

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id          TEXT PRIMARY KEY,
    course      TEXT NOT NULL,
    unit        TEXT,
    chapter     TEXT,
    title       TEXT NOT NULL,
    url         TEXT NOT NULL,
    body        TEXT,
    edited_at   TEXT,
    indexed_at  REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, chapter, unit, course, body,
    content='notes', content_rowid='rowid', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid, title, chapter, unit, course, body)
  VALUES (new.rowid, new.title, new.chapter, new.unit, new.course, new.body);
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, title, chapter, unit, course, body)
  VALUES ('delete', old.rowid, old.title, old.chapter, old.unit, old.course, old.body);
END;
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, title, chapter, unit, course, body)
  VALUES ('delete', old.rowid, old.title, old.chapter, old.unit, old.course, old.body);
  INSERT INTO notes_fts(rowid, title, chapter, unit, course, body)
  VALUES (new.rowid, new.title, new.chapter, new.unit, new.course, new.body);
END;

CREATE TABLE IF NOT EXISTS crawl_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass
class Note:
    id: str
    course: str
    unit: str
    chapter: str
    title: str
    url: str
    body: str
    edited_at: str = ""

    def spoken_location(self) -> str:
        """How a person would say where this note is."""
        bits = [b for b in (self.course, self.chapter or self.unit) if b]
        return " — ".join(bits) if bits else self.course


class NoteIndex:
    def __init__(self, path: Path = INDEX_PATH):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---- writing ---------------------------------------------------------

    def upsert(self, note: Note) -> None:
        self._conn.execute(
            "INSERT INTO notes (id, course, unit, chapter, title, url, body,"
            " edited_at, indexed_at) VALUES (?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET course=excluded.course,"
            " unit=excluded.unit, chapter=excluded.chapter, title=excluded.title,"
            " url=excluded.url, body=excluded.body, edited_at=excluded.edited_at,"
            " indexed_at=excluded.indexed_at",
            (note.id, note.course, note.unit, note.chapter, note.title, note.url,
             note.body, note.edited_at, time.time()),
        )
        self._conn.commit()

    def edited_at(self, note_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT edited_at FROM notes WHERE id=?", (note_id,)
        ).fetchone()
        return row["edited_at"] if row else None

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM notes").fetchone()["n"]

    def set_state(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO crawl_state (key, value) VALUES (?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_state(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM crawl_state WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    # ---- reading ---------------------------------------------------------

    def search(self, query: str, *, course: str | None = None,
               limit: int = 8) -> list[dict]:
        """Full-text search. Title and chapter matches outrank body matches."""
        match = _to_fts_query(query)
        if not match:
            return []

        sql = (
            "SELECT n.id, n.course, n.unit, n.chapter, n.title, n.url,"
            "       snippet(notes_fts, 4, '', '', '…', 18) AS excerpt,"
            "       bm25(notes_fts, 8.0, 4.0, 2.0, 1.0, 1.0) AS score"
            " FROM notes_fts JOIN notes n ON n.rowid = notes_fts.rowid"
            " WHERE notes_fts MATCH ?"
        )
        params: list = [match]
        if course:
            sql += " AND n.course = ?"
            params.append(course)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            log.warning("fts query failed for %r: %s", query, exc)
            return []
        return [dict(row) for row in rows]

    def by_course(self, course: str, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, course, unit, chapter, title, url FROM notes"
            " WHERE course LIKE ? ORDER BY chapter, title LIMIT ?",
            (f"%{course}%", limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def courses(self) -> list[str]:
        return [r["course"] for r in
                self._conn.execute("SELECT DISTINCT course FROM notes ORDER BY course")]

    def close(self) -> None:
        self._conn.close()


def _to_fts_query(query: str) -> str:
    """Turn spoken phrasing into an FTS5 expression.

    Speech gives you "uh the notes about riemann sums", not a search operator,
    so strip punctuation and filler, then OR the remaining terms so a partial
    match still returns something rather than nothing.
    """
    stop = {
        "the", "a", "an", "my", "me", "i", "about", "on", "for", "of", "in",
        "notes", "note", "find", "show", "open", "what", "where", "that",
        "talk", "talks", "talked", "regarding", "related", "to", "and", "is",
        "uh", "um", "like", "please", "can", "you", "do", "have", "with",
    }
    words = re.findall(r"[a-z0-9']+", query.lower())
    terms = [w for w in words if w not in stop and len(w) > 1]
    if not terms:
        terms = [w for w in words if len(w) > 2]
    if not terms:
        return ""
    # Prefix-match the last term so "integra" still finds "integrals".
    quoted = [f'"{t}"' for t in terms[:-1]] + [f'"{terms[-1]}"*']
    return " OR ".join(quoted)


class NoteCrawler:
    """Walks the note tree and fills the index."""

    def __init__(self, client: NotionClient, index: NoteIndex):
        self._client = client
        self._index = index

    async def crawl(self, *, force: bool = False) -> int:
        """Index every note. Returns how many were written."""
        if not self._client.enabled:
            log.info("skipping note crawl — no NOTION_TOKEN")
            return 0

        started = time.monotonic()
        written = 0

        for course, database_id in NOTE_ROOTS.items():
            try:
                written += await self._crawl_course(course, database_id, force)
            except Exception:
                log.exception("crawl failed for %s", course)

        self._index.set_state("last_crawl", str(time.time()))
        log.info("indexed %d notes in %.1fs (%d total)",
                 written, time.monotonic() - started, self._index.count())
        return written

    async def _crawl_course(self, course: str, database_id: str, force: bool) -> int:
        written = 0
        async for unit_page in self._client.query(database_id):
            unit = title_of(unit_page)
            unit_id = unit_page["id"]

            # The unit page itself is worth indexing — it holds the chapter
            # headings and any loose notes typed directly into it.
            written += await self._walk(
                block_id=unit_id, course=course, unit=unit, chapter="",
                title=unit, url=page_url(unit_id), page_id=unit_id,
                edited_at=unit_page.get("last_edited_time", ""),
                force=force, depth=0,
            )
        return written

    async def _walk(self, *, block_id: str, course: str, unit: str, chapter: str,
                    title: str, url: str, page_id: str, edited_at: str,
                    force: bool, depth: int) -> int:
        """Collect this page's own text, then recurse into nested child pages.

        `chapter` is threaded down from enclosing toggles and headings, because
        that's the only place the chapter name exists.
        """
        if depth > MAX_DEPTH:
            return 0

        if not force and edited_at and self._index.edited_at(page_id) == edited_at:
            return 0  # unchanged since last crawl

        own_text: list[str] = []
        children: list[tuple[str, str, str]] = []  # (child_page_id, title, chapter)

        async def descend(current_id: str, breadcrumb: str, level: int) -> None:
            if level > MAX_DEPTH:
                return
            try:
                blocks = [b async for b in self._client.blocks(current_id)]
            except Exception as exc:
                log.debug("could not read blocks of %s: %s", current_id, exc)
                return

            for block in blocks:
                block_type = block.get("type", "")

                if block_type == "child_page":
                    child_title = (block.get("child_page") or {}).get("title", "")
                    children.append((block["id"], child_title, breadcrumb))
                    continue

                text = block_text(block)
                if text:
                    own_text.append(text)

                if block.get("has_children"):
                    # A toggle or heading renames the breadcrumb for everything
                    # beneath it — that's how "Chapter 5: Integrals" is captured.
                    next_crumb = (
                        text if block_type in BREADCRUMB_BLOCKS and text else breadcrumb
                    )
                    await descend(block["id"], next_crumb, level + 1)

        await descend(block_id, chapter, depth)

        body = "\n".join(own_text).strip()
        written = 0
        if body or not children:
            self._index.upsert(Note(
                id=page_id, course=course, unit=unit, chapter=chapter,
                title=title or unit, url=url, body=body, edited_at=edited_at,
            ))
            written = 1

        # Recurse into the notes themselves, one level of concurrency at a time
        # so the rate limiter isn't fighting a fan-out.
        for child_id, child_title, child_chapter in children:
            try:
                child_page = await self._client.page(child_id)
            except Exception as exc:
                log.debug("could not read page %s: %s", child_id, exc)
                continue

            written += await self._walk(
                block_id=child_id, course=course, unit=unit,
                chapter=_clean_chapter(child_chapter),
                title=child_title or title_of(child_page),
                url=page_url(child_id), page_id=child_id,
                edited_at=child_page.get("last_edited_time", ""),
                force=force, depth=depth + 1,
            )
            await asyncio.sleep(0)  # let other tasks breathe

        return written


def _clean_chapter(raw: str) -> str:
    """'Topics/Chapters' is scaffolding, not a chapter name."""
    if not raw or raw.strip().lower() in {"topics/chapters", "topics", "chapters"}:
        return ""
    return raw.strip()
