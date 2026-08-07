"""Notion REST client.

Direct REST rather than the MCP server: your plan rate-limits MCP's SQL query
tool, and the REST API has no such cap and is faster.

Every database id below was read off your actual workspace, so nothing here is
guessed. The ids are stable — Notion doesn't change them when you rename or
move things.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import httpx

log = logging.getLogger("jarvis.notion")

API = "https://api.notion.com/v1"
VERSION = "2022-06-28"


# Databases are found by title rather than by hardcoded id.
#
# Two reasons. First, Notion has two different identifiers for what looks like
# one database — a *database* id (what the REST API wants) and a *data source*
# id (what the MCP surfaces). They are not interchangeable, and using the wrong
# one gives a 404 that reads exactly like a permissions error. Second, an id
# baked into source breaks silently the day a database is recreated.
#
# So: ask Notion what it can see, match on title. Self-healing, and when a
# database is genuinely missing the error can say which one.

# key -> the titles that count as a match, best first.
WANTED: dict[str, list[str]] = {
    "assignments": ["Assignments"],
    "courses": ["Courses"],
    "class_notes": ["Class Notes"],
    "units": ["Units"],
    "tasks": ["Tasks"],
    "planner": ["Planner"],
    "goals": ["Goals"],
    "bookmarks": ["Bookmarks"],
    "drive": ["Drive"],
    "brain_dump": ["Brain Dump"],
    "days_off": ["Days Off", "Days off"],
}

# The six per-course Unit databases behind the Notes page. Each row is a unit,
# and the notes themselves are child pages nested inside it.
NOTE_ROOT_TITLES: dict[str, list[str]] = {
    "AP Calculus AB": ["AP Calculus AB Units"],
    "AP Environmental Science": ["AP Environmental Science Units",
                                 "AP Environmental Science"],
    "AP Psychology": ["AP Psychology Units"],
    "English Seminar: Garden State": ["English Seminar: Garden State Units"],
    "Spanish Cinema": ["Spanish Cinema Units"],
    "AP Economics": ["AP Economics Units"],
}


class NotionError(RuntimeError):
    pass


class DatabaseRegistry:
    """Maps friendly names to whatever database ids Notion actually reports."""

    def __init__(self) -> None:
        self._ids: dict[str, str] = {}
        self._note_roots: dict[str, str] = {}
        self._seen: list[str] = []

    async def discover(self, client: "NotionClient") -> None:
        databases = await client.all_databases()
        self._seen = sorted(d["title"] for d in databases if d["title"])

        by_title = {d["title"].strip().lower(): d["id"] for d in databases}

        def find(candidates: list[str]) -> str | None:
            for candidate in candidates:
                if (found := by_title.get(candidate.strip().lower())):
                    return found
            # Fall back to a unique substring match — catches trailing spaces
            # and small renames like "Courses " or "My Courses".
            for candidate in candidates:
                needle = candidate.strip().lower()
                hits = [i for t, i in by_title.items() if needle in t]
                if len(hits) == 1:
                    return hits[0]
            return None

        for key, titles in WANTED.items():
            if (found := find(titles)):
                self._ids[key] = found

        for course, titles in NOTE_ROOT_TITLES.items():
            if (found := find(titles)):
                self._note_roots[course] = found

        log.info("notion: %d databases visible, resolved %d/%d core + %d/%d note roots",
                 len(databases), len(self._ids), len(WANTED),
                 len(self._note_roots), len(NOTE_ROOT_TITLES))

        missing = [k for k in WANTED if k not in self._ids]
        if missing:
            log.warning("notion: could not find %s", ", ".join(missing))

    def id_for(self, key: str) -> str:
        found = self._ids.get(key)
        if not found:
            raise NotionError(
                f"the {key.replace('_', ' ')} database isn't shared with this "
                f"integration. Visible databases: "
                f"{', '.join(self._seen) if self._seen else 'none'}"
            )
        return found

    def has(self, key: str) -> bool:
        return key in self._ids

    @property
    def note_roots(self) -> dict[str, str]:
        return dict(self._note_roots)

    @property
    def visible(self) -> list[str]:
        return list(self._seen)

    @property
    def resolved(self) -> dict[str, str]:
        return dict(self._ids)


class NotionClient:
    def __init__(self, token: str, timeout_s: float = 20.0):
        self._token = token
        self._timeout = timeout_s
        self._client: httpx.AsyncClient | None = None
        # Notion allows ~3 requests/second; the crawler would blow straight
        # through that without this.
        self._gate = asyncio.Semaphore(3)

    @property
    def enabled(self) -> bool:
        return bool(self._token)

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=API,
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Notion-Version": VERSION,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if not self.enabled:
            raise NotionError("NOTION_TOKEN is not set")

        client = await self._http()
        async with self._gate:
            for attempt in range(4):
                response = await client.request(method, path, **kwargs)

                if response.status_code == 429:
                    wait = float(response.headers.get("Retry-After", 1 + attempt))
                    log.warning("notion rate limited, waiting %.1fs", wait)
                    await asyncio.sleep(wait)
                    continue

                if response.status_code >= 500:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue

                if response.status_code >= 400:
                    detail = response.json().get("message", response.text[:200])
                    raise NotionError(f"{response.status_code}: {detail}")

                return response.json()

        raise NotionError(f"{method} {path} failed after retries")

    # ---- reads -----------------------------------------------------------

    async def query(self, database_id: str, *, filter: dict | None = None,
                    sorts: list | None = None,
                    page_size: int = 100) -> AsyncIterator[dict[str, Any]]:
        """Yield every row, following pagination."""
        cursor: str | None = None
        while True:
            body: dict[str, Any] = {"page_size": page_size}
            if filter:
                body["filter"] = filter
            if sorts:
                body["sorts"] = sorts
            if cursor:
                body["start_cursor"] = cursor

            data = await self._request("POST", f"/databases/{database_id}/query", json=body)
            for row in data.get("results", []):
                yield row

            if not data.get("has_more"):
                return
            cursor = data.get("next_cursor")

    async def page(self, page_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/pages/{page_id}")

    async def blocks(self, block_id: str,
                     page_size: int = 100) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": page_size}
            if cursor:
                params["start_cursor"] = cursor
            data = await self._request("GET", f"/blocks/{block_id}/children", params=params)
            for block in data.get("results", []):
                yield block
            if not data.get("has_more"):
                return
            cursor = data.get("next_cursor")

    async def database(self, database_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/databases/{database_id}")

    async def whoami(self) -> dict[str, Any]:
        return await self._request("GET", "/users/me")

    async def all_databases(self) -> list[dict[str, Any]]:
        """Every database this integration can actually see.

        Also the single most useful diagnostic there is: if this comes back
        empty the token is fine and the integration simply hasn't been
        connected to any pages.
        """
        found: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            body: dict[str, Any] = {
                "filter": {"value": "database", "property": "object"},
                "page_size": 100,
            }
            if cursor:
                body["start_cursor"] = cursor

            data = await self._request("POST", "/search", json=body)
            for row in data.get("results", []):
                title = "".join(t.get("plain_text", "") for t in row.get("title") or [])
                found.append({"id": row["id"], "title": title.strip()})

            if not data.get("has_more"):
                return found
            cursor = data.get("next_cursor")

    # ---- writes ----------------------------------------------------------

    async def create_page(self, parent_database_id: str,
                          properties: dict[str, Any],
                          children: list[dict] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "parent": {"database_id": parent_database_id},
            "properties": properties,
        }
        if children:
            body["children"] = children
        return await self._request("POST", "/pages", json=body)

    async def update_page(self, page_id: str,
                          properties: dict[str, Any]) -> dict[str, Any]:
        return await self._request("PATCH", f"/pages/{page_id}",
                                   json={"properties": properties})
