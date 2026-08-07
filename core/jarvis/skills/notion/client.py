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


class DB:
    """Database ids from your workspace."""

    ASSIGNMENTS = "24a4b8e1-c691-818e-b01d-000b969755f8"
    COURSES = "24a4b8e1-c691-819e-9f34-000ba0782495"
    CLASS_NOTES = "24a4b8e1-c691-818e-bfb1-000b8a151f7f"
    UNITS = "6b6235fc-aa46-4354-b36e-b7db46199d72"
    TASKS = "24a4b8e1-c691-816f-b4fb-000b1f02de60"
    PLANNER = "24a4b8e1-c691-8139-9463-000b44e3ca50"
    GOALS = "24a4b8e1-c691-8116-82ef-000b884d7223"
    BOOKMARKS = "24a4b8e1-c691-817c-9b84-000bea29423b"
    DRIVE = "24a4b8e1-c691-81b2-9a1e-000b44645bce"
    BRAIN_DUMP = "3a99caf1-1b45-46a5-a593-a1dde40d3ea1"
    DAYS_OFF = "5eb5b06a-5171-4ac5-810a-96b375073d4d"


# The six per-course Unit databases behind your Notes page. These are the
# entry points for note search — each row is a unit, and the notes themselves
# are child pages nested inside it.
NOTE_ROOTS: dict[str, str] = {
    "AP Calculus AB": "3a14b8e1-c691-8018-90a0-000baf11ed64",
    "AP Environmental Science": "3a14b8e1-c691-80e0-ab09-000b9c30bea6",
    "AP Psychology": "0de4b8e1-c691-838a-9d25-07c8437eec96",
    "English Seminar: Garden State": "3a14b8e1-c691-8099-9fa6-000be3092417",
    "Spanish Cinema": "3a14b8e1-c691-80e5-af0d-000be7d89b95",
    "AP Economics": "3a14b8e1-c691-8082-9547-000bb6ef4602",
}


class NotionError(RuntimeError):
    pass


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
