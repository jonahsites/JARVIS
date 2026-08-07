"""Passive observation.

Scope is exactly what you approved: which apps and Chrome tabs are open, their
titles and URLs, and when. Page *contents* are never read — `observe_page_content`
defaults to False and nothing in the schema stores body text, so turning it on
would be a deliberate, visible change rather than a slow drift.

This is what makes "open the tab about that thing" work later, and what lets the
proactive engine notice you've been stuck in one document for forty minutes.
"""

from __future__ import annotations

import asyncio
import logging
import time

from ..config import MemoryConfig
from ..memory.db import Memory
from . import mac

log = logging.getLogger("jarvis.observer")


class Observer:
    def __init__(self, memory: Memory, config: MemoryConfig):
        self._memory = memory
        self._config = config
        self._last_app: str | None = None
        self._app_since: float = time.monotonic()
        # Tabs churn constantly; only log a URL when it's new to this session,
        # otherwise the table fills with duplicates within a day.
        self._seen_tabs: set[str] = set()

    @property
    def current_app(self) -> str | None:
        return self._last_app

    @property
    def app_minutes(self) -> float:
        """How long the frontmost app has been frontmost."""
        return (time.monotonic() - self._app_since) / 60

    async def tick(self) -> None:
        if self._config.observe_apps:
            await self._observe_app()
        if self._config.observe_tabs:
            await self._observe_tabs()

    async def _observe_app(self) -> None:
        try:
            app = await mac.frontmost_app()
        except Exception as exc:
            log.debug("frontmost app unavailable: %s", exc)
            return

        if app and app != self._last_app:
            self._memory.observe("app", app=app, active=True)
            self._last_app = app
            self._app_since = time.monotonic()

    async def _observe_tabs(self) -> None:
        try:
            tabs = await mac.list_tabs()
        except Exception as exc:
            log.debug("chrome tabs unavailable: %s", exc)
            return

        for tab in tabs:
            url = tab.get("url") or ""
            if not url or url.startswith("chrome://"):
                continue
            # Re-log an active tab occasionally so recency ordering stays useful.
            if url in self._seen_tabs and not tab.get("active"):
                continue
            self._seen_tabs.add(url)
            self._memory.observe(
                "tab", app="Google Chrome", title=tab.get("title"),
                url=url, active=tab.get("active", False),
            )

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("observer tick failed")
            await asyncio.sleep(self._config.poll_seconds)
