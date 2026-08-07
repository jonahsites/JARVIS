"""The capability ledger — your permission rule, implemented literally.

    Confirmation is per capability, not per argument.

The first time JARVIS ever opens a Chrome tab it asks. After you say yes it
opens YouTube, Instagram, anything, forever, without asking again. Same for
texting: approved once, then it texts anyone. Same for adding assignments:
approved once, then any class.

So the grant key is the *tool id* and never includes arguments. There is
deliberately no per-argument path in this file — that was the whole point.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Awaitable, Callable

from ..bus import EV_CAPABILITY, EV_CAPABILITY_REPLY, Bus, Event
from ..memory.db import Memory

log = logging.getLogger("jarvis.capabilities")

# Spoken once, the first time each capability is exercised. Written to be
# answerable with a plain yes or no.
PROMPTS: dict[str, str] = {
    "chrome.open_tab": "I haven't opened a browser tab before. Okay if I start doing that?",
    "chrome.close_tab": "Okay if I start closing tabs?",
    "chrome.switch_tab": "Okay if I start switching between your tabs?",
    "chrome.read_tabs": "Okay if I keep track of what tabs you have open?",
    "apps.open": "I haven't opened an app before. Okay if I start doing that?",
    "apps.quit": "Okay if I start quitting apps?",
    "messages.send": "I've never sent a message for you. Okay if I start texting people?",
    "messages.read": "Okay if I read your messages?",
    "notion.read": "Okay if I read your Notion?",
    "notion.create_assignment": "I haven't added an assignment before. Okay if I start?",
    "notion.update_assignment": "Okay if I start editing assignments?",
    "notion.create_note": "Okay if I start creating notes in Notion?",
    "lecturesynth.read": "Okay if I pull from LectureSynth?",
    "system.run_shortcut": "Okay if I run shortcuts on your Mac?",
}


class CapabilityDenied(Exception):
    def __init__(self, capability_id: str):
        self.capability_id = capability_id
        super().__init__(f"capability not granted: {capability_id}")


class CapabilityLedger:
    def __init__(self, memory: Memory, bus: Bus, *, ask_timeout_s: float = 45.0):
        self._memory = memory
        self._bus = bus
        self._timeout = ask_timeout_s
        self._pending: dict[str, asyncio.Future[bool]] = {}
        # One asker at a time — two tools racing would talk over each other.
        self._ask_lock = asyncio.Lock()
        # Set by the daemon so the prompt is spoken and answerable out loud,
        # rather than requiring you to look at the tab.
        self.ask_aloud: Callable[[str], Awaitable[bool]] | None = None
        bus.on(EV_CAPABILITY_REPLY, self._on_reply)

    async def _on_reply(self, event: Event) -> None:
        request_id = event.payload.get("id")
        future = self._pending.pop(request_id, None)
        if future and not future.done():
            future.set_result(bool(event.payload.get("granted")))

    def is_granted(self, capability_id: str) -> bool:
        rows = self._memory.execute(
            "SELECT granted, revoked_at FROM capabilities WHERE id=?", (capability_id,)
        )
        return bool(rows) and bool(rows[0]["granted"]) and rows[0]["revoked_at"] is None

    def grant(self, capability_id: str, title: str = "") -> None:
        self._memory.execute(
            "INSERT INTO capabilities (id, title, granted, granted_at)"
            " VALUES (?,?,1,?)"
            " ON CONFLICT(id) DO UPDATE SET granted=1, granted_at=?, revoked_at=NULL",
            (capability_id, title or capability_id, time.time(), time.time()),
        )
        log.info("capability granted: %s", capability_id)

    def revoke(self, capability_id: str) -> None:
        """Backs 'stop letting yourself send messages'."""
        self._memory.execute(
            "UPDATE capabilities SET granted=0, revoked_at=? WHERE id=?",
            (time.time(), capability_id),
        )
        log.info("capability revoked: %s", capability_id)

    def granted_list(self) -> list[str]:
        return [
            r["id"] for r in self._memory.execute(
                "SELECT id FROM capabilities WHERE granted=1 AND revoked_at IS NULL"
            )
        ]

    def _bump(self, capability_id: str) -> None:
        self._memory.execute(
            "UPDATE capabilities SET use_count = use_count + 1 WHERE id=?", (capability_id,)
        )

    async def require(self, capability_id: str, *, title: str = "") -> None:
        """Gate a tool call. Returns silently if allowed, raises if refused.

        Note what is *absent*: no arguments are accepted, so there is no way for
        a caller to accidentally make this per-recipient or per-URL.
        """
        if self.is_granted(capability_id):
            self._bump(capability_id)
            return

        async with self._ask_lock:
            # Re-check — an earlier waiter may have just been granted this.
            if self.is_granted(capability_id):
                self._bump(capability_id)
                return

            request_id = uuid.uuid4().hex
            prompt = PROMPTS.get(capability_id, f"Okay if I use {capability_id}?")
            future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future

            await self._bus.emit(
                EV_CAPABILITY, id=request_id, capability=capability_id,
                title=title or capability_id, prompt=prompt,
            )

            # Ask out loud and by button at the same time; whichever answers
            # first wins, and the other is discarded.
            waiters = [asyncio.ensure_future(asyncio.wait_for(future, self._timeout))]
            if self.ask_aloud is not None:
                waiters.append(asyncio.ensure_future(self.ask_aloud(prompt)))

            done, pending = await asyncio.wait(
                waiters, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()

            try:
                granted = next(iter(done)).result()
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._pending.pop(request_id, None)
                log.warning("capability prompt timed out: %s", capability_id)
                raise CapabilityDenied(capability_id) from None
            finally:
                self._pending.pop(request_id, None)

        if not granted:
            raise CapabilityDenied(capability_id)

        self.grant(capability_id, title)
        self._bump(capability_id)
