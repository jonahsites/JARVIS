"""The cancel window.

You chose: JARVIS says what it's about to send, then sends unless you stop it.
Not a yes/no gate — no friction on the common case — but a real chance to catch
a misheard name or a garbled sentence before it reaches a person.

    JARVIS  "Texting Mom: running twenty minutes late."
    (3 seconds)
    you     "stop"        -> nothing is sent
    silence               -> it goes

Cancelling works three ways: saying stop, the button in the UI, or the wake
word (if you're talking to it again, you clearly want its attention). All three
land on `cancel()`.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..bus import EV_PENDING_SEND, EV_TOAST, Bus

log = logging.getLogger("jarvis.outbox")


@dataclass
class Pending:
    id: str
    summary: str
    task: asyncio.Task


class Outbox:
    def __init__(self, bus: Bus, window_s: float = 3.0):
        self._bus = bus
        self._window = window_s
        self._pending: Pending | None = None

    @property
    def has_pending(self) -> bool:
        return self._pending is not None and not self._pending.task.done()

    def cancel(self) -> bool:
        """Stop whatever is queued. True if something was actually stopped."""
        if not self.has_pending:
            return False
        assert self._pending is not None
        log.info("cancelled: %s", self._pending.summary)
        self._pending.task.cancel()
        self._pending = None
        return True

    async def queue(self, *, summary: str,
                    send: Callable[[], Awaitable[dict[str, Any]]],
                    speak: Callable[[str], Awaitable[Any]]) -> dict[str, Any]:
        """Announce, wait out the window, then send unless cancelled.

        Returns once the outcome is known, so the agent can report what
        actually happened rather than guessing.
        """
        # A second send while one is queued replaces it — you changed your mind.
        if self.has_pending:
            self.cancel()

        request_id = uuid.uuid4().hex
        await self._bus.emit(EV_PENDING_SEND, id=request_id, summary=summary,
                             seconds=self._window)
        await speak(summary)

        async def run() -> dict[str, Any]:
            await asyncio.sleep(self._window)
            return await send()

        task = asyncio.create_task(run())
        self._pending = Pending(request_id, summary, task)

        try:
            result = await task
        except asyncio.CancelledError:
            await self._bus.emit(EV_TOAST, level="info", text="Cancelled.")
            return {"ok": False, "cancelled": True,
                    "say": "Okay, didn't send it."}
        finally:
            if self._pending and self._pending.id == request_id:
                self._pending = None

        return result
