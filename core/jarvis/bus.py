"""Event bus between the Python daemon and the Electron UI.

This is also the boundary you cared about most: everything JARVIS *thinks* goes
to the log, and only what it *says* crosses this bus as a `speech` event. The
UI never renders reasoning, and Kokoro never voices it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

import websockets
from websockets.asyncio.server import ServerConnection, serve

log = logging.getLogger("jarvis.bus")


@dataclass
class Event:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def json(self) -> str:
        return json.dumps(asdict(self))


# Daemon -> UI
EV_STATE = "state"                  # {state: idle|listening|thinking|speaking|offline}
EV_LEVEL = "level"                  # {rms: 0.0-1.0} drives glob morph amplitude
EV_TRANSCRIPT = "transcript"        # {text, final: bool}
EV_SPEECH = "speech"                # {text} — the only thing ever spoken aloud
EV_CAPABILITY = "capability"        # {id, title, detail} first-use approval prompt
EV_PENDING_SEND = "pending_send"    # {id, summary, seconds} 3s cancel window
EV_TOAST = "toast"                  # {level, text} non-spoken status
EV_ONBOARD_FORM = "onboard_form"    # {id, title, intro, fields[]} fill this in

# UI -> daemon
EV_HOTKEY = "hotkey"
EV_CANCEL = "cancel"
EV_CAPABILITY_REPLY = "capability_reply"   # {id, granted: bool}
EV_TEXT_INPUT = "text_input"               # {text} typed instead of spoken
EV_ONBOARD_SUBMIT = "onboard_submit"       # {id, values: {key: value}}

Handler = Callable[[Event], Awaitable[None]]


class Bus:
    def __init__(self, port: int):
        self._port = port
        self._clients: set[ServerConnection] = set()
        self._handlers: dict[str, list[Handler]] = {}
        self._server = None
        # Replayed to any UI that connects late, so a freshly launched window
        # immediately shows the right colour instead of defaulting to idle.
        self._sticky: dict[str, Event] = {}

    def on(self, event_type: str, handler: Handler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def emit(self, type: str, **payload: Any) -> None:
        event = Event(type, payload)
        if type in (EV_STATE,):
            self._sticky[type] = event
        if not self._clients:
            return
        message = event.json()
        dead: set[ServerConnection] = set()
        for client in self._clients:
            try:
                await client.send(message)
            except websockets.ConnectionClosed:
                dead.add(client)
        self._clients -= dead

    async def _dispatch(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            try:
                await handler(event)
            except Exception:
                log.exception("handler for %s failed", event.type)

    async def _serve_client(self, conn: ServerConnection) -> None:
        self._clients.add(conn)
        log.info("ui connected (%d total)", len(self._clients))
        try:
            for event in self._sticky.values():
                await conn.send(event.json())
            async for raw in conn:
                try:
                    data = json.loads(raw)
                    await self._dispatch(Event(data["type"], data.get("payload", {})))
                except (json.JSONDecodeError, KeyError):
                    log.warning("malformed message from ui: %r", raw[:200])
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(conn)
            log.info("ui disconnected (%d left)", len(self._clients))

    async def start(self) -> None:
        self._server = await serve(self._serve_client, "127.0.0.1", self._port)
        log.info("bus listening on ws://127.0.0.1:%d", self._port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def run_forever(self) -> None:
        await self.start()
        await asyncio.Future()
