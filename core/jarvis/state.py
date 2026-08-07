"""The four states that drive the glob's colour, and the transitions between them.

You specified: blue when idle, smoothly green when it hears you, yellow while
thinking, multicolour while responding. The colours themselves live in the UI
(ui/src/lib/globStates.ts); this is the authority on *which* state is current.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Callable


class AgentState(str, Enum):
    IDLE = "idle"           # blue — waiting for the wake word
    LISTENING = "listening"  # green — capturing your speech
    THINKING = "thinking"    # yellow — routing, calling tools, generating
    SPEAKING = "speaking"    # multicolour — Kokoro is playing
    OFFLINE = "offline"      # grey — daemon down or a subsystem failed


# Guards against nonsense transitions (e.g. SPEAKING straight back to LISTENING
# without passing through IDLE), which would make the glob flicker.
_ALLOWED: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.LISTENING, AgentState.THINKING, AgentState.SPEAKING,
                      AgentState.OFFLINE},
    AgentState.LISTENING: {AgentState.THINKING, AgentState.IDLE, AgentState.OFFLINE},
    AgentState.THINKING: {AgentState.SPEAKING, AgentState.IDLE, AgentState.LISTENING,
                          AgentState.OFFLINE},
    AgentState.SPEAKING: {AgentState.IDLE, AgentState.LISTENING, AgentState.THINKING,
                          AgentState.OFFLINE},
    AgentState.OFFLINE: {AgentState.IDLE},
}


class StateMachine:
    def __init__(self, on_change: Callable[[AgentState, AgentState], None] | None = None):
        self._state = AgentState.OFFLINE
        self._since = time.monotonic()
        self._on_change = on_change
        self._lock = asyncio.Lock()

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def elapsed(self) -> float:
        """Seconds in the current state — used to decide if it's safe to interrupt."""
        return time.monotonic() - self._since

    async def transition(self, to: AgentState) -> bool:
        async with self._lock:
            if to == self._state:
                return True
            if to not in _ALLOWED[self._state]:
                return False
            previous, self._state = self._state, to
            self._since = time.monotonic()
            if self._on_change:
                self._on_change(previous, to)
            return True

    def can_interrupt(self) -> bool:
        """Proactive speech only cuts in when nothing is already happening."""
        return self._state is AgentState.IDLE
