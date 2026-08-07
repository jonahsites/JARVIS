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
    IDLE = "idle"            # blue — waiting for the wake word
    LISTENING = "listening"  # green — capturing your speech
    THINKING = "thinking"    # amber — routing, calling tools, generating
    SPEAKING = "speaking"    # multicolour — Kokoro is playing
    WORKING = "working"      # violet — busy in the background, not with you
    OFFLINE = "offline"      # grey — daemon down or a subsystem failed


# WORKING is deliberately a sibling of IDLE rather than of THINKING. THINKING
# means "you asked me something and I'm on it"; WORKING means "I'm crawling
# Notion and you can still interrupt me at any moment". Everything that works
# from idle works from working.
_ALLOWED: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.LISTENING, AgentState.THINKING, AgentState.SPEAKING,
                      AgentState.WORKING, AgentState.OFFLINE},
    AgentState.LISTENING: {AgentState.THINKING, AgentState.IDLE, AgentState.WORKING,
                           AgentState.OFFLINE},
    AgentState.THINKING: {AgentState.SPEAKING, AgentState.IDLE, AgentState.LISTENING,
                          AgentState.WORKING, AgentState.OFFLINE},
    AgentState.SPEAKING: {AgentState.IDLE, AgentState.LISTENING, AgentState.THINKING,
                          AgentState.WORKING, AgentState.OFFLINE},
    AgentState.WORKING: {AgentState.IDLE, AgentState.LISTENING, AgentState.THINKING,
                         AgentState.SPEAKING, AgentState.OFFLINE},
    AgentState.OFFLINE: {AgentState.IDLE, AgentState.WORKING},
}

# States where JARVIS isn't engaged with you, so the wake word should be live
# and a proactive nudge is allowed.
AVAILABLE = {AgentState.IDLE, AgentState.WORKING}


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
        """Proactive speech only cuts in when you aren't mid-exchange.

        Background work doesn't count as being busy — you can't see it and
        it isn't a conversation.
        """
        return self._state in AVAILABLE
