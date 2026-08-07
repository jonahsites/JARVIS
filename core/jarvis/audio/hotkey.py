"""Global hotkey — the fallback for when the wake word mishears you.

Lives in the daemon rather than the UI so it works whether or not the browser
tab is open. Needs Accessibility permission on macOS:
System Settings -> Privacy & Security -> Accessibility -> add your terminal.
`jarvis doctor` checks for this.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

log = logging.getLogger("jarvis.hotkey")


class GlobalHotkey:
    def __init__(self, combo: str, on_press: Callable[[], Awaitable[None]]):
        self.combo = combo
        self._on_press = on_press
        self._listener = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> bool:
        """Returns False if the hotkey couldn't be registered (usually perms)."""
        try:
            from pynput import keyboard
        except Exception as exc:
            log.warning("pynput unavailable, hotkey disabled: %s", exc)
            return False

        self._loop = asyncio.get_running_loop()

        def fire() -> None:
            # pynput's thread — bounce into the event loop.
            if self._loop is None:
                return
            asyncio.run_coroutine_threadsafe(self._on_press(), self._loop)

        try:
            self._listener = keyboard.GlobalHotKeys({self.combo: fire})
            self._listener.start()
            log.info("hotkey registered: %s", self.combo)
            return True
        except Exception as exc:
            log.warning(
                "could not register hotkey %s — grant Accessibility permission "
                "to your terminal and restart (%s)", self.combo, exc,
            )
            return False

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
