"""Logging, with a hard wall between reasoning and speech.

You were explicit: you never want to hear "alright, the user wants me to do
this, let me open that". So reasoning has exactly one destination — this log —
and there is no code path from a log record to the TTS engine.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

from .config import VAR_DIR

LOG_PATH = VAR_DIR / "jarvis.log"


def setup(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)-22s %(message)s")
    )

    console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    console.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    logging.basicConfig(level=level, handlers=[file_handler, console], force=True)

    # These are chatty and tell you nothing you want at INFO.
    for noisy in ("websockets", "httpx", "urllib3", "numba", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def thinking(logger: logging.Logger, message: str, *args) -> None:
    """Model reasoning. Goes to the log and nowhere else — never spoken."""
    logger.debug("[think] " + message, *args)
