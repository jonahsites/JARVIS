"""macOS control — apps and Chrome.

Chrome is driven by AppleScript rather than the DevTools protocol on purpose:
the debug-port approach requires relaunching Chrome with a flag, which throws
away your logged-in session. AppleScript talks to the browser you already have
open. The tradeoff is that it needs Automation permission the first time, which
macOS prompts for by itself.

The Chrome extension in extension/ is the richer path (live tab events, no
per-call AppleScript latency). This module is what works with zero install.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any

log = logging.getLogger("jarvis.mac")


async def osascript(script: str, *, language: str = "AppleScript") -> str:
    """Run a script, return stdout. Raises RuntimeError with stderr on failure."""
    if shutil.which("osascript") is None:
        raise RuntimeError("osascript not found — this module is macOS only")

    args = ["osascript"]
    if language == "JavaScript":
        args += ["-l", "JavaScript"]
    args += ["-e", script]

    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode().strip() or "osascript failed")
    return stdout.decode().strip()


# ---- apps ----------------------------------------------------------------

async def open_app(name: str) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        "open", "-a", name,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        return {"ok": False, "error": stderr.decode().strip() or f"couldn't open {name}"}
    return {"ok": True, "opened": name}


async def quit_app(name: str) -> dict[str, Any]:
    await osascript(f'tell application "{name}" to quit')
    return {"ok": True, "quit": name}


async def frontmost_app() -> str:
    return await osascript(
        'tell application "System Events" to get name of first application process '
        'whose frontmost is true'
    )


async def running_apps() -> list[str]:
    raw = await osascript(
        'tell application "System Events" to get name of every application process '
        'whose background only is false'
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---- chrome --------------------------------------------------------------

_LIST_TABS_JS = """
const chrome = Application('Google Chrome');
if (!chrome.running()) { JSON.stringify([]); }
else {
  const out = [];
  const windows = chrome.windows();
  for (let w = 0; w < windows.length; w++) {
    const tabs = windows[w].tabs();
    const activeIndex = windows[w].activeTabIndex();
    for (let t = 0; t < tabs.length; t++) {
      out.push({
        window: w, index: t + 1,
        title: tabs[t].title(), url: tabs[t].url(),
        active: (t + 1) === activeIndex,
      });
    }
  }
  JSON.stringify(out);
}
"""


async def list_tabs() -> list[dict[str, Any]]:
    try:
        raw = await osascript(_LIST_TABS_JS, language="JavaScript")
        return json.loads(raw) if raw else []
    except Exception as exc:
        log.warning("could not list chrome tabs: %s", exc)
        return []


async def open_tab(url: str) -> dict[str, Any]:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    safe = url.replace('"', '%22')
    await osascript(
        f'tell application "Google Chrome"\n'
        f'  if (count of windows) = 0 then make new window\n'
        f'  tell window 1 to make new tab with properties {{URL:"{safe}"}}\n'
        f'  activate\n'
        f'end tell'
    )
    return {"ok": True, "url": url}


async def focus_tab(window: int, index: int) -> dict[str, Any]:
    await osascript(
        f'tell application "Google Chrome"\n'
        f'  set targetWindow to window {window + 1}\n'
        f'  set active tab index of targetWindow to {index}\n'
        f'  set index of targetWindow to 1\n'
        f'  activate\n'
        f'end tell'
    )
    return {"ok": True, "focused": {"window": window, "index": index}}


async def close_tab(window: int, index: int) -> dict[str, Any]:
    await osascript(
        f'tell application "Google Chrome" to close tab {index} of window {window + 1}'
    )
    return {"ok": True}


async def search(query: str) -> dict[str, Any]:
    from urllib.parse import quote_plus

    return await open_tab(f"https://www.google.com/search?q={quote_plus(query)}")


# ---- notifications -------------------------------------------------------

async def notify(title: str, message: str) -> None:
    """Silent channel — used for proactive nudges that don't warrant speech."""
    safe_title = title.replace('"', "'")
    safe_message = message.replace('"', "'")
    try:
        await osascript(
            f'display notification "{safe_message}" with title "{safe_title}"'
        )
    except Exception as exc:
        log.debug("notification failed: %s", exc)
