"""Tool definitions handed to the model.

Descriptions are written for the model, not for you — they're what it reads
when deciding whether to call something, so they say when to use a tool rather
than what it does internally.

Every tool that touches the outside world names a capability. The first time
each one fires, JARVIS asks once and then never again — including for different
arguments. Approving `chrome.open_tab` approves every URL forever.
"""

from __future__ import annotations

import logging
from typing import Any

from ..agent.capabilities import CapabilityLedger
from ..agent.registry import Registry
from ..memory.db import Memory
from . import mac
from .schedule import ScheduleResolver

log = logging.getLogger("jarvis.tools")

NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}}


def register_all(registry: Registry, memory: Memory, schedule: ScheduleResolver,
                 ledger: CapabilityLedger) -> None:

    # ---- time and schedule ----------------------------------------------

    @registry.register(
        "get_time",
        "The current date and time. Use whenever he asks what time or day it is, "
        "or when you need to reason about how soon something is.",
        NO_ARGS,
    )
    async def get_time() -> dict[str, Any]:
        now = schedule.now()
        return {
            "iso": now.isoformat(),
            "spoken": now.strftime("%A, %B %-d, %-I:%M %p"),
            "day_type": schedule.day_type(),
        }

    @registry.register(
        "get_schedule_today",
        "Which classes he has today and in what order, based on the A/B rotation. "
        "Use for 'what do I have today', 'what's my next class', 'am I on an A day'.",
        NO_ARGS,
    )
    async def get_schedule_today() -> dict[str, Any]:
        day_type = schedule.day_type()
        if day_type is None:
            return {"school": False, "next_school_day": str(schedule.next_school_day())}
        # Populated from Notion once the token is set; empty until then.
        courses = _COURSE_CACHE.get("rows", [])
        current, when = schedule.current_or_next_class(courses)
        return {
            "school": True,
            "day_type": day_type,
            "classes": schedule.classes_today(courses),
            "current_or_next": current,
            "relation": when,
            "summary": schedule.describe_today(courses),
        }

    # ---- apps ------------------------------------------------------------

    @registry.register(
        "open_app",
        "Open a Mac application by name, e.g. 'Notion', 'Spotify', 'Google Chrome'.",
        {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Exact app name"}},
            "required": ["name"],
        },
        capability="apps.open",
    )
    async def open_app(name: str) -> dict[str, Any]:
        return await mac.open_app(name)

    @registry.register(
        "list_running_apps",
        "Which apps are open right now. Use before assuming something is running.",
        NO_ARGS,
        capability="apps.open",
    )
    async def list_running_apps() -> dict[str, Any]:
        return {"apps": await mac.running_apps()}

    # ---- chrome ----------------------------------------------------------

    @registry.register(
        "open_tab",
        "Open a URL in a new Chrome tab. Use when you know the exact address. "
        "For a topic rather than an address, use search_web instead.",
        {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        capability="chrome.open_tab",
    )
    async def open_tab(url: str) -> dict[str, Any]:
        return await mac.open_tab(url)

    @registry.register(
        "search_web",
        "Open a Google search. Use when he wants information you don't have and "
        "there's no specific site to go to.",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        capability="chrome.open_tab",
    )
    async def search_web(query: str) -> dict[str, Any]:
        return await mac.search(query)

    @registry.register(
        "list_open_tabs",
        "Every Chrome tab open right now, with titles and URLs. Use before "
        "opening something new, in case it's already open.",
        NO_ARGS,
        capability="chrome.read_tabs",
    )
    async def list_open_tabs() -> dict[str, Any]:
        tabs = await mac.list_tabs()
        for tab in tabs:
            memory.observe("tab", app="Google Chrome", title=tab.get("title"),
                           url=tab.get("url"), active=tab.get("active", False))
        return {"tabs": tabs}

    @registry.register(
        "find_tab",
        "Search tabs he has open now or had open recently, by topic or site name. "
        "Use for 'open the tab about X' — check here before opening a new one.",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        capability="chrome.read_tabs",
    )
    async def find_tab(query: str) -> dict[str, Any]:
        live = await mac.list_tabs()
        lowered = query.lower()
        matches = [
            tab for tab in live
            if lowered in (tab.get("title") or "").lower()
            or lowered in (tab.get("url") or "").lower()
        ]
        if matches:
            return {"open_now": True, "matches": matches}
        remembered = memory.search_tabs(query)
        return {
            "open_now": False,
            "previously_seen": [dict(row) for row in remembered],
        }

    @registry.register(
        "focus_tab",
        "Switch to a tab that's already open. Get window and index from "
        "list_open_tabs or find_tab.",
        {
            "type": "object",
            "properties": {
                "window": {"type": "integer"},
                "index": {"type": "integer"},
            },
            "required": ["window", "index"],
        },
        capability="chrome.switch_tab",
    )
    async def focus_tab(window: int, index: int) -> dict[str, Any]:
        return await mac.focus_tab(window, index)

    # ---- memory ----------------------------------------------------------

    @registry.register(
        "remember",
        "Store something about him for later — a preference, a routine, a name, "
        "how he likes something done. Use whenever he tells you a lasting fact.",
        {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "short snake_case label"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    )
    async def remember(key: str, value: str) -> dict[str, Any]:
        memory.remember(key, value, source="told")
        return {"ok": True}

    @registry.register(
        "recall",
        "Look up something you stored earlier with remember.",
        {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    )
    async def recall(key: str) -> dict[str, Any]:
        return {"key": key, "value": memory.recall(key)}

    # ---- permissions -----------------------------------------------------

    @registry.register(
        "revoke_permission",
        "Take away a permission he previously gave you. Use when he says things "
        "like 'stop sending messages' or 'don't open tabs anymore'.",
        {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": "e.g. messages.send, chrome.open_tab, apps.open",
                }
            },
            "required": ["capability"],
        },
    )
    async def revoke_permission(capability: str) -> dict[str, Any]:
        ledger.revoke(capability)
        return {"ok": True, "revoked": capability}

    @registry.register(
        "list_permissions",
        "What he's already allowed you to do.",
        NO_ARGS,
    )
    async def list_permissions() -> dict[str, Any]:
        return {"granted": ledger.granted_list()}

    log.info("registered %d tools: %s", len(registry.names()), ", ".join(registry.names()))


# Refreshed from Notion at startup and on a timer once NOTION_TOKEN is set.
# Until then get_schedule_today still reports the correct A/B letter, just
# without course names attached.
_COURSE_CACHE: dict[str, list[dict[str, Any]]] = {"rows": []}


def set_courses(rows: list[dict[str, Any]]) -> None:
    _COURSE_CACHE["rows"] = rows
