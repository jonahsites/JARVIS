"""Notion tools exposed to the model."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from ...agent.registry import Registry
from ..schedule import ScheduleResolver
from .assignments import IMPORTANCE_WEIGHT, TYPES, Assignments
from .notes import NoteCrawler, NoteIndex

log = logging.getLogger("jarvis.notion.tools")

NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}}

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday"]


def parse_spoken_date(raw: str, today: date) -> date | None:
    """Accept an ISO date, or the loose phrasings speech actually produces.

    The model is told to send ISO and usually does, but "friday" and "tomorrow"
    slip through often enough to be worth handling rather than dropping the
    due date silently.
    """
    if not raw:
        return None
    text = raw.strip().lower()

    try:
        return datetime.fromisoformat(text.replace("z", "+00:00")).date()
    except ValueError:
        pass

    if text in ("today", "tonight"):
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    if text in ("next week",):
        return today + timedelta(days=7)

    next_week = text.startswith("next ")
    bare = text.removeprefix("next ").removeprefix("this ").strip()
    if bare in _WEEKDAYS:
        target = _WEEKDAYS.index(bare)
        ahead = (target - today.weekday()) % 7
        if ahead == 0:
            ahead = 7
        if next_week and ahead < 7:
            ahead += 7
        return today + timedelta(days=ahead)

    log.warning("could not parse date %r", raw)
    return None


def register_notion(registry: Registry, assignments: Assignments,
                    index: NoteIndex, crawler: NoteCrawler,
                    schedule: ScheduleResolver) -> None:

    def offline() -> dict[str, Any] | None:
        """A clear sentence the model can repeat, instead of a stack trace.

        Without this the tool raises and the model gets 'tool_failed:
        NOTION_TOKEN is not set', which it tends to relay as gibberish.
        """
        if assignments.enabled:
            return None
        return {
            "error": "notion_not_connected",
            "say": "I'm not connected to Notion yet — the token isn't set up.",
        }

    # ---- assignments -----------------------------------------------------

    @registry.register(
        "get_upcoming_work",
        "What he has due, ranked by how urgent and important it is. Use for "
        "'what do I have', 'what's due', 'what should I work on', 'am I behind'.",
        {
            "type": "object",
            "properties": {
                "within_days": {"type": "integer",
                                "description": "How far ahead to look. Default 14."},
                "course": {"type": "string",
                           "description": "Optional — limit to one class."},
            },
        },
        capability="notion.read",
    )
    async def get_upcoming_work(within_days: int = 14,
                                course: str | None = None) -> dict[str, Any]:
        if (down := offline()):
            return down
        today = schedule.now().date()
        items = await assignments.upcoming(today=today, within_days=within_days)

        if course:
            matched = assignments.resolve_course(course)
            if matched:
                items = [a for a in items if a.course_id == matched.id]

        overdue = [a for a in items if a.days_until(today) is not None
                   and a.days_until(today) < 0]
        return {
            "count": len(items),
            "overdue_count": len(overdue),
            "assignments": [a.as_dict(today) for a in items],
        }

    @registry.register(
        "add_assignment",
        "Add an assignment to Notion. Always pass an ISO date (YYYY-MM-DD) for "
        "due_date — call get_time first if you need to work out what 'Friday' "
        "means. course can be said loosely ('calc', 'apes') and will be matched.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "course": {"type": "string",
                           "description": "e.g. 'calc', 'AP Environmental Science'"},
                "due_date": {"type": "string", "description": "YYYY-MM-DD"},
                "type": {"type": "string", "enum": TYPES},
                "importance": {"type": "string",
                               "enum": list(IMPORTANCE_WEIGHT)},
                "effort_hours": {"type": "number"},
                "description": {"type": "string"},
            },
            "required": ["name"],
        },
        capability="notion.create_assignment",
    )
    async def add_assignment(name: str, course: str | None = None,
                             due_date: str | None = None,
                             type: str | None = None,
                             importance: str | None = None,
                             effort_hours: float | None = None,
                             description: str | None = None) -> dict[str, Any]:
        if (down := offline()):
            return down
        today = schedule.now().date()
        due = parse_spoken_date(due_date, today) if due_date else None

        result = await assignments.create(
            name=name, course=course, due=due, type=type,
            importance=importance, effort_hours=effort_hours,
            description=description,
        )
        # Surfaced so the model can mention it rather than silently filing the
        # assignment under no class at all.
        if course and not result["course_matched"]:
            result["warning"] = f"No class matched '{course}' — added without one."
        return result

    @registry.register(
        "update_assignment",
        "Change an assignment's status or progress. Get the id from "
        "get_upcoming_work. Use when he says something is done or started.",
        {
            "type": "object",
            "properties": {
                "assignment_id": {"type": "string"},
                "status": {"type": "string",
                           "enum": ["Not started", "In progress", "Completed",
                                    "Submitted", "Incomplete"]},
                "progress": {"type": "number", "description": "0-100"},
            },
            "required": ["assignment_id"],
        },
        capability="notion.update_assignment",
    )
    async def update_assignment(assignment_id: str, status: str | None = None,
                                progress: float | None = None) -> dict[str, Any]:
        if (down := offline()):
            return down
        out: dict[str, Any] = {"ok": True}
        if status:
            out |= await assignments.set_status(assignment_id, status)
        if progress is not None:
            out |= await assignments.set_progress(assignment_id, progress)
        return out

    # ---- notes -----------------------------------------------------------

    @registry.register(
        "search_notes",
        "Search his class notes by what they're ABOUT, not just their titles — "
        "this searches the full text of every note. Use for 'find my notes on "
        "X', 'what did we cover about Y', 'pull up the notes about Z'.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the topic"},
                "course": {"type": "string", "description": "optional class filter"},
            },
            "required": ["query"],
        },
        capability="notion.read",
    )
    async def search_notes(query: str, course: str | None = None) -> dict[str, Any]:
        course_name = None
        if course:
            matched = assignments.resolve_course(course)
            course_name = matched.name if matched else course

        hits = index.search(query, course=course_name)
        if not hits and index.count() == 0:
            return {"results": [], "note": "The note index is empty — "
                                           "run reindex_notes first."}
        return {
            "results": [
                {
                    "title": hit["title"],
                    "where": " — ".join(
                        b for b in (hit["course"], hit["chapter"] or hit["unit"]) if b
                    ),
                    "excerpt": hit["excerpt"],
                    "url": hit["url"],
                }
                for hit in hits
            ]
        }

    @registry.register(
        "open_note",
        "Open a note in Chrome. Pass the url from search_notes.",
        {
            "type": "object",
            "properties": {"url": {"type": "string"}, "title": {"type": "string"}},
            "required": ["url"],
        },
        capability="chrome.open_tab",
    )
    async def open_note(url: str, title: str = "") -> dict[str, Any]:
        from ..mac import open_tab

        result = await open_tab(url)
        result["title"] = title
        return result

    @registry.register(
        "list_notes_for_class",
        "Every note he has for one class, so you can tell him what's there.",
        {
            "type": "object",
            "properties": {"course": {"type": "string"}},
            "required": ["course"],
        },
        capability="notion.read",
    )
    async def list_notes_for_class(course: str) -> dict[str, Any]:
        matched = assignments.resolve_course(course)
        name = matched.name if matched else course
        return {"course": name, "notes": index.by_course(name)}

    @registry.register(
        "reindex_notes",
        "Re-scan Notion for new or changed notes. Use when he says he just "
        "added notes and you can't find them.",
        NO_ARGS,
        capability="notion.read",
    )
    async def reindex_notes() -> dict[str, Any]:
        if (down := offline()):
            return down
        written = await crawler.crawl()
        return {"ok": True, "updated": written, "total_indexed": index.count()}

    # ---- schedule --------------------------------------------------------

    @registry.register(
        "get_class_schedule",
        "His classes today or on a given day, in period order, using the A/B "
        "rotation. Use for 'what do I have today', 'what's my next class', "
        "'when is calc'.",
        {
            "type": "object",
            "properties": {
                "day": {"type": "string", "description": "YYYY-MM-DD, or omit for today"},
            },
        },
        capability="notion.read",
    )
    async def get_class_schedule(day: str | None = None) -> dict[str, Any]:
        today = schedule.now().date()
        target = parse_spoken_date(day, today) if day else today

        day_type = schedule.day_type(target)
        if day_type is None:
            return {
                "date": target.isoformat(), "school": False,
                "next_school_day": schedule.next_school_day(target).isoformat(),
            }

        rows = [c.as_dict() for c in assignments.courses]
        classes = schedule.classes_today(rows) if target == today else [
            c for c in rows if c.get("day_type") == day_type and c.get("class_time")
        ]
        current, relation = (schedule.current_or_next_class(rows)
                             if target == today else (None, "other_day"))

        return {
            "date": target.isoformat(),
            "school": True,
            "day_type": day_type,
            "classes": classes,
            "current_or_next": current,
            "relation": relation,
        }
