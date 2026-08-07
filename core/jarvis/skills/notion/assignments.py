"""Assignments and courses.

One thing worth knowing: your Assignments database computes Priority Score,
Start By, Countdown, overdue, due-today and due-this-week as Notion formulas,
and Notion's API refuses to return formula *values* through the query endpoint
(they're listed under notAvailableInQuerySql). So JARVIS recomputes that logic
here rather than reading it.

That means the numbers can drift from what your Notion dashboard shows. The
ranking below uses importance x urgency the same way yours appears to, but if
your formula weights differ, adjust IMPORTANCE_WEIGHT and the urgency curve —
this is the only place it's defined.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from .blocks import page_url, plain_property, title_of
from .client import DB, NotionClient

log = logging.getLogger("jarvis.assignments")

IMPORTANCE_WEIGHT = {"Critical": 4.0, "High": 3.0, "Medium": 2.0, "Low": 1.0}
DONE_STATUSES = {"Submitted", "Completed"}

# From the Type select in your Assignments database.
TYPES = [
    "Research", "Reading", "Extra Credit", "Quiz", "Case Study", "Journal",
    "Group Work", "Presentation", "Project", "Report", "Worksheet", "Essay",
    "Homework", "Draft", "Lab", "Assignment", "Test", "Practice", "Reminder",
    "MIDTERM", "Meeting",
]


@dataclass
class Course:
    id: str
    name: str
    day_type: str | None = None
    class_time: str | None = None
    location: str | None = None
    professor: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "day_type": self.day_type,
            "class_time": self.class_time, "location": self.location,
            "professor": self.professor,
        }


@dataclass
class Assignment:
    id: str
    name: str
    course: str = ""
    course_id: str = ""
    type: str | None = None
    status: str | None = None
    importance: str | None = None
    due: date | None = None
    effort_hours: float | None = None
    progress: float | None = None
    description: str = ""
    url: str = ""
    relations: dict[str, list[str]] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.status in DONE_STATUSES

    def days_until(self, today: date) -> int | None:
        return (self.due - today).days if self.due else None

    def urgency(self, today: date) -> float:
        """Steep near the deadline, flat far out — matches how panic works."""
        days = self.days_until(today)
        if days is None:
            return 0.5
        if days < 0:
            return 6.0          # overdue outranks everything
        if days == 0:
            return 5.0
        if days == 1:
            return 4.0
        if days <= 3:
            return 2.5
        if days <= 7:
            return 1.5
        return 0.75

    def priority(self, today: date) -> float:
        if self.done:
            return 0.0
        weight = IMPORTANCE_WEIGHT.get(self.importance or "", 2.0)
        score = weight * self.urgency(today)
        # Big things need starting sooner than small ones.
        if self.effort_hours:
            score *= 1 + min(self.effort_hours, 10) / 20
        if self.progress:
            score *= max(0.3, 1 - self.progress / 100)
        return round(score, 2)

    def start_by(self, today: date) -> date | None:
        """When to begin, so a long assignment doesn't ambush you."""
        if not self.due or not self.effort_hours:
            return self.due
        lead = max(1, int(self.effort_hours / 2))
        return self.due - timedelta(days=lead)

    def spoken(self, today: date) -> str:
        """One assignment, phrased the way a person would say it out loud."""
        when = "no due date"
        days = self.days_until(today)
        if days is not None:
            if days < -1:
                when = f"{abs(days)} days overdue"
            elif days == -1:
                when = "due yesterday"
            elif days == 0:
                when = "due today"
            elif days == 1:
                when = "due tomorrow"
            elif days <= 6:
                when = f"due {self.due.strftime('%A')}"
            else:
                when = f"due {self.due.strftime('%B %-d')}"
        course = f" for {self.course}" if self.course else ""
        return f"{self.name}{course}, {when}"

    def as_dict(self, today: date) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "course": self.course,
            "type": self.type, "status": self.status, "importance": self.importance,
            "due": self.due.isoformat() if self.due else None,
            "days_until": self.days_until(today),
            "priority": self.priority(today),
            "start_by": (self.start_by(today).isoformat()
                         if self.start_by(today) else None),
            "effort_hours": self.effort_hours, "progress": self.progress,
            "url": self.url, "spoken": self.spoken(today),
        }


def _parse_date(value: Any) -> date | None:
    if not value or not isinstance(value, dict):
        return None
    start = value.get("start")
    if not start:
        return None
    try:
        return datetime.fromisoformat(start.replace("Z", "+00:00")).date()
    except ValueError:
        return None


class Assignments:
    def __init__(self, client: NotionClient):
        self._client = client
        self._courses: dict[str, Course] = {}

    @property
    def enabled(self) -> bool:
        return self._client.enabled

    # ---- courses ---------------------------------------------------------

    async def load_courses(self) -> list[Course]:
        courses: dict[str, Course] = {}
        async for row in self._client.query(DB.COURSES):
            course = Course(
                id=row["id"],
                name=title_of(row),
                day_type=plain_property(row, "Day Type"),
                class_time=plain_property(row, "Class Times"),
                location=plain_property(row, "Location"),
                professor=plain_property(row, "Professor"),
            )
            if course.name:
                courses[course.id] = course
        self._courses = courses
        log.info("loaded %d courses", len(courses))
        return list(courses.values())

    def resolve_course(self, spoken: str) -> Course | None:
        """Match a course by how you'd actually say it.

        'calc', 'AP Calc', 'calculus' all need to land on 'AP Calculus AB',
        and 'apes' on 'AP Environmental Science'.
        """
        if not spoken:
            return None
        needle = spoken.lower().strip()

        aliases = {
            "calc": "calculus", "apes": "environmental", "env": "environmental",
            "econ": "economics", "psych": "psychology", "spanish": "spanish",
            "english": "english", "gsa": "garden state", "history": "history",
            "us history": "history", "apush": "history", "pe": "physical",
            "gym": "physical",
        }
        needle = aliases.get(needle, needle)

        candidates = list(self._courses.values())
        for course in candidates:                       # exact
            if course.name.lower() == needle:
                return course
        for course in candidates:                       # substring either way
            lowered = course.name.lower()
            if needle in lowered or lowered in needle:
                return course

        # Fall back to best word overlap.
        words = set(needle.split())
        best, best_score = None, 0
        for course in candidates:
            score = len(words & set(course.name.lower().split()))
            if score > best_score:
                best, best_score = course, score
        return best

    @property
    def courses(self) -> list[Course]:
        return list(self._courses.values())

    # ---- assignments -----------------------------------------------------

    def _to_assignment(self, row: dict[str, Any]) -> Assignment:
        course_ids = plain_property(row, "Course") or []
        course_id = course_ids[0] if course_ids else ""
        course = self._courses.get(course_id)
        return Assignment(
            id=row["id"],
            name=title_of(row) or "Untitled",
            course=course.name if course else "",
            course_id=course_id,
            type=plain_property(row, "Type"),
            status=plain_property(row, "Status"),
            importance=plain_property(row, "Importance"),
            due=_parse_date(plain_property(row, "AssignmentDate")),
            effort_hours=plain_property(row, "Effort (hrs)"),
            progress=plain_property(row, "Progress"),
            description=plain_property(row, "Description") or "",
            url=page_url(row["id"]),
        )

    async def open_assignments(self) -> list[Assignment]:
        """Everything not yet submitted or written off."""
        filter_ = {
            "and": [
                {"property": "Status", "status": {"does_not_equal": "Submitted"}},
                {"property": "Status", "status": {"does_not_equal": "Incomplete"}},
            ]
        }
        return [self._to_assignment(row)
                async for row in self._client.query(DB.ASSIGNMENTS, filter=filter_)]

    async def upcoming(self, *, today: date, within_days: int = 14,
                       limit: int = 12) -> list[Assignment]:
        horizon = today + timedelta(days=within_days)
        items = [
            a for a in await self.open_assignments()
            if a.due is None or a.due <= horizon
        ]
        items.sort(key=lambda a: -a.priority(today))
        return items[:limit]

    async def due_on(self, day: date) -> list[Assignment]:
        return [a for a in await self.open_assignments() if a.due == day]

    async def overdue(self, today: date) -> list[Assignment]:
        items = [a for a in await self.open_assignments()
                 if a.due and a.due < today]
        items.sort(key=lambda a: a.due or today)
        return items

    # ---- writes ----------------------------------------------------------

    async def create(self, *, name: str, course: str | None = None,
                     due: date | None = None, type: str | None = None,
                     importance: str | None = None,
                     effort_hours: float | None = None,
                     description: str | None = None) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "Name": {"title": [{"text": {"content": name}}]}
        }

        matched: Course | None = None
        if course:
            matched = self.resolve_course(course)
            if matched:
                properties["Course"] = {"relation": [{"id": matched.id}]}
            else:
                log.warning("no course matched %r — creating without one", course)

        if due:
            properties["AssignmentDate"] = {"date": {"start": due.isoformat()}}
        if type:
            match = next((t for t in TYPES if t.lower() == type.lower()), None)
            if match:
                properties["Type"] = {"select": {"name": match}}
        if importance and importance in IMPORTANCE_WEIGHT:
            properties["Importance"] = {"select": {"name": importance}}
        if effort_hours is not None:
            properties["Effort (hrs)"] = {"number": effort_hours}
        if description:
            properties["Description"] = {"rich_text": [{"text": {"content": description}}]}

        properties["Status"] = {"status": {"name": "Not started"}}

        page = await self._client.create_page(DB.ASSIGNMENTS, properties)
        return {
            "ok": True,
            "id": page["id"],
            "url": page_url(page["id"]),
            "name": name,
            "course": matched.name if matched else None,
            "course_matched": matched is not None,
            "due": due.isoformat() if due else None,
        }

    async def set_status(self, assignment_id: str, status: str) -> dict[str, Any]:
        await self._client.update_page(
            assignment_id, {"Status": {"status": {"name": status}}}
        )
        return {"ok": True, "status": status}

    async def set_progress(self, assignment_id: str, percent: float) -> dict[str, Any]:
        await self._client.update_page(
            assignment_id, {"Progress": {"number": max(0.0, min(100.0, percent))}}
        )
        return {"ok": True, "progress": percent}

    # ---- days off (feeds the A/B rotation) -------------------------------

    async def days_off(self) -> set[date]:
        days: set[date] = set()
        try:
            async for row in self._client.query(DB.DAYS_OFF):
                for prop in (row.get("properties") or {}).values():
                    if prop.get("type") != "date":
                        continue
                    payload = prop.get("date") or {}
                    start = _parse_date(payload)
                    if not start:
                        continue
                    end = _parse_date({"start": payload.get("end")}) if payload.get("end") else None
                    cursor = start
                    while cursor <= (end or start):
                        days.add(cursor)
                        cursor += timedelta(days=1)
        except Exception as exc:
            log.warning("could not read Days Off: %s", exc)
        log.info("loaded %d days off", len(days))
        return days
