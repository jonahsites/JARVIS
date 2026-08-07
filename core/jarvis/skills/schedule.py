"""A/B day rotation.

BLOCKER 3, encoded. Your Courses database has Day Type (A/B), Class Times and a
Days Off relation, but Notion's API will not return the source of your
`Class today?` formula, so I could not read your actual rule. This implements
the standard one: A and B alternate across school days, and days in Days Off
don't advance the rotation.

If it ever says the wrong letter, fix `anchor_date` / `anchor_day_type` in
config.toml — you don't need to touch this file. Set the anchor to any day you
know for certain and everything before and after it re-derives.

Your schedule as it stands in Notion:
    A: Garden State ACC 8:00 (272) · PE 9:30 (Gym) · AP Psych 11:46 (Cafeteria)
       · AP Calc AB 1:12 (133)
    B: Spanish Cinema 8:00 (138) · APES 9:30 (124) · AP Econ 11:46 (Mill-6)
       · US History 1:12 (239)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from ..config import ScheduleConfig

log = logging.getLogger("jarvis.schedule")

# Parsed from the Class Times select options in your Courses database.
PERIOD_STARTS: dict[str, time] = {
    "8:00 am - 9:24 am": time(8, 0),
    "9:30 am - 10:50 am": time(9, 30),
    "11:46 am - 1:08 pm": time(11, 46),
    "1:12 pm - 2:36 pm": time(13, 12),
}
PERIOD_ENDS: dict[str, time] = {
    "8:00 am - 9:24 am": time(9, 24),
    "9:30 am - 10:50 am": time(10, 50),
    "11:46 am - 1:08 pm": time(13, 8),
    "1:12 pm - 2:36 pm": time(14, 36),
}


class ScheduleResolver:
    def __init__(self, config: ScheduleConfig, days_off: set[date] | None = None):
        self._config = config
        self._days_off = days_off or set()
        self._tz = ZoneInfo(config.timezone)

    def set_days_off(self, days: set[date]) -> None:
        """Populated from your Notion Days Off database at startup."""
        self._days_off = days

    def now(self) -> datetime:
        return datetime.now(self._tz)

    def is_school_day(self, day: date) -> bool:
        return day.weekday() in self._config.school_weekdays and day not in self._days_off

    def day_type(self, day: date | None = None) -> str | None:
        """'A', 'B', or None if it isn't a school day."""
        day = day or self.now().date()
        if not self.is_school_day(day):
            return None

        anchor = self._config.anchor_date
        if day == anchor:
            return self._config.anchor_day_type

        # Count school days between anchor and target; parity gives the letter.
        step = 1 if day > anchor else -1
        cursor, school_days = anchor, 0
        while cursor != day:
            cursor += timedelta(days=step)
            if self.is_school_day(cursor):
                school_days += 1

        flipped = school_days % 2 == 1
        if not flipped:
            return self._config.anchor_day_type
        return "B" if self._config.anchor_day_type == "A" else "A"

    def next_school_day(self, after: date | None = None) -> date:
        cursor = (after or self.now().date()) + timedelta(days=1)
        for _ in range(30):
            if self.is_school_day(cursor):
                return cursor
            cursor += timedelta(days=1)
        return cursor

    def classes_today(self, courses: list[dict]) -> list[dict]:
        """Filter a list of Notion Course rows down to today's, in period order."""
        today_type = self.day_type()
        if today_type is None:
            return []
        todays = [c for c in courses if c.get("day_type") == today_type and c.get("class_time")]
        return sorted(todays, key=lambda c: PERIOD_STARTS.get(c["class_time"], time(23, 59)))

    def current_or_next_class(self, courses: list[dict]) -> tuple[dict | None, str]:
        """Returns (course, 'now' | 'next' | 'done' | 'none')."""
        todays = self.classes_today(courses)
        if not todays:
            return None, "none"

        clock = self.now().time()
        for course in todays:
            slot = course["class_time"]
            if PERIOD_STARTS[slot] <= clock <= PERIOD_ENDS[slot]:
                return course, "now"
        for course in todays:
            if clock < PERIOD_STARTS[course["class_time"]]:
                return course, "next"
        return None, "done"

    def describe_today(self, courses: list[dict]) -> str:
        """A spoken-shaped summary — no lists, no times in 24-hour form."""
        day_type = self.day_type()
        if day_type is None:
            return "No school today."
        todays = self.classes_today(courses)
        if not todays:
            return f"It's an {day_type} day, but I don't have any classes on file for it."
        names = [c.get("name", "something") for c in todays]
        joined = ", ".join(names[:-1]) + f", and {names[-1]}" if len(names) > 1 else names[0]
        return f"It's an {day_type} day — {joined}."
