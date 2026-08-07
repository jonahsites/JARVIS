"""Speaking up unprompted.

You chose fully autonomous — it says something when it judges something is worth
saying, rather than only at fixed times. The risk with that is obvious: an
assistant that interrupts constantly gets muted within a day. So the design is
"generous triggers, strict gates".

Triggers are cheap and specific (below). Gates are what keep it liveable:

  - never while it's already listening, thinking or speaking
  - never during quiet hours
  - never while you're in a call
  - never more than max_per_hour
  - never the same nudge twice within its own cooldown

Anything that clears all five gets spoken. Anything that doesn't gets dropped
silently rather than queued, because a nudge that arrives an hour late is worse
than no nudge.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time, timedelta
from typing import Awaitable, Callable

from ..config import ProactiveConfig
from ..memory.db import Memory
from ..skills.observer import Observer
from ..skills.schedule import PERIOD_STARTS, ScheduleResolver
from ..state import StateMachine

log = logging.getLogger("jarvis.proactive")


@dataclass
class Nudge:
    kind: str          # dedupe key
    text: str          # spoken verbatim
    cooldown_s: float  # how long before this kind may fire again


class ProactiveEngine:
    def __init__(self, *, config: ProactiveConfig, state: StateMachine,
                 memory: Memory, schedule: ScheduleResolver,
                 observer: Observer, speak: Callable[[str], Awaitable[None]],
                 assignments_provider: Callable[[], list] | None = None):
        self._config = config
        self._state = state
        self._memory = memory
        self._schedule = schedule
        self._observer = observer
        self._speak = speak
        self._assignments = assignments_provider or (lambda: [])
        self._greeted_on: date | None = None
        self._courses: list[dict] = []

    # ---- gates -----------------------------------------------------------

    def _in_quiet_hours(self, now: datetime) -> bool:
        try:
            start = clock_time.fromisoformat(self._config.quiet_hours[0])
            end = clock_time.fromisoformat(self._config.quiet_hours[1])
        except (ValueError, IndexError):
            return False
        current = now.time()
        # Quiet hours normally wrap midnight (22:30 -> 07:00).
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    def _may_speak(self, now: datetime) -> tuple[bool, str]:
        if not self._config.enabled:
            return False, "disabled"
        if not self._state.can_interrupt():
            return False, f"busy ({self._state.state.value})"
        if self._in_quiet_hours(now):
            return False, "quiet hours"
        if self._observer.current_app in self._config.focus_apps:
            return False, f"in {self._observer.current_app}"
        if self._memory.proactive_count_since(3600) >= self._config.max_per_hour:
            return False, "hourly limit"
        return True, ""

    # ---- triggers --------------------------------------------------------

    def _consider(self, now: datetime) -> Nudge | None:
        """First matching trigger wins — they're ordered by how much you'd
        want to be interrupted for them."""
        today = now.date()
        items = [a for a in self._assignments() if not a.done]

        # 1. Something is overdue and untouched.
        overdue = [a for a in items if a.due and a.due < today]
        if overdue:
            worst = min(overdue, key=lambda a: a.due)
            days = (today - worst.due).days
            return Nudge(
                "overdue",
                f"Heads up — {worst.name}"
                + (f" for {worst.course}" if worst.course else "")
                + f" was due {'yesterday' if days == 1 else f'{days} days ago'}.",
                cooldown_s=6 * 3600,
            )

        # 2. Due today, and the day is getting on.
        due_today = [a for a in items if a.due == today]
        if due_today and now.hour >= 15:
            names = ", ".join(a.name for a in due_today[:2])
            return Nudge(
                "due_today",
                f"You've still got {names} due today.",
                cooldown_s=3 * 3600,
            )

        # 3. Next class starts soon and something is due for it.
        upcoming = self._next_class_within(now, minutes=20)
        if upcoming:
            course_name = upcoming.get("name", "")
            for assignment in items:
                if assignment.course == course_name and assignment.due == today:
                    return Nudge(
                        f"before_class:{course_name}",
                        f"{course_name} is in a few minutes, and {assignment.name} "
                        f"is due for it.",
                        cooldown_s=2 * 3600,
                    )
            return Nudge(
                f"before_class:{course_name}",
                f"{course_name} starts in a few minutes"
                + (f", room {upcoming['location']}" if upcoming.get("location") else "")
                + ".",
                cooldown_s=2 * 3600,
            )

        # 4. Morning greeting, once, on a school day.
        if (self._greeted_on != today and 6 <= now.hour < 11
                and self._schedule.day_type(today)):
            self._greeted_on = today
            day_type = self._schedule.day_type(today)
            due_soon = [a for a in items
                        if a.due and 0 <= (a.due - today).days <= 2]
            tail = (f" You've got {len(due_soon)} thing"
                    f"{'s' if len(due_soon) != 1 else ''} due in the next couple of days."
                    if due_soon else " Nothing urgent on your plate.")
            return Nudge("morning", f"Morning. It's a {day_type} day.{tail}",
                         cooldown_s=20 * 3600)

        # 5. A big assignment has hit the day it should be started.
        for assignment in items:
            start_by = assignment.start_by(today)
            if start_by and start_by == today and assignment.effort_hours:
                return Nudge(
                    f"start_by:{assignment.id}",
                    f"If you want {assignment.name} done comfortably, today's "
                    f"the day to start it.",
                    cooldown_s=20 * 3600,
                )

        # 6. Long stretch in one app with something due today.
        if self._observer.app_minutes > 45 and due_today:
            return Nudge(
                "long_focus",
                f"You've been in {self._observer.current_app} a while — "
                f"{due_today[0].name} is still due today.",
                cooldown_s=4 * 3600,
            )

        return None

    def _next_class_within(self, now: datetime, minutes: int) -> dict | None:
        rows = self._courses
        if not rows:
            return None
        day_type = self._schedule.day_type(now.date())
        if day_type is None:
            return None

        for course in self._schedule.classes_today(rows):
            start = PERIOD_STARTS.get(course.get("class_time", ""))
            if not start:
                continue
            starts_at = datetime.combine(now.date(), start, tzinfo=now.tzinfo)
            gap = (starts_at - now).total_seconds() / 60
            if 0 < gap <= minutes:
                return course
        return None

    def set_courses(self, rows: list[dict]) -> None:
        self._courses = rows

    # ---- loop ------------------------------------------------------------

    async def tick(self) -> None:
        now = self._schedule.now()

        allowed, reason = self._may_speak(now)
        if not allowed:
            log.debug("staying quiet: %s", reason)
            return

        nudge = self._consider(now)
        if nudge is None:
            return

        if self._memory.said_recently(nudge.kind, nudge.cooldown_s):
            log.debug("already said %s recently", nudge.kind)
            return

        # Re-check right before speaking — a tick can take long enough for you
        # to have started talking to it in the meantime.
        allowed, reason = self._may_speak(self._schedule.now())
        if not allowed:
            log.debug("dropped %s: %s", nudge.kind, reason)
            return

        log.info("proactive: %s", nudge.kind)
        self._memory.log_proactive(nudge.kind, nudge.text)
        await self._speak(nudge.text)

    async def run(self) -> None:
        # Don't greet the moment the daemon starts — it's jarring, and startup
        # is exactly when you're least ready to be talked at.
        await asyncio.sleep(60)
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("proactive tick failed")
            await asyncio.sleep(self._config.tick_seconds)
