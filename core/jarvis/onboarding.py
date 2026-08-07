"""First run — the conversation where JARVIS learns who you are.

You asked for something that goes past name-and-birthday: it should tell you to
do things, watch you do them, and learn how. That's the DEMONSTRATE steps below.
When it says "open whatever you write essays in", it watches the frontmost app
and the tab list, sees you open Google Docs, and stores `essay_app = Google
Docs`. From then on "open my essay doc" resolves without a guess.

Three kinds of step:

  ASK         a spoken question; the answer is stored as a fact
  DEMONSTRATE a spoken instruction; JARVIS watches and stores what you did
  GRANT       offers a capability up front so it doesn't interrupt later

Everything is skippable — saying "skip" or "I don't know" moves on. It resumes
where you left off if you quit halfway, so it never restarts from the top.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

from .agent.capabilities import CapabilityLedger
from .memory.db import Memory
from .skills import mac

log = logging.getLogger("jarvis.onboarding")

StepKind = Literal["ask", "demonstrate", "grant", "say"]

SKIP_WORDS = {"skip", "pass", "next", "dunno", "unsure", "later", "nothing"}


@dataclass
class Step:
    kind: StepKind
    prompt: str
    key: str = ""
    capability: str = ""
    # DEMONSTRATE only: what to watch. "app" reads the frontmost application,
    # "tab" reads whichever Chrome tab is active.
    watch: str = "app"


STEPS: list[Step] = [
    Step("say", "Alright — this'll take about three minutes, and it means I "
                "stop guessing at things. Say skip to any of these and I'll "
                "move on."),

    # --- who you are ---
    Step("ask", "What should I call you?", key="preferred_name"),
    Step("ask", "And what should I call myself? Jarvis is fine, or pick "
                "something else.", key="assistant_name"),
    Step("ask", "What time do you usually get up on a school day?",
         key="wake_time"),
    Step("ask", "What time are you usually done with homework?",
         key="homework_end_time"),
    Step("ask", "Anything I should know about how you like to work? Music on, "
                "one thing at a time, whatever it is.", key="work_style"),

    # --- learning by watching ---
    Step("demonstrate",
         "Now open whatever you use to write essays. I'm watching.",
         key="essay_app", watch="app"),
    Step("demonstrate",
         "Open the app you use for anything with numbers or problem sets.",
         key="math_app", watch="app"),
    Step("demonstrate",
         "Open the site you check for grades or assignments from school.",
         key="grades_site", watch="tab"),
    Step("demonstrate",
         "Open the first thing you open when you sit down to work.",
         key="first_thing", watch="app"),
    Step("demonstrate",
         "Open whatever you put on in the background while you're working.",
         key="background_app", watch="app"),
    Step("demonstrate",
         "Last one — open the site you go to when you're procrastinating. "
         "No judgement, I just want to know what to close.",
         key="distraction_site", watch="tab"),

    # --- permissions, offered rather than sprung on you ---
    Step("grant", "Okay if I open browser tabs for you?",
         capability="chrome.open_tab"),
    Step("grant", "Okay if I keep track of which tabs you have open?",
         capability="chrome.read_tabs"),
    Step("grant", "Okay if I open and close apps?", capability="apps.open"),
    Step("grant", "Okay if I read and add things in your Notion?",
         capability="notion.read"),
    Step("grant", "Okay if I add assignments for you?",
         capability="notion.create_assignment"),
    Step("grant", "And okay if I send texts? I'll always read one out before "
                  "it goes.", capability="messages.send"),

    Step("say", "That's it. I've got the shape of your day now — just say hey "
                "Jarvis whenever you need me."),
]


class Onboarding:
    def __init__(self, *, memory: Memory, ledger: CapabilityLedger,
                 speak: Callable[[str], Awaitable[None]],
                 ask: Callable[[str], Awaitable[str]],
                 ask_yes_no: Callable[[str], Awaitable[bool]]):
        self._memory = memory
        self._ledger = ledger
        self._speak = speak
        self._ask = ask
        self._ask_yes_no = ask_yes_no

    @property
    def complete(self) -> bool:
        return self._memory.recall("onboarding_complete") == "yes"

    async def run(self, *, restart: bool = False) -> None:
        if restart:
            self._memory.remember("onboarding_step", "0", source="onboarding")

        start = int(self._memory.recall("onboarding_step") or 0)
        if start:
            await self._speak("Picking up where we left off.")

        for index in range(start, len(STEPS)):
            step = STEPS[index]
            self._memory.remember("onboarding_step", str(index), source="onboarding")
            try:
                await self._run_step(step)
            except Exception:
                log.exception("onboarding step %d failed", index)

        self._memory.remember("onboarding_complete", "yes", source="onboarding")
        log.info("onboarding finished")

    async def _run_step(self, step: Step) -> None:
        if step.kind == "say":
            await self._speak(step.prompt)

        elif step.kind == "ask":
            answer = (await self._ask(step.prompt)).strip()
            if not answer or answer.lower().strip(".!?") in SKIP_WORDS:
                return
            self._memory.remember(step.key, answer, source="onboarding")
            log.info("learned %s = %r", step.key, answer)

        elif step.kind == "grant":
            if self._ledger.is_granted(step.capability):
                return
            if await self._ask_yes_no(step.prompt):
                self._ledger.grant(step.capability, step.prompt)

        elif step.kind == "demonstrate":
            await self._demonstrate(step)

    async def _demonstrate(self, step: Step) -> None:
        """Say the instruction, then watch for what changes."""
        before_app = await _safe(mac.frontmost_app, "")
        before_tab = await _current_tab()

        await self._speak(step.prompt)

        # Poll rather than sleep-then-look, so a fast answer isn't kept waiting
        # and a slow one still gets caught.
        observed: str | None = None
        for _ in range(20):  # up to ~20s
            await asyncio.sleep(1.0)

            if step.watch == "app":
                current = await _safe(mac.frontmost_app, "")
                if current and current != before_app and current != "Terminal":
                    observed = current
                    break
            else:
                current = await _current_tab()
                if current and current != before_tab:
                    observed = current
                    break

        if not observed:
            await self._speak("Didn't catch that one — moving on.")
            return

        self._memory.remember(step.key, observed, source="onboarding")
        log.info("learned %s = %r (by demonstration)", step.key, observed)
        await self._speak(f"Got it — {observed}.")


async def _safe(fn, default):
    try:
        return await fn()
    except Exception:
        return default


async def _current_tab() -> str:
    """Host of the active Chrome tab, which is the stable part of a URL."""
    try:
        tabs = await mac.list_tabs()
    except Exception:
        return ""
    for tab in tabs:
        if tab.get("active"):
            url = tab.get("url") or ""
            if "://" in url:
                return url.split("://", 1)[1].split("/", 1)[0]
            return url
    return ""
