"""First run — the conversation where JARVIS learns who you are.

Two kinds of step, because two kinds of question deserve different interfaces.

Things you can just say — your name, when you get up, how you like to work —
are asked out loud. You answer whenever it finishes speaking; no wake word,
and you can talk over it.

Things that are fiddly to say — exact app names, URLs — are collected in a form
instead. JARVIS opens a tab and you type or paste. The earlier version told you
to go open an app while it watched the frontmost window, which was both slow
and unreliable: it gave you twenty seconds, guessed from whatever happened to
come to the front, and couldn't capture a specific link at all.

Everything is skippable, and progress is saved after each step so quitting
halfway doesn't start you over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal

from .agent.capabilities import CapabilityLedger
from .memory.db import Memory

log = logging.getLogger("jarvis.onboarding")

StepKind = Literal["ask", "form", "grant", "say"]

SKIP_WORDS = {"skip", "pass", "next", "dunno", "unsure", "later", "nothing",
              "no idea", "none"}


@dataclass
class Field:
    key: str
    label: str
    hint: str = ""
    placeholder: str = ""


@dataclass
class Step:
    kind: StepKind
    prompt: str
    key: str = ""
    capability: str = ""
    title: str = ""
    fields: list[Field] = field(default_factory=list)


STEPS: list[Step] = [
    Step("say", "Alright — this'll take a couple of minutes, and it means I "
                "stop guessing at things. Say skip to anything you'd rather "
                "not answer, and feel free to talk over me."),

    # --- spoken: things that are natural to say ---
    Step("ask", "What should I call you?", key="preferred_name"),
    Step("ask", "What time do you usually get up on a school day?",
         key="wake_time"),
    Step("ask", "What time are you usually done with homework?",
         key="homework_end_time"),
    Step("ask", "Anything I should know about how you like to work? Music on, "
                "one thing at a time, whatever it is.", key="work_style"),
    Step("ask", "Is there anything you want me to nag you about?",
         key="nag_about"),

    # --- form: things that are fiddly to say out loud ---
    Step(
        "form",
        "I've opened a tab — fill in whatever you know and skip the rest. "
        "App names or links, either works.",
        title="What do you use for what?",
        fields=[
            Field("essay_app", "Writing essays",
                  "App name or a link", "Google Docs, or a link to your folder"),
            Field("math_app", "Math and problem sets",
                  "App name or a link", "Notion, Desmos, a textbook link"),
            Field("grades_site", "Checking grades",
                  "Usually a link", "https://..."),
            Field("first_thing", "First thing you open to start working",
                  "App name or a link", "Notion"),
            Field("background_app", "On in the background while you work",
                  "App name or a link", "Spotify"),
            Field("distraction_site", "Where you go when you're procrastinating",
                  "So I know what to close", "youtube.com"),
            Field("school_portal", "Your school's main site",
                  "Optional", "https://..."),
            Field("project_folder", "A project or folder you open a lot",
                  "Optional", "~/Projects/something"),
        ],
    ),

    # --- permissions, offered rather than sprung on you mid-task ---
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
                  "it goes, so you can stop it.", capability="messages.send"),

    Step("say", "That's it. I've got the shape of your day now — just say hey "
                "Jarvis whenever you need me."),
]

# Facts that came from onboarding, cleared by --restart so a re-run is a genuine
# fresh start rather than a re-confirmation of stale answers.
def _onboarding_keys() -> list[str]:
    keys = [s.key for s in STEPS if s.kind == "ask" and s.key]
    for step in STEPS:
        if step.kind == "form":
            keys.extend(f.key for f in step.fields)
    return keys


FormPresenter = Callable[[str, str, list[Field]], Awaitable[dict[str, str]]]


class Onboarding:
    def __init__(self, *, memory: Memory, ledger: CapabilityLedger,
                 speak: Callable[[str], Awaitable[None]],
                 ask: Callable[[str], Awaitable[str]],
                 ask_yes_no: Callable[[str], Awaitable[bool]],
                 show_form: FormPresenter | None = None):
        self._memory = memory
        self._ledger = ledger
        self._speak = speak
        self._ask = ask
        self._ask_yes_no = ask_yes_no
        self._show_form = show_form

    @property
    def complete(self) -> bool:
        return self._memory.recall("onboarding_complete") == "yes"

    def reset(self) -> None:
        """Forget everything onboarding taught it, and start from step zero."""
        for key in [*_onboarding_keys(), "onboarding_complete", "onboarding_step"]:
            self._memory.execute("DELETE FROM facts WHERE key=?", (key,))
        log.info("onboarding reset")

    async def run(self, *, restart: bool = False) -> None:
        if restart:
            self.reset()

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

        elif step.kind == "form":
            await self._collect(step)

    async def _collect(self, step: Step) -> None:
        if self._show_form is None:
            log.warning("no form presenter — skipping %s", step.title)
            return

        await self._speak(step.prompt)
        values = await self._show_form(step.title, step.prompt, step.fields)

        saved = 0
        for key, value in values.items():
            value = (value or "").strip()
            if not value:
                continue
            self._memory.remember(key, value, source="onboarding")
            saved += 1

        log.info("form saved %d of %d fields", saved, len(step.fields))
        if saved:
            await self._speak(f"Got it, {saved} thing{'s' if saved != 1 else ''} saved.")
        else:
            await self._speak("No problem, we can fill that in later.")
