"""Wires everything together and runs it."""

from __future__ import annotations

import asyncio
import logging

from .agent.capabilities import CapabilityLedger
from .agent.loop import Agent
from .agent.registry import Registry
from .audio.hotkey import GlobalHotkey
from .audio.pipeline import VoicePipeline
from .bus import EV_CANCEL, EV_HOTKEY, EV_STATE, EV_TEXT_INPUT, Bus, Event
from .config import JarvisConfig, Secrets
from .llm.ollama_client import OllamaClient
from .llm.openrouter_client import OpenRouterClient
from .llm.prompts import context_block
from .llm.router import Router
from .memory.db import Memory
from .skills.notion.assignments import Assignments
from .skills.notion.client import NotionClient
from .skills.notion.notes import NoteCrawler, NoteIndex
from .skills.notion.tools import register_notion
from .skills.schedule import ScheduleResolver
from .skills.tools import register_all
from .state import AgentState, StateMachine

log = logging.getLogger("jarvis.daemon")


class Daemon:
    def __init__(self, config: JarvisConfig, secrets: Secrets):
        self.config = config
        self.secrets = secrets

        self.bus = Bus(secrets.jarvis_bus_port)
        self.memory = Memory()
        self.state = StateMachine(on_change=self._on_state_change)
        self.ledger = CapabilityLedger(self.memory, self.bus)
        self.registry = Registry(self.ledger)
        self.schedule = ScheduleResolver(config.schedule)

        self.notion = NotionClient(secrets.notion_token)
        self.assignments = Assignments(self.notion)
        self.notes = NoteIndex()
        self.crawler = NoteCrawler(self.notion, self.notes)

        self.router = Router(
            OllamaClient(secrets.ollama_host, secrets.ollama_model,
                         config.llm.local_timeout_s),
            OpenRouterClient(secrets.openrouter_api_key, secrets.openrouter_model,
                             config.llm.cloud_timeout_s),
            config.llm,
        )

        self.pipeline = VoicePipeline(config, self.bus, self.state)
        self.agent = Agent(config, self.router, self.registry, self.memory,
                           self._build_context)
        self.hotkey = GlobalHotkey(config.voice.hotkey, self._on_hotkey)

        self._due_soon: list[str] = []
        self._tasks: list[asyncio.Task] = []

    # ---- glue ------------------------------------------------------------

    def _on_state_change(self, previous: AgentState, current: AgentState) -> None:
        log.info("%s -> %s", previous.value, current.value)
        asyncio.create_task(self.bus.emit(EV_STATE, state=current.value))

    def _build_context(self) -> str:
        """Rebuilt every turn so the model never reasons from stale state.

        Reads only from caches refreshed in the background — a context build
        must never make a network call, or every reply would pay for it.
        """
        now = self.schedule.now()
        return context_block(
            now=now.strftime("%A, %B %-d %Y, %-I:%M %p"),
            day_type=self.schedule.day_type(),
            classes=self._todays_classes(),
            due_soon=self._due_soon,
            facts=self.memory.all_facts(),
            recent=[(r["transcript"], r["response"])
                    for r in self.memory.recent_interactions(4)],
        )

    def _todays_classes(self) -> list[str]:
        rows = [c.as_dict() for c in self.assignments.courses]
        if not rows:
            return []
        return [
            f"{c['name']} at {c['class_time'].split(' - ')[0]}"
            + (f" in {c['location']}" if c.get("location") else "")
            for c in self.schedule.classes_today(rows)
        ]

    async def _refresh_notion(self) -> None:
        """Courses, days off and what's due. Cheap enough to run on a timer."""
        if not self.notion.enabled:
            return
        try:
            await self.assignments.load_courses()
            self.schedule.set_days_off(await self.assignments.days_off())

            today = self.schedule.now().date()
            upcoming = await self.assignments.upcoming(today=today, limit=8)
            self._due_soon = [a.spoken(today) for a in upcoming]
            log.info("notion refreshed — %d courses, %d due soon",
                     len(self.assignments.courses), len(self._due_soon))
        except Exception:
            log.exception("notion refresh failed")

    async def _notion_loop(self) -> None:
        while True:
            await asyncio.sleep(900)  # 15 min
            await self._refresh_notion()

    async def _crawl_loop(self) -> None:
        """Incremental — unchanged pages are skipped, so this is cheap."""
        while True:
            try:
                await self.crawler.crawl()
            except Exception:
                log.exception("note crawl failed")
            await asyncio.sleep(3600)

    async def _on_hotkey(self) -> None:
        log.info("hotkey pressed")
        await self.pipeline.begin_listening()

    async def _on_bus_hotkey(self, _: Event) -> None:
        await self.pipeline.begin_listening()

    async def _on_cancel(self, _: Event) -> None:
        """Cuts speech immediately — backs the 3-second message cancel window."""
        self.pipeline.tts.stop()

    async def _on_text_input(self, event: Event) -> None:
        """Typed input from the UI. Same path as speech, minus STT."""
        text = (event.payload.get("text") or "").strip()
        if not text:
            return
        await self.state.transition(AgentState.THINKING)
        reply = await self.agent.respond(text)
        await self.pipeline.say(reply)

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        await self.bus.start()
        self.bus.on(EV_HOTKEY, self._on_bus_hotkey)
        self.bus.on(EV_CANCEL, self._on_cancel)
        self.bus.on(EV_TEXT_INPUT, self._on_text_input)

        register_all(self.registry, self.memory, self.schedule, self.ledger)
        register_notion(self.registry, self.assignments, self.notes,
                        self.crawler, self.schedule)
        log.info("%d tools registered", len(self.registry.names()))

        # First-use permission prompts are spoken and answerable out loud.
        self.ledger.ask_aloud = self.pipeline.ask_yes_no

        if self.notion.enabled:
            await self._refresh_notion()
        else:
            log.warning("NOTION_TOKEN not set — assignments and notes are off")

        local_ok, cloud_ok = await self.router.probe()
        if not local_ok and not cloud_ok:
            log.error(
                "no model available — start Ollama (`ollama serve`) or set "
                "OPENROUTER_API_KEY in .env"
            )

        await self.pipeline.load()
        self.pipeline.set_responder(self.agent.respond)
        await self.pipeline.start()

        self.hotkey.start()

        pruned = self.memory.prune(self.config.memory.retention_days)
        if pruned:
            log.info("pruned %d old observations", pruned)

        if self.notion.enabled:
            # Background so a first crawl of a big workspace doesn't delay the
            # microphone coming up.
            self._tasks.append(asyncio.create_task(self._crawl_loop()))
            self._tasks.append(asyncio.create_task(self._notion_loop()))

        log.info(
            "JARVIS is up. Say \"hey Jarvis\" or press %s. UI at http://localhost:5173",
            self.config.voice.hotkey,
        )

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        self.hotkey.stop()
        await self.pipeline.stop()
        await self.bus.stop()
        await self.notion.close()
        self.notes.close()
        self.memory.close()

    async def run(self) -> None:
        await self.start()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()
