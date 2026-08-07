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

        self._loop_task: asyncio.Task | None = None

    # ---- glue ------------------------------------------------------------

    def _on_state_change(self, previous: AgentState, current: AgentState) -> None:
        log.info("%s -> %s", previous.value, current.value)
        asyncio.create_task(self.bus.emit(EV_STATE, state=current.value))

    def _build_context(self) -> str:
        """Rebuilt every turn so the model never reasons from stale state."""
        now = self.schedule.now()
        return context_block(
            now=now.strftime("%A, %B %-d %Y, %-I:%M %p"),
            day_type=self.schedule.day_type(),
            classes=[],      # filled in once NOTION_TOKEN is set
            due_soon=[],     # ditto
            facts=self.memory.all_facts(),
            recent=[(r["transcript"], r["response"])
                    for r in self.memory.recent_interactions(4)],
        )

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

        # First-use permission prompts are spoken and answerable out loud.
        self.ledger.ask_aloud = self.pipeline.ask_yes_no

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

        log.info(
            "JARVIS is up. Say \"hey Jarvis\" or press %s. UI at http://localhost:5173",
            self.config.voice.hotkey,
        )

    async def stop(self) -> None:
        self.hotkey.stop()
        await self.pipeline.stop()
        await self.bus.stop()
        self.memory.close()

    async def run(self) -> None:
        await self.start()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()
