"""The agent loop: transcript in, one spoken sentence out.

Where the thinking/speaking split is actually enforced:
- tool calls and their results never leave this function
- `strip_thinking` runs on the final content before it is returned
- the caller (VoicePipeline.say) is the only path to Kokoro

So even if qwen3 ignores /no_think and reasons out loud, you never hear it.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from ..config import JarvisConfig
from ..llm.base import ChatResult, LLMClient
from ..llm.prompts import strip_thinking, system_prompt
from ..llm.router import Router
from ..memory.db import Memory
from .registry import Registry

log = logging.getLogger("jarvis.agent")

MAX_TOOL_ROUNDS = 5


class Agent:
    def __init__(self, config: JarvisConfig, router: Router, registry: Registry,
                 memory: Memory, context_builder: Callable[[], str]):
        self._config = config
        self._router = router
        self._registry = registry
        self._memory = memory
        self._build_context = context_builder

    async def respond(self, transcript: str) -> str:
        started = time.monotonic()
        used_tools: list[str] = []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(self._config, self._build_context())},
            {"role": "user", "content": transcript},
        ]

        client: LLMClient = await self._router.pick(transcript)
        log.info("routing to %s", client.name)

        escalated = False
        result: ChatResult | None = None

        for round_index in range(MAX_TOOL_ROUNDS):
            try:
                result = await client.chat(messages, tools=self._registry.schemas())
            except Exception as exc:
                log.warning("%s failed: %s", client.name, exc)
                fallback = self._router.fallback_for(client)
                if fallback is None or escalated:
                    return "I couldn't reach a model just now."
                client, escalated = fallback, True
                log.info("escalating to %s", client.name)
                continue

            # Local model produced nothing usable — hand the whole thread over
            # rather than making you repeat yourself.
            if not Router.is_usable(result) and not escalated:
                fallback = self._router.fallback_for(client)
                if fallback is not None:
                    client, escalated = fallback, True
                    log.info("escalating to %s (empty local response)", client.name)
                    continue

            if not result.wants_tools:
                break

            messages.append({
                "role": "assistant",
                "content": result.content,
                "tool_calls": [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.name,
                                  "arguments": json.dumps(call.arguments)}}
                    for call in result.tool_calls
                ],
            })

            for call in result.tool_calls:
                log.info("tool: %s(%s)", call.name, _brief(call.arguments))
                output = await self._registry.call(call.name, call.arguments)
                used_tools.append(call.name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(output, default=str)[:4000],
                })

            if round_index == MAX_TOOL_ROUNDS - 1:
                log.warning("hit tool round limit")

        if result is None:
            return "I couldn't reach a model just now."

        spoken = strip_thinking(result.content)
        if not spoken:
            spoken = "Done." if used_tools else "I'm not sure how to answer that."

        self._memory.log_interaction(
            transcript, spoken, result.route, used_tools,
            int((time.monotonic() - started) * 1000),
        )
        return spoken


def _brief(arguments: dict[str, Any], limit: int = 120) -> str:
    text = json.dumps(arguments, default=str)
    return text if len(text) <= limit else text[:limit] + "…"
