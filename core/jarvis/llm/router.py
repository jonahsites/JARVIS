"""Local first, escalate on complexity — your choice, implemented in three tiers.

1. Cheap heuristics (length, keywords). Free, catches most cases.
2. If those are inconclusive, qwen3 classifies it in one word. ~150 ms.
3. Mid-conversation escalation: if the local model gets stuck looping on tools
   or produces nothing usable, the agent hands the whole thread to the cloud.

Tier 3 is the one that matters in practice — it's the safety net for when the
8B model confidently does the wrong thing.
"""

from __future__ import annotations

import logging

from ..config import LLMConfig
from .base import ChatResult, LLMClient
from .ollama_client import OllamaClient
from .openrouter_client import OpenRouterClient
from .prompts import ROUTER_PROMPT

log = logging.getLogger("jarvis.router")


class Router:
    def __init__(self, local: OllamaClient, cloud: OpenRouterClient, config: LLMConfig):
        self.local = local
        self.cloud = cloud
        self._config = config
        self._cloud_ok = False
        self._local_ok = False

    async def probe(self) -> tuple[bool, bool]:
        self._local_ok = await self.local.available()
        self._cloud_ok = await self.cloud.available()
        log.info("routes — local:%s cloud:%s", self._local_ok, self._cloud_ok)
        return self._local_ok, self._cloud_ok

    async def pick(self, request: str) -> LLMClient:
        if not self._local_ok:
            return self.cloud if self._cloud_ok else self.local
        if not self._cloud_ok:
            return self.local

        lowered = request.lower()

        if any(keyword in lowered for keyword in self._config.escalate_keywords):
            log.debug("escalating: keyword match")
            return self.cloud

        # Short imperatives are almost always one-step actions.
        if len(request.split()) <= 8:
            return self.local

        if len(request) > self._config.escalate_on_token_estimate:
            return self.cloud

        try:
            verdict = await self.local.chat(
                [{"role": "user", "content": ROUTER_PROMPT.format(request=request)}]
            )
            if "complex" in verdict.content.lower():
                log.debug("escalating: classifier")
                return self.cloud
        except Exception as exc:
            log.debug("classifier failed, staying local: %s", exc)

        return self.local

    def fallback_for(self, client: LLMClient) -> LLMClient | None:
        """Tier 3 — where to go when `client` isn't getting there."""
        if client is self.local and self._cloud_ok:
            return self.cloud
        if client is self.cloud and self._local_ok:
            return self.local
        return None

    @staticmethod
    def is_usable(result: ChatResult) -> bool:
        return bool(result.content.strip() or result.tool_calls)
