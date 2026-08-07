"""Local model via Ollama (qwen3:8b by default).

Qwen3 is a hybrid reasoning model — left alone it emits a <think> block before
answering, which costs you a second or two of latency on requests that don't
need it. For the fast path we send `/no_think`; the escalation path to
OpenRouter exists precisely so the local model never has to do the hard
reasoning anyway.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from .base import ChatResult, ToolCall

log = logging.getLogger("jarvis.llm.ollama")


class OllamaClient:
    name = "local"

    def __init__(self, host: str, model: str, timeout_s: float = 20.0,
                 thinking: bool = False):
        self.host = host
        self.model = model
        self.timeout_s = timeout_s
        self.thinking = thinking
        self._client = None

    def _get(self):
        if self._client is None:
            from ollama import AsyncClient

            self._client = AsyncClient(host=self.host, timeout=self.timeout_s)
        return self._client

    async def available(self) -> bool:
        try:
            models = await self._get().list()
            names = {m.get("model", "") for m in models.get("models", [])}
            if self.model in names:
                return True
            # Ollama reports "qwen3:8b"; tolerate a bare "qwen3" in config.
            return any(n.split(":")[0] == self.model.split(":")[0] for n in names)
        except Exception as exc:
            log.warning("ollama unreachable at %s: %s", self.host, exc)
            return False

    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None) -> ChatResult:
        payload = list(messages)
        if not self.thinking and payload:
            # Qwen3's soft switch. Harmless on models that don't recognise it.
            last = dict(payload[-1])
            last["content"] = f"{last.get('content', '')}\n/no_think"
            payload[-1] = last

        response = await self._get().chat(
            model=self.model, messages=payload, tools=tools or None,
            options={"temperature": 0.4, "num_predict": 512},
        )

        message = response.get("message", {}) or {}
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            calls.append(
                ToolCall(id=raw.get("id") or uuid.uuid4().hex,
                         name=function.get("name", ""), arguments=arguments)
            )

        return ChatResult(content=message.get("content", "") or "",
                          tool_calls=calls, route="local")
