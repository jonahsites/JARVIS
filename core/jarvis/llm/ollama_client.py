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

    @staticmethod
    def _render(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Canonical -> Ollama.

        Ollama validates with pydantic: `arguments` must be a dict (not the
        JSON string OpenAI uses), there is no `tool_call_id` field, and tool
        results are matched back by `tool_name`.
        """
        out: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")

            if role == "assistant" and message.get("tool_calls"):
                out.append({
                    "role": "assistant",
                    "content": message.get("content", "") or "",
                    "tool_calls": [
                        {"function": {"name": call["name"],
                                      "arguments": call["arguments"]}}
                        for call in message["tool_calls"]
                    ],
                })
            elif role == "tool":
                out.append({
                    "role": "tool",
                    "content": message.get("content", ""),
                    "tool_name": message.get("name", ""),
                })
            else:
                out.append({k: v for k, v in message.items()
                            if k in ("role", "content")})
        return out

    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None) -> ChatResult:
        payload = self._render(messages)

        if not self.thinking and payload and payload[0].get("role") == "system":
            # Qwen3's soft switch, on the system message so it can't land on a
            # tool result. Harmless on models that don't recognise it.
            payload[0] = dict(payload[0])
            payload[0]["content"] = f"{payload[0].get('content', '')}\n\n/no_think"

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
