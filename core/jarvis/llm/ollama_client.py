"""Local model via Ollama (qwen3:8b by default).

Three things here exist purely for latency, because a secretary that takes
fifteen seconds to tell you the time is not a secretary:

`think=False` turns off Qwen3's reasoning block at the template level. That's
the real switch — the `/no_think` string is only a fallback for builds that
don't support the parameter. A thinking pass can easily double the time to
first word, and the escalation path to OpenRouter exists precisely so the local
model never needs to do hard reasoning anyway.

`keep_alive` stops Ollama evicting the model between turns. Reloading 5 GB
costs several seconds and it happens silently.

`num_ctx` is raised because the default is small enough that thirteen tool
schemas plus the system prompt can overflow it, which makes Ollama silently
re-process context on every call.
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
                 thinking: bool = False, keep_alive: str = "30m",
                 num_ctx: int = 8192):
        self.host = host
        self.model = model
        self.timeout_s = timeout_s
        self.thinking = thinking
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self._client = None
        # Older ollama-python has no `think` parameter; detected once on the
        # first call rather than guessed from a version string.
        self._supports_think = True

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
            # Fallback switch, on the system message so it can never land on a
            # tool result. Harmless on models that don't recognise it.
            payload[0] = dict(payload[0])
            payload[0]["content"] = f"{payload[0].get('content', '')}\n\n/no_think"

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "tools": tools or None,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.4,
                "num_predict": 512,
                "num_ctx": self.num_ctx,
            },
        }
        if self._supports_think and not self.thinking:
            kwargs["think"] = False

        try:
            response = await self._get().chat(**kwargs)
        except TypeError as exc:
            if "think" not in str(exc) or not self._supports_think:
                raise
            log.info("ollama build has no `think` parameter — using /no_think only")
            self._supports_think = False
            kwargs.pop("think", None)
            response = await self._get().chat(**kwargs)

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
