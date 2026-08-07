"""Escalation path. Used when the local model isn't the right tool for the job.

Worth knowing if you're on the free tier (`openrouter/free`): it's rate limited
to roughly 20 requests a minute and 200 a day, and not every free model handles
tool calling well. Both failure modes are treated the same way — log it and let
the agent fall back to the local model, which is always available. A slower
answer beats no answer.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from .base import ChatResult, ToolCall

log = logging.getLogger("jarvis.llm.openrouter")

BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    name = "cloud"

    def __init__(self, api_key: str, model: str, timeout_s: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self._client = None

    def _get(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=BASE_URL, api_key=self.api_key, timeout=self.timeout_s,
                default_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "JARVIS",
                },
            )
        return self._client

    async def available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _render(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Canonical -> OpenAI. `arguments` is a JSON string here, and tool
        results are matched back by `tool_call_id`."""
        out: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")

            if role == "assistant" and message.get("tool_calls"):
                out.append({
                    "role": "assistant",
                    "content": message.get("content", "") or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in message["tool_calls"]
                    ],
                })
            elif role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id", ""),
                    "content": message.get("content", ""),
                })
            else:
                out.append({k: v for k, v in message.items()
                            if k in ("role", "content")})
        return out

    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None) -> ChatResult:
        try:
            response = await self._get().chat.completions.create(
                model=self.model, messages=self._render(messages),
                tools=tools or None, temperature=0.4, max_tokens=800,
            )
        except Exception as exc:
            # Raised straight through so the agent's escalation logic drops
            # back to the local model. Logged distinctly because "you're out of
            # free requests" and "the model is broken" need different fixes.
            text = str(exc)
            if "429" in text or "rate" in text.lower():
                log.warning("openrouter rate limited (free tier is ~20/min, "
                            "~200/day) — falling back to the local model")
            elif "404" in text or "not found" in text.lower():
                log.warning("openrouter model %r not found — check "
                            "OPENROUTER_MODEL in .env", self.model)
            else:
                log.warning("openrouter failed: %s", text[:200])
            raise

        if not response.choices:
            raise RuntimeError("openrouter returned no choices")

        choice = response.choices[0].message
        calls: list[ToolCall] = []
        for raw in choice.tool_calls or []:
            try:
                arguments = json.loads(raw.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(id=raw.id or uuid.uuid4().hex,
                         name=raw.function.name, arguments=arguments)
            )

        return ChatResult(content=choice.content or "", tool_calls=calls, route="cloud")
