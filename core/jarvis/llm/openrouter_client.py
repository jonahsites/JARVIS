"""Escalation path. Used when the local 8B isn't the right tool for the job."""

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
        response = await self._get().chat.completions.create(
            model=self.model, messages=self._render(messages), tools=tools or None,
            temperature=0.4, max_tokens=800,
        )

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
