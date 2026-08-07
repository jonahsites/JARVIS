"""Shared LLM types, so the agent doesn't care whether it's talking to Ollama
or OpenRouter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    route: str = "local"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    name: str

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult: ...

    async def available(self) -> bool: ...


# ---------------------------------------------------------------------------
# Canonical conversation format.
#
# Providers disagree about how tool traffic is represented — OpenAI wants
# `arguments` as a JSON *string* and pairs results by `tool_call_id`; Ollama
# wants a *dict* and pairs by `tool_name`. Sending one shape to the other fails
# validation outright.
#
# So the agent builds messages in the neutral shape below and each client
# converts on the way out. That also means a conversation can be handed from
# the local model to the cloud one mid-flight (which is exactly what escalation
# does) without carrying the wrong provider's formatting with it.
# ---------------------------------------------------------------------------


def assistant_turn(result: ChatResult) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.content,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in result.tool_calls
        ],
    }


def tool_turn(call: ToolCall, output: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": output,
    }
