"""Tool registry.

Every tool declares the capability it needs. The registry is the only place
that calls into the ledger, so there is no way to add a tool that silently
skips the first-use prompt — you'd have to delete the `require` line, which is
obvious in review.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .capabilities import CapabilityDenied, CapabilityLedger

log = logging.getLogger("jarvis.registry")

ToolFn = Callable[..., Awaitable[Any]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    capability: str | None
    fn: ToolFn

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Registry:
    def __init__(self, ledger: CapabilityLedger):
        self._tools: dict[str, Tool] = {}
        self._ledger = ledger

    def register(self, name: str, description: str, parameters: dict[str, Any],
                 capability: str | None = None) -> Callable[[ToolFn], ToolFn]:
        def decorator(fn: ToolFn) -> ToolFn:
            self._tools[name] = Tool(name, description, parameters, capability, fn)
            return fn

        return decorator

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"no such tool: {name}"}

        if tool.capability:
            try:
                # Capability id only — never the arguments. Approving
                # "open a tab" approves every tab, which is the rule you set.
                await self._ledger.require(tool.capability, title=tool.description)
            except CapabilityDenied:
                return {"error": "not_permitted",
                        "detail": f"I don't have permission for {tool.capability} yet."}

        try:
            result = tool.fn(**arguments)
            if inspect.isawaitable(result):
                result = await result
            return result
        except TypeError as exc:
            log.warning("bad arguments for %s: %s", name, exc)
            return {"error": "bad_arguments", "detail": str(exc)}
        except Exception as exc:
            log.exception("tool %s failed", name)
            return {"error": "tool_failed", "detail": str(exc)}
