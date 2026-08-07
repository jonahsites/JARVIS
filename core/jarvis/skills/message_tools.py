"""Message tools exposed to the model."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from ..agent.registry import Registry
from .messages import Messages, MessagesUnavailable
from .outbox import Outbox

log = logging.getLogger("jarvis.message_tools")

NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}}


def _ago(when: datetime) -> str:
    """Spoken-shaped relative time. Nobody says '2026-08-07T14:32Z'."""
    delta = datetime.now(timezone.utc) - when
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    return "yesterday" if days == 1 else f"{days} days ago"


def register_messages(registry: Registry, messages: Messages, outbox: Outbox,
                      speak: Callable[[str], Awaitable[Any]]) -> None:

    def offline() -> dict[str, Any] | None:
        if messages.available:
            return None
        return {
            "error": "messages_unavailable",
            "say": "I can't get at your messages — I probably need Full Disk Access.",
        }

    @registry.register(
        "read_messages",
        "His recent iMessages. Use for 'any messages', 'what did X say', "
        "'did anyone text me'.",
        {
            "type": "object",
            "properties": {
                "contact": {"type": "string",
                            "description": "Optional — just this person's thread."},
                "limit": {"type": "integer", "description": "Default 15."},
            },
        },
        capability="messages.read",
    )
    async def read_messages(contact: str | None = None,
                            limit: int = 15) -> dict[str, Any]:
        if (down := offline()):
            return down
        try:
            found = (await messages.with_contact(contact, limit) if contact
                     else await messages.recent(limit))
        except MessagesUnavailable as exc:
            return {"error": "messages_unavailable", "say": str(exc)}

        if contact and not found:
            return {"messages": [],
                    "say": f"I don't see any messages with {contact}."}

        return {
            "messages": [
                {
                    "from": "you" if m.from_me else m.contact,
                    "text": m.text,
                    "when": _ago(m.sent_at),
                }
                for m in found if m.text
            ]
        }

    @registry.register(
        "check_unread",
        "Messages from other people that he hasn't replied to yet. Use for "
        "'do I need to reply to anyone', 'anything I'm missing'.",
        {
            "type": "object",
            "properties": {"since_hours": {"type": "integer", "description": "Default 24."}},
        },
        capability="messages.read",
    )
    async def check_unread(since_hours: int = 24) -> dict[str, Any]:
        if (down := offline()):
            return down
        try:
            found = await messages.unread(since_hours)
        except MessagesUnavailable as exc:
            return {"error": "messages_unavailable", "say": str(exc)}

        return {
            "count": len(found),
            "messages": [
                {"from": m.contact, "text": m.text, "when": _ago(m.sent_at)}
                for m in found if m.text
            ],
        }

    @registry.register(
        "send_message",
        "Send an iMessage. Write the message as HE would write it — first "
        "person, casual, no quotation marks, no 'he says'. It is read aloud "
        "and then sent automatically, so make sure it is exactly what should go.",
        {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Contact name, e.g. 'Mom'"},
                "text": {"type": "string", "description": "The message itself"},
            },
            "required": ["to", "text"],
        },
        capability="messages.send",
    )
    async def send_message(to: str, text: str) -> dict[str, Any]:
        if (down := offline()):
            return down
        if messages.is_blocked(to):
            return {"ok": False,
                    "say": f"{to} is on your never-text list, so I didn't."}

        # Resolve before announcing — no point reading out a text to someone
        # who isn't in his contacts.
        handle = await messages.resolve_contact(to)
        if not handle:
            return {"ok": False, "error": "no_contact",
                    "say": f"I couldn't find {to} in your contacts."}

        result = await outbox.queue(
            summary=f"Texting {to}: {text}",
            send=lambda: messages.send(to, text),
            speak=speak,
        )
        if result.get("ok"):
            result["say"] = ""  # already spoken during the cancel window
        return result

    @registry.register(
        "cancel_send",
        "Stop a message that's about to go out. Use the instant he says stop, "
        "wait, no, cancel, or don't send that.",
        NO_ARGS,
    )
    async def cancel_send() -> dict[str, Any]:
        stopped = outbox.cancel()
        return {
            "ok": True,
            "cancelled": stopped,
            "say": "Okay, didn't send it." if stopped else "Nothing was queued.",
        }
