"""iMessage — reading and sending.

There is no API for this, so both halves are unofficial:

Reading is a direct read of ~/Library/Messages/chat.db, the SQLite database
Messages.app keeps. Opened strictly read-only (immutable=1) so JARVIS can never
corrupt your message history, even if it crashes mid-query. Needs Full Disk
Access.

Sending is AppleScript against Messages.app. Needs Automation permission, which
macOS prompts for on first use.

Two things that make chat.db annoying, both handled below:
  - Timestamps are nanoseconds since 2001-01-01, not the Unix epoch. Older
    macOS used seconds, so both are detected.
  - On modern macOS `text` is usually NULL and the real body is a serialised
    NSAttributedString in `attributedBody`, which has to be picked apart by
    hand.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .mac import osascript

log = logging.getLogger("jarvis.messages")

CHAT_DB = Path.home() / "Library" / "Messages" / "chat.db"
APPLE_EPOCH = 978307200  # 2001-01-01 in Unix seconds


@dataclass
class Message:
    id: int
    text: str
    from_me: bool
    handle: str
    contact: str
    sent_at: datetime
    chat_name: str = ""

    def spoken(self, now: datetime) -> str:
        who = "you" if self.from_me else self.contact
        return f"{who}: {self.text}" if self.text else f"{who} sent an attachment"


def _decode_attributed_body(blob: bytes | None) -> str:
    """Pull the plain text out of a serialised NSAttributedString.

    Apple's typedstream format isn't documented and there's no stdlib decoder,
    so this reads the length-prefixed NSString payload directly. It is
    deliberately forgiving — a message we can't decode should come back empty,
    not raise.
    """
    if not blob:
        return ""
    try:
        if b"NSString" not in blob:
            return ""
        tail = blob.split(b"NSString", 1)[1][5:]
        if not tail:
            return ""

        if tail[0] == 0x81:            # 2-byte little-endian length
            length = int.from_bytes(tail[1:3], "little")
            payload = tail[3:3 + length]
        else:                          # single-byte length
            length = tail[0]
            payload = tail[1:1 + length]

        return payload.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _apple_time(raw: int | None) -> datetime:
    if not raw:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    seconds = raw / 1_000_000_000 if raw > 1e11 else raw
    return datetime.fromtimestamp(seconds + APPLE_EPOCH, tz=timezone.utc)


class MessagesUnavailable(RuntimeError):
    pass


class Messages:
    def __init__(self, blocklist: list[str] | None = None):
        self._blocklist = {n.lower() for n in (blocklist or [])}
        # handle (phone/email) -> display name, resolved via Contacts.app.
        # Cached because each lookup is a ~200ms AppleScript round trip.
        self._names: dict[str, str] = {}

    @property
    def available(self) -> bool:
        return CHAT_DB.exists()

    # ---- reading ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if not CHAT_DB.exists():
            raise MessagesUnavailable(
                "chat.db not found — is this a Mac with Messages set up?"
            )
        try:
            # immutable=1 means we never write, never lock, and never risk the
            # file even if Messages.app is mid-write.
            conn = sqlite3.connect(f"file:{CHAT_DB}?immutable=1", uri=True)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.OperationalError as exc:
            raise MessagesUnavailable(
                "can't open chat.db — grant Full Disk Access to your terminal "
                "in System Settings > Privacy & Security"
            ) from exc

    def _query(self, where: str = "", params: tuple = (),
               limit: int = 30) -> list[Message]:
        sql = f"""
            SELECT m.ROWID       AS id,
                   m.text        AS text,
                   m.attributedBody AS body,
                   m.is_from_me  AS from_me,
                   m.date        AS date,
                   h.id          AS handle,
                   c.display_name AS chat_name
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            LEFT JOIN chat c ON c.ROWID = cmj.chat_id
            {where}
            ORDER BY m.date DESC
            LIMIT ?
        """
        conn = self._connect()
        try:
            rows = conn.execute(sql, (*params, limit)).fetchall()
        finally:
            conn.close()

        out: list[Message] = []
        for row in rows:
            text = (row["text"] or "").strip() or _decode_attributed_body(row["body"])
            handle = row["handle"] or ""
            out.append(Message(
                id=row["id"], text=text, from_me=bool(row["from_me"]),
                handle=handle, contact=self._names.get(handle, handle),
                sent_at=_apple_time(row["date"]),
                chat_name=row["chat_name"] or "",
            ))
        return out

    async def recent(self, limit: int = 20) -> list[Message]:
        messages = await asyncio.to_thread(self._query, "", (), limit)
        await self._name_all(messages)
        return messages

    async def unread(self, since_hours: int = 24) -> list[Message]:
        """Incoming messages you haven't replied to since."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        recent = await self.recent(limit=60)

        # Anything from someone after your last outgoing message to them.
        last_sent: dict[str, datetime] = {}
        for message in recent:
            if message.from_me and message.handle not in last_sent:
                last_sent[message.handle] = message.sent_at

        return [
            m for m in recent
            if not m.from_me and m.sent_at > cutoff
            and m.sent_at > last_sent.get(m.handle, datetime.fromtimestamp(0, timezone.utc))
        ]

    async def with_contact(self, name: str, limit: int = 20) -> list[Message]:
        handle = await self.resolve_contact(name)
        if not handle:
            return []
        digits = "".join(c for c in handle if c.isdigit())[-10:]
        messages = await asyncio.to_thread(
            self._query, "WHERE h.id LIKE ?", (f"%{digits}%",), limit
        )
        await self._name_all(messages)
        return messages

    # ---- contacts --------------------------------------------------------

    async def resolve_contact(self, name: str) -> str | None:
        """Name to phone number, via Contacts.app."""
        if "@" in name or any(c.isdigit() for c in name):
            return name  # already a handle

        safe = name.replace('"', "")
        try:
            result = await osascript(
                f'tell application "Contacts"\n'
                f'  set matches to (every person whose name contains "{safe}")\n'
                f'  if (count of matches) = 0 then return ""\n'
                f'  set thePerson to item 1 of matches\n'
                f'  if (count of phones of thePerson) > 0 then\n'
                f'    return value of first phone of thePerson\n'
                f'  else if (count of emails of thePerson) > 0 then\n'
                f'    return value of first email of thePerson\n'
                f'  end if\n'
                f'  return ""\n'
                f'end tell'
            )
        except Exception as exc:
            log.warning("contact lookup failed for %r: %s", name, exc)
            return None

        handle = result.strip()
        if handle:
            self._names[handle] = name
        return handle or None

    async def _name_all(self, messages: list[Message]) -> None:
        """Fill in display names for handles we haven't seen before."""
        unknown = {m.handle for m in messages
                   if m.handle and m.handle not in self._names}
        for handle in list(unknown)[:12]:  # bounded — each is an AppleScript call
            safe = handle.replace('"', "")
            try:
                name = (await osascript(
                    f'tell application "Contacts"\n'
                    f'  set matches to (every person whose value of phones contains '
                    f'"{safe}" or value of emails contains "{safe}")\n'
                    f'  if (count of matches) = 0 then return ""\n'
                    f'  return name of item 1 of matches\n'
                    f'end tell'
                )).strip()
            except Exception:
                name = ""
            self._names[handle] = name or handle

        for message in messages:
            message.contact = self._names.get(message.handle, message.handle)

    # ---- sending ---------------------------------------------------------

    def is_blocked(self, name: str) -> bool:
        return name.lower().strip() in self._blocklist

    async def send(self, to: str, text: str) -> dict[str, Any]:
        if shutil.which("osascript") is None:
            return {"ok": False, "error": "macOS only"}
        if self.is_blocked(to):
            return {"ok": False, "error": "blocked",
                    "detail": f"{to} is on your never-text list."}

        handle = await self.resolve_contact(to)
        if not handle:
            return {"ok": False, "error": "no_contact",
                    "detail": f"I couldn't find {to} in your contacts."}

        safe_text = text.replace("\\", "\\\\").replace('"', '\\"')
        safe_handle = handle.replace('"', "")
        try:
            await osascript(
                f'tell application "Messages"\n'
                f'  set targetService to 1st account whose service type = iMessage\n'
                f'  set targetBuddy to participant "{safe_handle}" of targetService\n'
                f'  send "{safe_text}" to targetBuddy\n'
                f'end tell'
            )
        except Exception as exc:
            return {"ok": False, "error": "send_failed", "detail": str(exc)}

        log.info("sent to %s (%s)", to, handle)
        return {"ok": True, "to": to, "handle": handle, "text": text}
