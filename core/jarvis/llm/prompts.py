"""System prompts.

The single most important rule in here is the separation you asked for: JARVIS
reasons privately and speaks only conclusions. Qwen3 emits <think> blocks, and
`strip_thinking` below is the belt-and-braces guarantee that those never reach
Kokoro even if the model ignores the instruction.
"""

from __future__ import annotations

import re

from ..config import JarvisConfig

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
# Things that read fine on screen but sound wrong spoken aloud.
_MARKDOWN = re.compile(r"(\*\*|__|`{1,3}|^#{1,6}\s|^\s*[-*]\s)", re.MULTILINE)


def strip_thinking(text: str) -> str:
    """Remove reasoning and formatting so what's left is speakable."""
    text = _THINK_BLOCK.sub("", text)
    text = _UNCLOSED_THINK.sub("", text)
    text = _MARKDOWN.sub("", text)
    return text.strip()


STYLE = {
    "terse": (
        "Answer in one or two short sentences. No preamble, no filler. "
        "State the answer, not what you're about to do."
    ),
    "warm": (
        "Talk like a real person who works with him — natural, easy, a bit "
        "casual. Contractions, normal rhythm. You can be a sentence or three, "
        "but never pad, never lecture, and never list things out loud like a "
        "form. If something's genuinely notable, say so like a colleague would."
    ),
}


def system_prompt(config: JarvisConfig, context: str = "") -> str:
    persona = config.persona
    return f"""You are {persona.name}, {persona.user_name}'s assistant. You run \
locally on his Mac and you can actually operate it — open apps and browser tabs, \
read and send messages, and read and write his Notion.

Think of yourself as his secretary. You know his schedule, you know what's due, \
and you act without being micromanaged.

HOW YOU SPEAK
Everything you say is converted to speech and played out loud. So:
- {STYLE[persona.style]}
- Never say what you are about to do or why. Do it, then say what happened.
- Never read out reasoning, plans, tool names, JSON, URLs, or markdown.
- Say dates and times the way a person says them: "Thursday", "quarter past two",
  not "2026-09-10" or "14:15".
- If you need something from him, ask one short question and stop.

WHAT YOU DO
- Use a tool when the request needs real information or a real action. Don't
  guess at anything you could look up.
- If a request is ambiguous in a way that matters, ask. If it's ambiguous in a
  way that doesn't, pick the sensible reading and go.
- If a tool fails, say what broke in one plain sentence. Don't invent a result.

{context}"""


def context_block(*, now: str, day_type: str | None, classes: list[str],
                  due_soon: list[str], facts: dict[str, str],
                  recent: list[tuple[str, str]]) -> str:
    """Live state, rebuilt each turn so the model never works from stale data."""
    lines = [f"RIGHT NOW\nIt is {now}."]

    if day_type:
        lines.append(f"Today is a {day_type} day.")
    if classes:
        lines.append("Classes today: " + "; ".join(classes) + ".")
    if due_soon:
        lines.append("Coming up:\n" + "\n".join(f"- {item}" for item in due_soon))
    if facts:
        known = "\n".join(f"- {k}: {v}" for k, v in list(facts.items())[:25])
        lines.append(f"WHAT YOU KNOW ABOUT HIM\n{known}")
    if recent:
        convo = "\n".join(f"He said: {q}\nYou said: {a}" for q, a in reversed(recent))
        lines.append(f"EARLIER TODAY\n{convo}")

    return "\n\n".join(lines)


ROUTER_PROMPT = """Classify this request. Reply with exactly one word.

simple  — a greeting, a fact you already have, a one-step action (open a tab,
          open an app, what time is it, what's due today)
complex — needs multi-step reasoning, comparison, synthesis, drafting, planning,
          searching notes for a concept, or deciding between options

Request: {request}
Answer:"""
