"""CLI: `jarvis run`, `jarvis doctor`, `jarvis say`."""

from __future__ import annotations

import asyncio

import typer

from . import log as jlog
from .config import config, secrets

app = typer.Typer(add_completion=False, help="JARVIS — local voice assistant")


@app.command()
def run(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Start the daemon: microphone, wake word, agent, speech."""
    jlog.setup(verbose)
    from .daemon import Daemon

    daemon = Daemon(config, secrets)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        typer.echo("\nstopped")


@app.command()
def doctor() -> None:
    """Check every dependency and permission, and say exactly what's missing."""
    jlog.setup(False)
    from .doctor import run_checks

    raise typer.Exit(code=asyncio.run(run_checks()))


@app.command()
def say(text: str) -> None:
    """Speak a line through Kokoro. Quickest way to audition a voice."""
    jlog.setup(False)
    from .audio.tts import TTS

    async def _speak() -> None:
        tts = TTS(config.voice.tts_voice, config.voice.tts_speed)
        await asyncio.to_thread(tts.load)
        await tts.speak(text)

    asyncio.run(_speak())


@app.command()
def ask(text: str) -> None:
    """Send one request through the agent and print the reply. No microphone."""
    jlog.setup(True)
    from .daemon import Daemon

    async def _ask() -> None:
        daemon = Daemon(config, secrets)
        await daemon.bus.start()
        from .skills.tools import register_all

        register_all(daemon.registry, daemon.memory, daemon.schedule, daemon.ledger)
        await daemon.router.probe()
        typer.echo(await daemon.agent.respond(text))
        await daemon.bus.stop()

    asyncio.run(_ask())


if __name__ == "__main__":
    app()
