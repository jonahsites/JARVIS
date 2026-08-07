"""`jarvis test <stage>` — bring JARVIS up one subsystem at a time.

doctor answers "is it installed and permitted". This answers "does it actually
work", which is a different question and the one that bites on first run.

Run them in order. Each stage is independent, so a failure tells you exactly
which integration is wrong instead of handing you one opaque traceback from a
pipeline with six moving parts.

    jarvis test audio      mic records, speakers play
    jarvis test tts        Kokoro says a line
    jarvis test stt        you speak, it prints what it heard
    jarvis test wake       say "hey Jarvis", watch the score
    jarvis test llm        one round trip through the router
    jarvis test mac        frontmost app, Chrome tabs
    jarvis test notion     token, courses, assignments, a small crawl
    jarvis test messages   reads your last few (never sends)
    jarvis test all        every non-interactive stage
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import numpy as np
from rich.console import Console

from .config import config, secrets

console = Console()

OK = "[green]pass[/green]"
BAD = "[red]FAIL[/red]"
MEH = "[yellow]warn[/yellow]"


def _ok(message: str) -> None:
    console.print(f"  {OK}  {message}")


def _bad(message: str, fix: str = "") -> None:
    console.print(f"  {BAD}  {message}")
    if fix:
        console.print(f"        [dim]{fix}[/dim]")


def _warn(message: str) -> None:
    console.print(f"  {MEH}  {message}")


def _stage(name: str) -> None:
    console.rule(f"[bold]{name}")


# ---------------------------------------------------------------------------


async def test_audio(seconds: float = 3.0) -> bool:
    """Record from the mic, play it back. Proves both halves independently."""
    _stage("audio")
    try:
        import sounddevice as sd
    except Exception as exc:
        _bad(f"sounddevice won't import: {exc}", "brew install portaudio && pip install -e .")
        return False

    try:
        console.print(f"  recording {seconds:.0f}s — say something…")
        recording = sd.rec(int(seconds * 16000), samplerate=16000,
                           channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:
        _bad(f"could not record: {exc}",
             "System Settings > Privacy & Security > Microphone > add your terminal")
        return False

    peak = float(np.abs(recording).max())
    rms = float(np.sqrt(np.mean(recording ** 2)))
    if peak < 0.005:
        _bad(f"recorded silence (peak {peak:.4f})",
             "Wrong input device, or the mic is muted. Check:  "
             "python -c \"import sounddevice as sd; print(sd.query_devices())\"")
        return False
    _ok(f"mic works — peak {peak:.3f}, rms {rms:.4f}")

    try:
        console.print("  playing it back…")
        sd.play(recording, samplerate=16000)
        sd.wait()
        _ok("speakers work")
    except Exception as exc:
        _bad(f"playback failed: {exc}")
        return False
    return True


async def test_tts() -> bool:
    _stage("text to speech")
    from .audio.tts import TTS

    tts = TTS(config.voice.tts_voice, config.voice.tts_speed)
    try:
        started = time.monotonic()
        await asyncio.to_thread(tts.load)
        _ok(f"Kokoro loaded in {time.monotonic() - started:.1f}s "
            f"(voice {config.voice.tts_voice})")
    except Exception as exc:
        _bad(f"Kokoro won't load: {exc}", "pip install kokoro soundfile")
        return False

    try:
        started = time.monotonic()
        await tts.speak("If you can hear this, speech synthesis is working.")
        _ok(f"spoke in {time.monotonic() - started:.1f}s")
    except Exception as exc:
        _bad(f"speaking failed: {exc}")
        return False
    return True


async def test_stt(seconds: float = 5.0) -> bool:
    _stage("speech to text")
    import sounddevice as sd

    from .audio.stt import STT

    stt = STT(config.voice.stt_engine, config.voice.qwen_model,
              config.voice.stt_model, config.voice.whisper_model)
    try:
        started = time.monotonic()
        await asyncio.to_thread(stt.load)
        _ok(f"engine loaded in {time.monotonic() - started:.1f}s")
    except Exception as exc:
        _bad(f"no engine could load: {exc}",
             "pip install mlx-qwen3-asr  (or parakeet-mlx, or mlx-whisper)")
        return False

    console.print(f"  say a full sentence — recording {seconds:.0f}s…")
    recording = sd.rec(int(seconds * 16000), samplerate=16000,
                       channels=1, dtype="float32")
    sd.wait()

    try:
        started = time.monotonic()
        text = await asyncio.to_thread(stt.transcribe, recording[:, 0])
        elapsed = time.monotonic() - started
    except Exception as exc:
        _bad(f"transcription failed: {exc}")
        return False

    if not text:
        _bad("transcribed nothing", "Too quiet, or the mic picked up the wrong device.")
        return False

    _ok(f'heard: "{text}"')
    _ok(f"{elapsed:.2f}s for {seconds:.0f}s of audio "
        f"({seconds / max(elapsed, 0.01):.1f}x realtime)")
    return True


async def test_wake(seconds: float = 15.0) -> bool:
    _stage("wake word")
    import sounddevice as sd

    from .audio.wake import WakeWord

    wake = WakeWord(config.voice.wake_model, config.voice.wake_threshold)
    try:
        await asyncio.to_thread(wake.load)
        _ok(f"model loaded ({config.voice.wake_model})")
    except Exception as exc:
        _bad(f"wake model won't load: {exc}", "pip install openwakeword onnxruntime")
        return False

    console.print(f'  say "hey Jarvis" a few times over the next {seconds:.0f}s…')
    hits = 0
    peak_score = 0.0
    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def on_audio(indata, frames, time_info, status) -> None:
        nonlocal hits, peak_score
        pcm16 = (np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16)
        if wake.feed(pcm16):
            hits += 1
            console.print(f"    [green]detected[/green] (#{hits})")

    with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                        blocksize=1280, callback=on_audio):
        try:
            await asyncio.wait_for(done, timeout=seconds)
        except asyncio.TimeoutError:
            pass

    if hits == 0:
        _bad("never detected the wake word",
             f"Lower wake_threshold (now {config.voice.wake_threshold}) to 0.4 "
             "in config.toml, or check the mic with `jarvis test audio`.")
        return False
    _ok(f"detected {hits} time(s)")
    return True


async def test_llm() -> bool:
    _stage("language model")
    from .agent.loop import Agent
    from .agent.capabilities import CapabilityLedger
    from .agent.registry import Registry
    from .bus import Bus
    from .llm.ollama_client import OllamaClient
    from .llm.openrouter_client import OpenRouterClient
    from .llm.router import Router
    from .memory.db import Memory
    from .skills.schedule import ScheduleResolver
    from .skills.tools import register_all

    router = Router(
        OllamaClient(secrets.ollama_host, secrets.ollama_model, config.llm.local_timeout_s),
        OpenRouterClient(secrets.openrouter_api_key, secrets.openrouter_model,
                         config.llm.cloud_timeout_s),
        config.llm,
    )
    local_ok, cloud_ok = await router.probe()

    if local_ok:
        _ok(f"Ollama reachable, {secrets.ollama_model} present")
    else:
        _bad(f"Ollama not usable at {secrets.ollama_host}",
             f"ollama serve  &&  ollama pull {secrets.ollama_model}")
    if cloud_ok:
        _ok(f"OpenRouter key set ({secrets.openrouter_model})")
    else:
        _warn("no OpenRouter key — complex requests stay on the 8B model")
    if not local_ok and not cloud_ok:
        return False

    bus = Bus(0)
    memory = Memory()
    ledger = CapabilityLedger(memory, bus)
    registry = Registry(ledger)
    schedule = ScheduleResolver(config.schedule)
    register_all(registry, memory, schedule, ledger)

    agent = Agent(config, router, registry, memory, lambda: "")

    # A question it can only answer by calling a tool — proves tool-calling
    # works, which is the part 8B models most often get wrong.
    for question in ["What time is it?", "Is today an A day or a B day?"]:
        try:
            started = time.monotonic()
            reply = await agent.respond(question)
            elapsed = time.monotonic() - started
        except Exception as exc:
            _bad(f'"{question}" failed: {exc}')
            return False

        if not reply:
            _bad(f'"{question}" produced an empty reply')
            return False
        _ok(f'"{question}" -> "{reply}"  [{elapsed:.1f}s]')

    if "<think" in reply.lower():
        _bad("reasoning leaked into the spoken reply",
             "strip_thinking() didn't catch it — please send me this output.")
        return False
    _ok("no reasoning leaked into speech")
    return True


async def test_mac() -> bool:
    _stage("macOS control")
    from .skills import mac

    try:
        app = await mac.frontmost_app()
        _ok(f"frontmost app: {app}")
    except Exception as exc:
        _bad(f"AppleScript blocked: {exc}",
             "System Settings > Privacy & Security > Accessibility > add your terminal, "
             "then fully quit it (Cmd-Q) and reopen")
        return False

    try:
        tabs = await mac.list_tabs()
        if tabs:
            _ok(f"{len(tabs)} Chrome tab(s); active: "
                f"{next((t['title'][:50] for t in tabs if t.get('active')), 'none')}")
        else:
            _warn("no Chrome tabs found — is Chrome running?")
    except Exception as exc:
        _bad(f"can't read Chrome: {exc}", "Approve the Automation prompt on first use")
        return False
    return True


async def test_notion() -> bool:
    _stage("notion")
    from .skills.notion.assignments import Assignments
    from .skills.notion.client import NotionClient
    from .skills.notion.notes import NoteCrawler, NoteIndex

    if not secrets.notion_token:
        _bad("NOTION_TOKEN not set", "notion.so/my-integrations, then add it to .env")
        return False

    client = NotionClient(secrets.notion_token)
    try:
        me = await client.whoami()
        _ok(f"authenticated as {me.get('name', '?')}")
    except Exception as exc:
        _bad(f"auth failed: {exc}", "Check the token in .env")
        await client.close()
        return False

    assignments = Assignments(client)
    try:
        courses = await assignments.load_courses()
        if not courses:
            _bad("no courses returned",
                 "Share your Dashboard page with the integration: "
                 "page > ... > Connections > add it")
            await client.close()
            return False
        _ok(f"{len(courses)} courses: " + ", ".join(c.name for c in courses[:4]) + "…")
    except Exception as exc:
        _bad(f"couldn't read Courses: {exc}",
             "Usually means the database isn't shared with the integration")
        await client.close()
        return False

    from datetime import date

    try:
        today = date.today()
        upcoming = await assignments.upcoming(today=today, limit=5)
        if upcoming:
            _ok(f"{len(upcoming)} assignments due soon; top: {upcoming[0].spoken(today)}")
        else:
            _warn("no upcoming assignments (fine if the term hasn't started)")
    except Exception as exc:
        _bad(f"couldn't read Assignments: {exc}")

    try:
        days = await assignments.days_off()
        _ok(f"{len(days)} days off loaded")
    except Exception as exc:
        _warn(f"Days Off unreadable ({exc}) — A/B rotation won't skip holidays")

    console.print("  crawling notes (this can take a minute)…")
    index = NoteIndex()
    crawler = NoteCrawler(client, index)
    try:
        started = time.monotonic()
        written = await crawler.crawl()
        _ok(f"indexed {written} notes in {time.monotonic() - started:.0f}s "
            f"({index.count()} total)")
        if index.count():
            sample = index.search("the")[:1]
            if sample:
                _ok(f"search works — e.g. \"{sample[0]['title']}\"")
    except Exception as exc:
        _bad(f"note crawl failed: {exc}")

    index.close()
    await client.close()
    return True


async def test_messages() -> bool:
    """Reads only. This never sends anything."""
    _stage("messages (read only)")
    from .skills.messages import Messages

    messages = Messages(config.messages.blocklist)
    if not messages.available:
        _bad("chat.db not found", "Is Messages.app set up on this Mac?")
        return False

    try:
        recent = await messages.recent(limit=5)
    except Exception as exc:
        _bad(f"can't read chat.db: {exc}",
             "System Settings > Privacy & Security > Full Disk Access > add your "
             "terminal, then FULLY quit it (Cmd-Q) and reopen")
        return False

    if not recent:
        _warn("no messages found — readable, but empty")
        return True

    _ok(f"read {len(recent)} recent message(s)")
    decoded = sum(1 for m in recent if m.text)
    if decoded == 0:
        _bad("every message body came back empty",
             "attributedBody decoding failed on your macOS version — send me this output")
        return False
    _ok(f"{decoded}/{len(recent)} bodies decoded")
    for message in recent[:3]:
        who = "you" if message.from_me else (message.contact or "?")
        console.print(f"    [dim]{who}: {message.text[:60]}[/dim]")
    return True


STAGES: dict[str, Callable] = {
    "audio": test_audio,
    "tts": test_tts,
    "stt": test_stt,
    "wake": test_wake,
    "llm": test_llm,
    "mac": test_mac,
    "notion": test_notion,
    "messages": test_messages,
}

# Everything that doesn't need you to speak into a microphone.
NON_INTERACTIVE = ["tts", "llm", "mac", "notion", "messages"]


async def run(stage: str) -> int:
    if stage == "all":
        results: dict[str, bool] = {}
        for name in NON_INTERACTIVE:
            try:
                results[name] = await STAGES[name]()
            except Exception as exc:
                _bad(f"{name} raised: {exc}")
                results[name] = False

        console.rule("[bold]summary")
        for name, passed in results.items():
            console.print(f"  {OK if passed else BAD}  {name}")
        console.print(
            "\n  [dim]Interactive stages need you at the mic:[/dim] "
            "jarvis test audio · stt · wake"
        )
        return 0 if all(results.values()) else 1

    if stage not in STAGES:
        console.print(f"[red]unknown stage {stage!r}[/red]")
        console.print("  " + " · ".join([*STAGES, "all"]))
        return 2

    return 0 if await STAGES[stage]() else 1
