"""The voice loop: mic -> wake word -> VAD -> STT -> agent -> Kokoro -> speakers.

All audio lives in this process. The browser tab is display only — it receives
state and level over the WebSocket and renders the glob. That means the loop
keeps working with no tab open at all, which is what you want from something
you shout at.

The mic callback runs on a PortAudio thread, so it does the absolute minimum
(copy, queue) and hands off to the asyncio loop for everything else.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

import numpy as np

from ..bus import EV_LEVEL, EV_TRANSCRIPT, Bus
from ..config import JarvisConfig
from ..state import AgentState, StateMachine
from .stt import STT
from .tts import TTS
from .vad import UtteranceDetector
from .wake import WakeWord

log = logging.getLogger("jarvis.pipeline")

SAMPLE_RATE = 16_000
BLOCK = 1280  # 80 ms — matches openWakeWord's frame size

_YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "fine", "go", "do",
        "please", "affirmative", "course", "definitely", "absolutely", "alright"}
_NO = {"no", "nope", "nah", "don't", "dont", "stop", "never", "negative", "cancel"}


def _is_yes(answer: str) -> bool:
    """Anything not clearly affirmative is treated as no — the safe default."""
    words = {w.strip(".,!?").lower() for w in answer.split()}
    if words & _NO:
        return False
    return bool(words & _YES)

# Handler the agent registers: transcript in, spoken reply out.
Responder = Callable[[str], Awaitable[str]]


class VoicePipeline:
    def __init__(self, config: JarvisConfig, bus: Bus, state: StateMachine):
        self._config = config
        self._bus = bus
        self._state = state
        self._loop: asyncio.AbstractEventLoop | None = None
        self._responder: Responder | None = None

        self.wake = WakeWord(config.voice.wake_model, config.voice.wake_threshold)
        self.vad = UtteranceDetector(config.voice.silence_ms, config.voice.max_utterance_s)
        self.stt = STT(config.voice.stt_engine, config.voice.stt_model,
                       config.voice.whisper_model)
        self.tts = TTS(config.voice.tts_voice, config.voice.tts_speed,
                       on_level=self._push_level_threadsafe)

        self._frames: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        self._stream = None
        self._listening = False
        # Suppresses wake detection while Kokoro is playing, so JARVIS doesn't
        # hear its own voice say "Jarvis" and wake itself up.
        self._muted_until = 0.0
        # When JARVIS asked *you* something, the next thing you say answers it
        # instead of starting a new request.
        self._awaiting_answer: asyncio.Future[str] | None = None

    def set_responder(self, responder: Responder) -> None:
        self._responder = responder

    # ---- model loading ---------------------------------------------------

    async def load(self) -> None:
        log.info("loading speech models (first run downloads a few GB)")
        started = time.monotonic()
        await asyncio.gather(
            asyncio.to_thread(self.wake.load),
            asyncio.to_thread(self.vad.load),
            asyncio.to_thread(self.stt.load),
            asyncio.to_thread(self.tts.load),
        )
        log.info("speech models ready in %.1fs", time.monotonic() - started)

    # ---- mic -------------------------------------------------------------

    def _on_audio(self, indata, frames, time_info, status) -> None:
        """PortAudio thread. Keep this cheap."""
        if status:
            log.debug("audio status: %s", status)
        if self._loop is None:
            return
        chunk = indata[:, 0].copy()
        try:
            self._loop.call_soon_threadsafe(self._frames.put_nowait, chunk)
        except (asyncio.QueueFull, RuntimeError):
            pass  # dropping a frame beats blocking the audio thread

    def _push_level_threadsafe(self, rms: float) -> None:
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._bus.emit(EV_LEVEL, rms=min(1.0, rms * 8)))
            )
        except RuntimeError:
            pass

    async def start(self) -> None:
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=BLOCK, callback=self._on_audio,
        )
        self._stream.start()
        log.info("microphone open")
        await self._state.transition(AgentState.IDLE)
        asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    # ---- the loop --------------------------------------------------------

    async def _run(self) -> None:
        while True:
            chunk = await self._frames.get()

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if self._state.state in (AgentState.IDLE, AgentState.LISTENING):
                await self._bus.emit(EV_LEVEL, rms=min(1.0, rms * 8))

            if self._listening:
                utterance = self.vad.feed(chunk)
                if utterance is not None:
                    self._listening = False
                    if utterance.size == 0:
                        log.info("woke but heard nothing")
                        await self._state.transition(AgentState.IDLE)
                    else:
                        asyncio.create_task(self._handle(utterance))
                continue

            # Idle: only the wake word runs. Nothing is buffered or stored.
            if self._state.state is AgentState.IDLE and time.monotonic() >= self._muted_until:
                pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                if self.wake.feed(pcm16):
                    await self.begin_listening()

    async def begin_listening(self) -> None:
        """Wake word fired, or you hit the hotkey."""
        if self._listening:
            return
        # Talking over JARVIS cuts it off — same as interrupting a person.
        if self.tts.is_speaking:
            self.tts.stop()
        self.vad.reset()
        self.wake.reset()
        self._listening = True
        await self._state.transition(AgentState.LISTENING)
        log.info("listening")

    async def _handle(self, utterance: np.ndarray) -> None:
        await self._state.transition(AgentState.THINKING)
        started = time.monotonic()

        transcript = await asyncio.to_thread(self.stt.transcribe, utterance)
        if not transcript:
            await self._state.transition(AgentState.IDLE)
            return

        log.info("heard: %s", transcript)
        await self._bus.emit(EV_TRANSCRIPT, text=transcript, final=True)

        # Answering a question JARVIS asked, not starting a new one.
        if self._awaiting_answer is not None and not self._awaiting_answer.done():
            self._awaiting_answer.set_result(transcript)
            return

        if self._responder is None:
            await self._state.transition(AgentState.IDLE)
            return

        try:
            reply = await self._responder(transcript)
        except Exception:
            log.exception("responder failed")
            reply = "Something went wrong on my end — check the log."

        log.info("round trip %.0f ms", (time.monotonic() - started) * 1000)
        await self.say(reply)

    async def ask_yes_no(self, question: str, timeout_s: float = 30.0) -> bool:
        """Speak a question and wait for a spoken yes or no.

        This is what makes first-use permission prompts answerable out loud
        instead of forcing you to look at the tab and click.
        """
        await self.say(question)

        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._awaiting_answer = future
        await self.begin_listening()

        try:
            answer = await asyncio.wait_for(future, timeout=timeout_s)
        except asyncio.TimeoutError:
            log.info("no spoken answer — treating as no")
            return False
        finally:
            self._awaiting_answer = None
            await self._state.transition(AgentState.IDLE)

        return _is_yes(answer)

    async def say(self, text: str) -> None:
        """Speak, then return to idle. The only path from text to your speakers."""
        if not text.strip():
            await self._state.transition(AgentState.IDLE)
            return
        await self._state.transition(AgentState.SPEAKING)
        try:
            await self.tts.speak(text)
        finally:
            # Brief deafness so the tail of its own speech can't re-trigger wake.
            self._muted_until = time.monotonic() + 0.4
            self.wake.reset()
            await self._bus.emit(EV_LEVEL, rms=0.0)
            await self._state.transition(AgentState.IDLE)
