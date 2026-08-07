"""Text to speech via Kokoro, with barge-in.

Kokoro was a good call — 82M params, genuinely natural, and fast enough on
Apple Silicon to start speaking before it has finished generating.

Two things matter here beyond making sound:

1. Sentences are synthesised and queued one at a time, so the first words play
   while later ones are still generating. Waiting for a whole paragraph would
   add a dead beat to every reply.
2. `stop()` cuts playback immediately. That's what makes the 3-second cancel
   window on messages work, and it's what lets you talk over JARVIS.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Callable

import numpy as np

log = logging.getLogger("jarvis.tts")

KOKORO_RATE = 24_000
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class TTS:
    def __init__(self, voice: str = "af_bella", speed: float = 1.0,
                 on_level: Callable[[float], None] | None = None):
        self.voice = voice
        self.speed = speed
        self._on_level = on_level
        self._pipeline = None
        self._stop = threading.Event()
        self._playing = threading.Event()

    def load(self) -> None:
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code="a")  # 'a' = American English
        log.info("tts ready: kokoro (%s @ %.2fx)", self.voice, self.speed)

    @property
    def is_speaking(self) -> bool:
        return self._playing.is_set()

    def stop(self) -> None:
        """Cut playback now. Safe to call when nothing is playing."""
        self._stop.set()

    async def speak(self, text: str) -> bool:
        """Speak text. Returns False if it was interrupted before finishing."""
        text = text.strip()
        if not text or self._pipeline is None:
            return True

        self._stop.clear()
        self._playing.set()
        try:
            return await asyncio.to_thread(self._speak_blocking, text)
        finally:
            self._playing.clear()

    def _speak_blocking(self, text: str) -> bool:
        import sounddevice as sd

        sentences = [s for s in _SENTENCE.split(text) if s.strip()]
        stream = sd.OutputStream(samplerate=KOKORO_RATE, channels=1, dtype="float32")
        stream.start()
        try:
            for sentence in sentences:
                if self._stop.is_set():
                    return False
                for _, _, audio in self._pipeline(sentence, voice=self.voice,
                                                  speed=self.speed):
                    if self._stop.is_set():
                        return False
                    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
                    # Write in blocks rather than one call, so a stop() lands
                    # within ~50 ms instead of after the whole sentence.
                    block = 1200
                    for i in range(0, len(samples), block):
                        if self._stop.is_set():
                            return False
                        chunk = samples[i:i + block]
                        if self._on_level is not None:
                            self._on_level(float(np.sqrt(np.mean(chunk ** 2))))
                        stream.write(chunk)
            return True
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            if self._on_level is not None:
                self._on_level(0.0)
