"""Wake word detection.

openWakeWord ships a pretrained "hey jarvis" model, which is why that's the
default — no training step, and it runs in a few hundred microseconds per frame
on Apple Silicon. This is the only thing listening while JARVIS is idle; nothing
is transcribed or stored until it fires.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("jarvis.wake")

FRAME_SAMPLES = 1280  # openWakeWord expects 80 ms at 16 kHz


class WakeWord:
    def __init__(self, model_name: str = "hey_jarvis_v0.1", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._buffer = np.zeros(0, dtype=np.int16)
        # After a hit, ignore the next few frames so one "Jarvis" doesn't fire
        # twice as the word tails off.
        self._cooldown = 0

    def load(self) -> None:
        from openwakeword.model import Model
        from openwakeword.utils import download_models

        download_models([self.model_name])
        self._model = Model(wakeword_models=[self.model_name], inference_framework="onnx")
        log.info("wake word ready: %s (threshold %.2f)", self.model_name, self.threshold)

    def feed(self, pcm16: np.ndarray) -> bool:
        """Push mic audio. True exactly once per detected wake word."""
        if self._model is None:
            return False

        self._buffer = np.concatenate([self._buffer, pcm16])
        fired = False

        while len(self._buffer) >= FRAME_SAMPLES:
            frame, self._buffer = self._buffer[:FRAME_SAMPLES], self._buffer[FRAME_SAMPLES:]

            if self._cooldown > 0:
                self._cooldown -= 1
                continue

            scores = self._model.predict(frame)
            score = scores.get(self.model_name, 0.0)
            if score >= self.threshold:
                log.info("wake word detected (%.2f)", score)
                self._cooldown = 25  # ~2 s
                fired = True

        return fired

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.int16)
        self._cooldown = 0
        if self._model is not None:
            self._model.reset()
