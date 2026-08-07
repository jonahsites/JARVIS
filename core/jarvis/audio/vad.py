"""Voice activity detection — decides when you've finished talking.

Silero over a plain energy threshold because rooms are noisy and an energy gate
would either cut you off mid-sentence or hang for seconds after you stop. The
tradeoff you can feel is `silence_ms`: lower is snappier but more likely to
treat a thinking pause as the end of your turn.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("jarvis.vad")

CHUNK_SAMPLES = 512  # Silero requires exactly 512 samples at 16 kHz
SAMPLE_RATE = 16_000


class UtteranceDetector:
    def __init__(self, silence_ms: int = 700, max_utterance_s: float = 30.0,
                 speech_threshold: float = 0.5, lead_silence_s: float = 2.5):
        self.silence_ms = silence_ms
        self.max_samples = int(max_utterance_s * SAMPLE_RATE)
        self.speech_threshold = speech_threshold
        # How long to wait for you to start talking before giving up. Short
        # after the wake word, longer when it's holding a conversation open.
        self.lead_silence_s = lead_silence_s

        self._model = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._captured: list[np.ndarray] = []
        self._captured_len = 0
        self._silence_samples = 0
        self._heard_speech = False

    def load(self) -> None:
        from silero_vad import load_silero_vad

        self._model = load_silero_vad(onnx=True)
        log.info("vad ready (silence %d ms)", self.silence_ms)

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._captured = []
        self._captured_len = 0
        self._silence_samples = 0
        self._heard_speech = False
        if self._model is not None:
            self._model.reset_states()

    def feed(self, pcm_f32: np.ndarray) -> np.ndarray | None:
        """Push audio while listening.

        Returns None while you're still talking, or the full utterance once
        you've stopped. Returns an empty array if the window closed without any
        speech at all, so the caller can go back to idle without transcribing.
        """
        if self._model is None:
            return None

        import torch

        self._pending = np.concatenate([self._pending, pcm_f32])
        silence_limit = int(self.silence_ms / 1000 * SAMPLE_RATE)

        while len(self._pending) >= CHUNK_SAMPLES:
            chunk, self._pending = self._pending[:CHUNK_SAMPLES], self._pending[CHUNK_SAMPLES:]

            prob = float(self._model(torch.from_numpy(chunk), SAMPLE_RATE).item())
            is_speech = prob >= self.speech_threshold

            # Always capture — including the moment *before* speech is confirmed,
            # otherwise the first consonant gets clipped.
            self._captured.append(chunk)
            self._captured_len += CHUNK_SAMPLES

            if is_speech:
                self._heard_speech = True
                self._silence_samples = 0
            else:
                self._silence_samples += CHUNK_SAMPLES

            if self._heard_speech and self._silence_samples >= silence_limit:
                return self._finish()

            # You've been talking a long time — cut it here rather than
            # buffering forever.
            if self._captured_len >= self.max_samples:
                log.info("utterance hit max length")
                return self._finish()

            # Listening but you haven't started. Dead air ends the window.
            if (not self._heard_speech
                    and self._silence_samples >= SAMPLE_RATE * self.lead_silence_s):
                self.reset()
                return np.zeros(0, dtype=np.float32)

        return None

    def _finish(self) -> np.ndarray:
        audio = np.concatenate(self._captured) if self._captured else np.zeros(0, np.float32)
        self.reset()
        return audio
