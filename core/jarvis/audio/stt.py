"""Speech to text.

You originally specced nvidia/canary-qwen-2.5b. It's a NeMo model built for
CUDA and it transcribes in batch rather than streaming — on Apple Silicon it
either won't run or will add seconds of latency to every single thing you say,
which is the opposite of "yell across the room and get an answer".

Parakeet-TDT via MLX is the Apple Silicon equivalent: same family of accuracy,
runs on the Neural Engine, transcribes a 5-second utterance in well under a
second. mlx-whisper large-v3-turbo is the fallback if parakeet won't load.

If you ever move JARVIS to a machine with an NVIDIA GPU, add a CanaryEngine
here implementing the same two methods and switch stt_engine in config.toml.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger("jarvis.stt")

SAMPLE_RATE = 16_000


class STTEngine(Protocol):
    def load(self) -> None: ...
    def transcribe(self, audio: np.ndarray) -> str: ...


class ParakeetEngine:
    def __init__(self, model_id: str = "mlx-community/parakeet-tdt-0.6b-v2"):
        self.model_id = model_id
        self._model = None

    def load(self) -> None:
        from parakeet_mlx import from_pretrained

        self._model = from_pretrained(self.model_id)
        log.info("stt ready: parakeet (%s)", self.model_id)

    def transcribe(self, audio: np.ndarray) -> str:
        import soundfile as sf

        # parakeet-mlx reads from a path; a few seconds of 16 kHz mono is a
        # trivially small temp file and this keeps us off its private API.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            path = Path(fh.name)
        try:
            sf.write(path, audio, SAMPLE_RATE)
            result = self._model.transcribe(str(path))
            return (result.text or "").strip()
        finally:
            path.unlink(missing_ok=True)


class WhisperEngine:
    def __init__(self, model_id: str = "mlx-community/whisper-large-v3-turbo"):
        self.model_id = model_id

    def load(self) -> None:
        import mlx_whisper  # noqa: F401  — import cost happens here, not mid-utterance

        log.info("stt ready: mlx-whisper (%s)", self.model_id)

    def transcribe(self, audio: np.ndarray) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio.astype(np.float32), path_or_hf_repo=self.model_id, language="en"
        )
        return (result.get("text") or "").strip()


class STT:
    """Wraps an engine and falls back to Whisper if the primary won't load."""

    def __init__(self, engine: str = "parakeet", parakeet_model: str = "",
                 whisper_model: str = ""):
        if engine == "parakeet":
            self._engine: STTEngine = ParakeetEngine(
                parakeet_model or "mlx-community/parakeet-tdt-0.6b-v2"
            )
            self._fallback: STTEngine | None = WhisperEngine(
                whisper_model or "mlx-community/whisper-large-v3-turbo"
            )
        else:
            self._engine = WhisperEngine(whisper_model or "mlx-community/whisper-large-v3-turbo")
            self._fallback = None

    def load(self) -> None:
        try:
            self._engine.load()
        except Exception as exc:
            if self._fallback is None:
                raise
            log.warning("primary stt failed to load (%s) — falling back to whisper", exc)
            self._engine = self._fallback
            self._fallback = None
            self._engine.load()

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < SAMPLE_RATE * 0.25:  # under 250 ms is a cough, not a sentence
            return ""
        started = time.monotonic()
        text = self._engine.transcribe(audio)
        log.info(
            "transcribed %.1fs of audio in %.0f ms",
            audio.size / SAMPLE_RATE, (time.monotonic() - started) * 1000,
        )
        return text
