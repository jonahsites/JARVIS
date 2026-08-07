"""Speech to text.

You specced nvidia/canary-qwen-2.5b. It's a NeMo model built for CUDA and it
transcribes in batch rather than streaming — on Apple Silicon it won't run at
all, which is why I went looking for a free, open-source replacement.

The good news is that what you actually wanted — a Qwen ASR model — exists for
Apple Silicon. Qwen3-ASR has been reimplemented on Apple's MLX framework
(mlx-qwen3-asr, Apache 2.0), and on the numbers it beats what I'd picked as a
substitute:

    engine          WER (LibriSpeech clean)   2.5s clip on M4 Pro
    Qwen3-ASR 0.6B  2.29%                     0.11s (8-bit) / 0.46s (fp16)
    Parakeet-TDT    ~2.5%                     ~0.3s
    Whisper turbo   ~3%                       ~0.5s

It also takes a numpy array directly, so unlike Parakeet there's no temp WAV
round-trip on every utterance.

So: Qwen3-ASR is the default, Parakeet is the first fallback, Whisper the
second. All three are free and open source. Switch with stt_engine in
config.toml — they share the same two-method interface, and so would a
CanaryEngine if you ever move to an NVIDIA box.
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


class QwenASREngine:
    """Qwen3-ASR on MLX. The closest thing to what you originally asked for
    that actually runs on this machine."""

    def __init__(self, model_id: str = "Qwen/Qwen3-ASR-0.6B"):
        self.model_id = model_id
        self._session = None

    def load(self) -> None:
        from mlx_qwen3_asr import Session

        # Session keeps weights resident; the module-level transcribe() helper
        # reloads the model per call, which would add ~2s to every utterance.
        self._session = Session(model=self.model_id)
        log.info("stt ready: qwen3-asr (%s)", self.model_id)

    def transcribe(self, audio: np.ndarray) -> str:
        result = self._session.transcribe(audio.astype(np.float32))
        return (getattr(result, "text", "") or "").strip()


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
    """Loads the configured engine, walking down the fallback chain if it fails.

    A missing optional package shouldn't stop JARVIS starting — it should just
    quietly cost you some accuracy, and say so in the log.
    """

    def __init__(self, engine: str = "qwen", qwen_model: str = "",
                 parakeet_model: str = "", whisper_model: str = ""):
        qwen = lambda: QwenASREngine(qwen_model or "Qwen/Qwen3-ASR-0.6B")  # noqa: E731
        parakeet = lambda: ParakeetEngine(  # noqa: E731
            parakeet_model or "mlx-community/parakeet-tdt-0.6b-v2"
        )
        whisper = lambda: WhisperEngine(  # noqa: E731
            whisper_model or "mlx-community/whisper-large-v3-turbo"
        )

        chains: dict[str, list] = {
            "qwen": [qwen, parakeet, whisper],
            "parakeet": [parakeet, qwen, whisper],
            "whisper": [whisper],
        }
        self._chain = chains.get(engine, chains["qwen"])
        self._engine: STTEngine | None = None

    def load(self) -> None:
        errors: list[str] = []
        for factory in self._chain:
            candidate = factory()
            try:
                candidate.load()
                self._engine = candidate
                return
            except Exception as exc:
                name = type(candidate).__name__
                errors.append(f"{name}: {exc}")
                log.warning("%s unavailable (%s) — trying next engine", name, exc)

        raise RuntimeError(
            "no speech recognition engine could load:\n  " + "\n  ".join(errors)
        )

    def transcribe(self, audio: np.ndarray) -> str:
        if self._engine is None:
            raise RuntimeError("STT.load() was never called")
        if audio.size < SAMPLE_RATE * 0.25:  # under 250 ms is a cough, not a sentence
            return ""
        started = time.monotonic()
        text = self._engine.transcribe(audio)
        log.info(
            "transcribed %.1fs of audio in %.0f ms",
            audio.size / SAMPLE_RATE, (time.monotonic() - started) * 1000,
        )
        return text
