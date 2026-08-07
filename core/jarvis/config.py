"""Configuration. Secrets come from .env; behaviour comes from config.toml.

Every value marked TODO(jonah) is a default I chose because the question was
left unanswered during the spec conversation. They are all safe to change and
none of them require touching code — edit config.toml and restart.
"""

from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
VAR_DIR = REPO_ROOT / "var"
MODELS_DIR = REPO_ROOT / "models"
CONFIG_PATH = REPO_ROOT / "config.toml"


class Secrets(BaseSettings):
    """Everything from .env. Missing values degrade gracefully — see doctor."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    notion_token: str = ""
    openrouter_api_key: str = ""
    # Free auto-router that filters for tool-calling support. Rate limited;
    # JARVIS falls back to the local model when it runs out.
    openrouter_model: str = "openrouter/free"

    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"

    lecturesynth_base_url: str = "https://shorts-employee-umbilical.ngrok-free.dev"
    lecturesynth_token: str = ""

    jarvis_bus_port: int = 8765
    jarvis_extension_port: int = 8766


class VoiceConfig(BaseModel):
    # openWakeWord ships a pretrained "hey jarvis" model, which is why this is
    # the default rather than a custom-trained one.
    wake_model: str = "hey_jarvis_v0.1"
    wake_threshold: float = 0.5

    # Registered by the daemon itself (pynput), so it works no matter what the
    # UI tab is doing — or whether it's even open. Needs Accessibility
    # permission on macOS; `jarvis doctor` tells you if it's missing.
    hotkey: str = "<alt>+<space>"

    # Qwen3-ASR on MLX — the open-source Qwen ASR model that actually runs on
    # Apple Silicon, standing in for the canary-qwen you specced. Falls through
    # to parakeet then whisper if it can't load.
    stt_engine: Literal["qwen", "parakeet", "whisper"] = "qwen"
    qwen_model: str = "Qwen/Qwen3-ASR-0.6B"  # or Qwen/Qwen3-ASR-1.7B for 1.99% WER
    stt_model: str = "mlx-community/parakeet-tdt-0.6b-v2"
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"

    # Male voice, per your call. am_michael is the best-quality American male in
    # Kokoro. The on-theme alternative is bm_george — British, and it sounds far
    # more like the JARVIS you're naming this after. Others: am_puck, am_adam,
    # am_eric, am_liam, bm_lewis, bm_fable. Swap the string and restart.
    tts_voice: str = "am_michael"
    tts_speed: float = 1.05

    # How long a pause ends your turn. Lower = snappier, more likely to cut you
    # off mid-sentence.
    silence_ms: int = 700
    max_utterance_s: float = 30.0

    # Talk over JARVIS and it stops. The wake word is then only for *starting*
    # a conversation, never for interrupting or answering.
    #
    # Without acoustic echo cancellation the mic hears the speakers, so this
    # calibrates: the first barge_calibrate_ms of playback establishes an echo
    # floor, and only sustained sound clearly above it counts as you talking.
    # Raise the multiplier if it cuts itself off; lower it if it ignores you.
    barge_in: bool = True
    barge_in_multiplier: float = 2.8
    barge_in_ms: int = 320
    barge_calibrate_ms: int = 450


class LLMConfig(BaseModel):
    # "Local first, escalate on complexity" — your choice.
    escalate_on_tool_depth: int = 2
    escalate_on_token_estimate: int = 1500
    escalate_keywords: list[str] = Field(
        default_factory=lambda: [
            "compare", "summarize", "explain why", "plan", "figure out",
            "draft", "write me", "analyze", "what should i",
        ]
    )
    local_timeout_s: float = 20.0
    cloud_timeout_s: float = 60.0

    # Keeps the model resident between turns. Reloading several GB costs
    # seconds and Ollama does it silently once the model goes idle.
    ollama_keep_alive: str = "30m"
    # The default is small enough that the tool schemas plus the system prompt
    # can overflow it, which forces context re-processing on every call.
    ollama_num_ctx: int = 8192


class PersonaConfig(BaseModel):
    name: str = "Jarvis"
    user_name: str = "Jonah"
    # Conversational, per your call. Note this changes *tone*, not length — it
    # still won't narrate its reasoning at you, it just talks like a person
    # instead of a status bar.
    style: Literal["terse", "warm"] = "warm"
    ack_phrases: list[str] = Field(
        default_factory=lambda: [
            "Yeah, on it.", "Sure thing.", "Got it, one sec.", "Yep, doing that now.",
        ]
    )


class ScheduleConfig(BaseModel):
    """Your school runs an A/B block rotation.

    BLOCKER 3 — I could see the Courses rows and the Days Off database, but
    Notion's API will not expose the code inside your `Class today?` formula,
    so I cannot derive the rule. I implemented the most likely one: A and B
    alternate across school days, skipping anything in Days Off. Set the anchor
    below to any date you know for certain and it self-corrects from there.
    """

    anchor_date: date = date(2026, 9, 8)  # TODO(jonah): first day of school
    anchor_day_type: Literal["A", "B"] = "A"  # TODO(jonah): was it an A or B day?
    school_weekdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    timezone: str = "America/New_York"


class MemoryConfig(BaseModel):
    # "Tabs, apps, and window titles" — your choice. Page *content* is never
    # read or stored; flipping this to True would change that, so it is opt-in.
    observe_tabs: bool = True
    observe_apps: bool = True
    observe_page_content: bool = False
    poll_seconds: int = 20
    retention_days: int = 90


class ProactiveConfig(BaseModel):
    """You chose fully autonomous — it speaks whenever it judges useful.

    The rate limits below exist so "autonomous" doesn't become "constantly
    interrupting". Raise max_per_hour if it feels too quiet.
    """

    enabled: bool = True
    max_per_hour: int = 4
    quiet_hours: list[str] = Field(default_factory=lambda: ["22:30", "07:00"])
    # Don't interrupt while these are frontmost and you've been in them a while.
    focus_apps: list[str] = Field(default_factory=lambda: ["zoom.us", "FaceTime"])
    tick_seconds: int = 60


class MessagesConfig(BaseModel):
    # "Speak it, 3-second cancel window" — your choice.
    cancel_window_s: float = 3.0
    blocklist: list[str] = Field(default_factory=list)  # TODO(jonah): never-text names


class JarvisConfig(BaseModel):
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    messages: MessagesConfig = Field(default_factory=MessagesConfig)

    @classmethod
    def load(cls) -> "JarvisConfig":
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("rb") as fh:
                return cls.model_validate(tomllib.load(fh))
        return cls()


VAR_DIR.mkdir(parents=True, exist_ok=True)

secrets = Secrets()
config = JarvisConfig.load()
