"""`jarvis doctor` — checks every moving part and tells you how to fix it.

This exists because I built JARVIS in a Linux container with no microphone, no
Ollama and no macOS, so I could not run any of it end to end. Rather than
pretend otherwise, this verifies each piece on your machine and prints the
exact command or setting that fixes anything broken.
"""

from __future__ import annotations

import asyncio
import platform
import socket
import sys
from dataclasses import dataclass

import httpx
from rich.console import Console
from rich.table import Table

from .config import config, secrets

console = Console()

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _import_check(module: str, package: str, why: str) -> Check:
    try:
        __import__(module)
        return Check(module, OK, "installed")
    except Exception as exc:
        return Check(module, FAIL, f"{why} — {type(exc).__name__}",
                     f"pip install {package}")


def _port_free(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


async def run_checks() -> int:
    checks: list[Check] = []

    # ---- platform --------------------------------------------------------

    if platform.system() != "Darwin":
        checks.append(Check(
            "platform", FAIL, f"{platform.system()} — built for macOS",
            "App control, Messages and the hotkey are all macOS-only.",
        ))
    elif platform.machine() != "arm64":
        checks.append(Check(
            "platform", WARN, "Intel Mac — MLX speech models need Apple Silicon",
            "Set stt_engine = \"whisper\" in config.toml; expect slower transcription.",
        ))
    else:
        checks.append(Check("platform", OK, f"macOS on {platform.machine()}"))

    version = sys.version_info
    checks.append(
        Check("python", OK if version >= (3, 11) else FAIL,
              f"{version.major}.{version.minor}.{version.micro}",
              "" if version >= (3, 11) else "Python 3.11+ required")
    )

    # ---- packages --------------------------------------------------------

    for module, package, why in [
        ("sounddevice", "sounddevice", "microphone and speakers"),
        ("numpy", "numpy", "audio buffers"),
        ("openwakeword", "openwakeword", "wake word"),
        ("silero_vad", "silero-vad", "end-of-speech detection"),
        ("kokoro", "kokoro", "speech synthesis"),
        ("ollama", "ollama", "local model"),
        ("openai", "openai", "OpenRouter escalation"),
        ("pynput", "pynput", "global hotkey"),
    ]:
        checks.append(_import_check(module, package, why))

    if platform.machine() == "arm64":
        stt = _import_check("parakeet_mlx", "parakeet-mlx", "speech recognition")
        if stt.status == FAIL:
            stt.status = WARN
            stt.fix = "pip install parakeet-mlx  (falls back to mlx-whisper for now)"
        checks.append(stt)

    # ---- audio devices ---------------------------------------------------

    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        outputs = [d for d in devices if d["max_output_channels"] > 0]

        checks.append(
            Check("microphone", OK if inputs else FAIL,
                  sd.query_devices(kind="input")["name"] if inputs else "none found",
                  "" if inputs else "System Settings > Privacy & Security > Microphone")
        )
        checks.append(
            Check("speakers", OK if outputs else FAIL,
                  sd.query_devices(kind="output")["name"] if outputs else "none found")
        )
    except Exception as exc:
        checks.append(Check("audio devices", FAIL, str(exc),
                            "Grant your terminal Microphone permission, then restart it"))

    # ---- ollama ----------------------------------------------------------

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{secrets.ollama_host}/api/tags")
            names = [m["model"] for m in response.json().get("models", [])]

        if secrets.ollama_model in names:
            checks.append(Check("ollama", OK, f"{secrets.ollama_model} ready"))
        else:
            checks.append(Check(
                "ollama", FAIL,
                f"running, but {secrets.ollama_model} is not pulled",
                f"ollama pull {secrets.ollama_model}",
            ))
    except Exception:
        checks.append(Check("ollama", FAIL, f"not reachable at {secrets.ollama_host}",
                            "ollama serve"))

    # ---- credentials -----------------------------------------------------

    if secrets.openrouter_api_key:
        checks.append(Check("openrouter", OK, f"key set ({secrets.openrouter_model})"))
    else:
        checks.append(Check("openrouter", WARN, "no key — local model only",
                            "Add OPENROUTER_API_KEY to .env (openrouter.ai/keys)"))

    if secrets.notion_token:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.notion.com/v1/users/me",
                    headers={
                        "Authorization": f"Bearer {secrets.notion_token}",
                        "Notion-Version": "2022-06-28",
                    },
                )
            checks.append(
                Check("notion", OK if response.status_code == 200 else FAIL,
                      f"HTTP {response.status_code}",
                      "" if response.status_code == 200
                      else "Check the token, and share your Dashboard page with the integration")
            )
        except Exception as exc:
            checks.append(Check("notion", FAIL, str(exc)))
    else:
        checks.append(Check("notion", WARN, "no token — assignments and notes are off",
                            "notion.so/my-integrations, then add NOTION_TOKEN to .env"))

    # ---- macOS permissions ----------------------------------------------

    if platform.system() == "Darwin":
        try:
            from .skills.mac import osascript

            await osascript('tell application "System Events" to get name of first '
                            'application process whose frontmost is true')
            checks.append(Check("accessibility", OK, "granted"))
        except Exception:
            checks.append(Check(
                "accessibility", FAIL, "denied — hotkey and app control won't work",
                "System Settings > Privacy & Security > Accessibility > add your terminal",
            ))

        try:
            from .skills.mac import list_tabs

            tabs = await list_tabs()
            checks.append(Check("chrome control", OK, f"{len(tabs)} tabs visible"))
        except Exception:
            checks.append(Check(
                "chrome control", WARN, "no response from Chrome",
                "Open Chrome, then approve the Automation prompt on first use",
            ))

    # ---- ports -----------------------------------------------------------

    for label, port in [("bus port", secrets.jarvis_bus_port),
                        ("extension port", secrets.jarvis_extension_port)]:
        free = _port_free(port)
        checks.append(Check(label, OK if free else FAIL, f"{port} {'free' if free else 'in use'}",
                            "" if free else f"Something else is on {port} — change it in .env"))

    # ---- schedule anchor -------------------------------------------------

    checks.append(Check(
        "a/b anchor", WARN,
        f"{config.schedule.anchor_date} = {config.schedule.anchor_day_type} day (my guess)",
        "If the letter is ever wrong, fix anchor_date/anchor_day_type in config.toml",
    ))

    # ---- report ----------------------------------------------------------

    table = Table(title="JARVIS doctor", show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("")
    table.add_column("detail")
    table.add_column("fix", style="dim")

    marks = {OK: "[green]OK[/green]", WARN: "[yellow]--[/yellow]", FAIL: "[red]XX[/red]"}
    for check in checks:
        table.add_row(check.name, marks[check.status], check.detail, check.fix)

    console.print(table)

    failures = sum(1 for c in checks if c.status == FAIL)
    warnings = sum(1 for c in checks if c.status == WARN)

    if failures:
        console.print(f"\n[red]{failures} blocking[/red], {warnings} warnings. "
                      "Fix the red rows and run doctor again.")
        return 1
    console.print(f"\n[green]Ready.[/green] {warnings} warnings — "
                  "JARVIS will run, just with those features off.")
    return 0
