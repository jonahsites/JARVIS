# Setup

macOS, Apple Silicon. Budget 20–30 minutes, most of it waiting on model
downloads (~4 GB total).

Run `python -m jarvis doctor` after every step — it tells you exactly what's
still missing rather than making you guess.

---

## 1. System dependencies

```bash
brew install ollama portaudio python@3.11
```

`portaudio` is what `sounddevice` binds to for the microphone. Without it, the
pip install of `sounddevice` succeeds but importing it fails at runtime.

## 2. The local model

```bash
ollama serve            # leave running, or `brew services start ollama`
ollama pull qwen3:8b    # ~5 GB
```

## 3. Python

```bash
cd core
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

First `python -m jarvis run` downloads the speech models (Parakeet ~600 MB,
Kokoro ~330 MB, the wake word ~2 MB). That happens once.

## 4. Credentials

```bash
cp .env.example .env
```

### Notion (needed for assignments and notes — Phase 2)

1. <https://notion.so/my-integrations> → **New integration** → Internal
2. Copy the secret into `NOTION_TOKEN`
3. Open your **Dashboard** page in Notion → `•••` → **Connections** → add it

Sharing Dashboard is enough — every database under it inherits access. That
covers Assignments, Class Notes, Courses, Units, Bookmarks, Tasks, Planner,
Goals, Drive and Days Off.

### OpenRouter (needed for the hard-question path)

<https://openrouter.ai/keys> → paste into `OPENROUTER_API_KEY`.

Without it JARVIS still runs, just entirely on qwen3:8b. You'll notice it on
anything requiring multi-step reasoning.

## 5. macOS permissions

macOS will prompt for most of these on first use. Grant them to **your
terminal** (Terminal, iTerm, whatever you launch from) — not to Python.

| Permission | Needed for | Where |
|---|---|---|
| Microphone | everything | Privacy & Security → Microphone |
| Accessibility | the ⌥Space hotkey | Privacy & Security → Accessibility |
| Automation → Chrome | tabs | prompts on first use |
| Automation → System Events | app control | prompts on first use |
| Full Disk Access | reading Messages (Phase 3) | Privacy & Security → Full Disk Access |

If you granted Accessibility and the hotkey still doesn't fire, quit the
terminal completely (⌘Q, not just closing the window) and reopen it. macOS
caches the permission per-process.

## 6. Check

```bash
python -m jarvis doctor
```

Fix every red row. Yellow rows are features that will simply be off.

## 7. Run

```bash
python -m jarvis run
```

Then in a second terminal:

```bash
cd ui
npm install
npm run dev
```

Open <http://localhost:5173> in Chrome and leave the tab open.

Say **"hey Jarvis"**, wait for the glob to go green, then talk.

---

## The one thing to check on day one

Ask it **"what day is it"** and confirm it says the right block letter.

Your school runs an A/B rotation, and Notion's API won't expose the code inside
your `Class today?` formula — so I couldn't read your actual rule. I implemented
the standard one: A and B alternate across school days, and days in your Days
Off database don't advance the rotation.

If the letter is wrong, edit **one line** in `config.toml`:

```toml
[schedule]
anchor_date = 2026-09-08      # any date you know for certain
anchor_day_type = "A"         # what that date actually was
```

Everything else re-derives from it, forwards and backwards.

---

## Tuning

All in `config.toml`, restart after editing.

**It doesn't wake reliably** → lower `wake_threshold` to `0.4`.
**It wakes when you didn't say it** → raise to `0.6`.

**It cuts you off mid-sentence** → raise `silence_ms` to `900`.
**It waits too long after you stop** → lower to `500`.

**Try a different voice** → `python -m jarvis say "testing one two three"` after
changing `tts_voice`. `bm_george` is British and sounds most like the JARVIS
you're naming this after.

**Replies are too chatty** → set `style = "terse"`.

**Transcription is wrong on names** → set `stt_engine = "whisper"`. Slower but
better on proper nouns.

---

## Troubleshooting

**`OSError: PortAudio library not found`** — `brew install portaudio`, then
`pip install --force-reinstall sounddevice`.

**Wake word never fires** — check the mic is the one you think:
```bash
python -c "import sounddevice as sd; print(sd.query_devices(kind='input'))"
```

**It hears its own voice and wakes itself** — the daemon already mutes wake
detection for 400ms after speaking. If it still happens with speakers turned
up, raise `wake_threshold`.

**`ollama: connection refused`** — `ollama serve` isn't running.

**Glob is a black circle** — a shader failed. Open Chrome DevTools console; look
for a GLSL error. `cd ui && npm run build && node state-test.mjs` reproduces it
outside the browser.

**UI says "daemon not running"** — the daemon isn't up, or something else is on
port 8765. `lsof -i :8765`, and change `JARVIS_BUS_PORT` in `.env` if needed.

**Everything is slow** — check whether it's escalating to OpenRouter on
everything. `python -m jarvis run -v` logs `routing to local` or
`routing to cloud` for each request.

---

## Where things are stored

Everything JARVIS remembers is in `var/`, which is gitignored:

- `var/jarvis.db` — capability grants, observed tabs and apps, facts, history
- `var/jarvis.log` — everything it thought, rotated at 5 MB

Delete `var/jarvis.db` to reset it completely, including permissions.
