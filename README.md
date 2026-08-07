# JARVIS

A local voice assistant for your Mac. You say "hey Jarvis" from across the room,
it answers out loud, and it can actually operate the machine — open apps and
tabs, read and send messages, read and write your Notion.

Everything that can run locally does. Your voice is transcribed on-device, the
reply is generated on-device by default, and the speech is synthesised
on-device. Nothing leaves the machine unless a request is complex enough to be
escalated to OpenRouter.

**Status: Phase 1 is built.** The voice loop and the UI work end to end. Notion,
Messages and LectureSynth have their interfaces defined but aren't wired yet —
see [Roadmap](#roadmap).

---

## How it works

```
  mic ──▶ wake word ──▶ VAD ──▶ Parakeet ──▶ router ──┬─▶ qwen3:8b (local)
       always on      end of     speech to            └─▶ OpenRouter (hard stuff)
       "hey jarvis"   turn       text                       │
                                                            ▼
  speakers ◀── Kokoro ◀── strip reasoning ◀── tool loop ◀────┘
                                                    │
   browser tab ◀── WebSocket ◀── state ─────────────┘
   (the glob)                    idle/listening/thinking/speaking
```

All audio lives in the Python daemon. The browser tab is display only, so the
loop keeps working whether or not it's open — which is the point of something
you shout at.

### The reasoning/speech wall

You never hear it think. Reasoning has exactly one destination — the log file —
and there is no code path from a log record to the speech engine. Concretely:

- Qwen3 is asked not to emit `<think>` blocks, and `strip_thinking()` removes
  them anyway if it does.
- Tool calls and their results never leave the agent loop.
- Only `VoicePipeline.say()` reaches Kokoro, and it's called with the final
  answer and nothing else.

### Permissions

The first time JARVIS does any *kind* of thing, it asks once, out loud. After
you say yes it never asks about that kind of thing again — including for
different arguments.

| You say | It asks? |
|---|---|
| "open YouTube" (first tab ever) | **Yes** — "okay if I start opening browser tabs?" |
| "open Instagram" | No — already approved |
| "text Mom I'm late" (first text ever) | **Yes** — "okay if I start texting people?" |
| "text Dad I'm late" | No — already approved |
| "add an essay to Calc" (first assignment) | **Yes** |
| "add a lab to APES" | No — already approved |

Grants persist across restarts. Revoke by voice at any time: *"stop letting
yourself send messages."*

Once `messages.send` is approved, individual texts are spoken before they go out
with a 3-second window to say stop — not a yes/no gate, just a chance to catch
a misheard name.

---

## Quick start

Full detail in **[SETUP.md](SETUP.md)**. The short version:

```bash
# 1. dependencies
brew install ollama portaudio
ollama serve &
ollama pull qwen3:8b

# 2. python
cd core && python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. credentials
cp ../.env.example ../.env    # add NOTION_TOKEN and OPENROUTER_API_KEY

# 4. check everything before you run it
python -m jarvis doctor

# 5. go
python -m jarvis run
```

Then in another terminal:

```bash
cd ui && npm install && npm run dev
```

Open <http://localhost:5173> in a Chrome tab and leave it there.

---

## The glob

Your Three.js component, rebuilt. It reflects state through colour:

| State | Colour | |
|---|---|---|
| idle | deep blue, slow drift | waiting for the wake word |
| listening | green, deforms with your voice | capturing what you say |
| thinking | amber, fast churn | routing, calling tools, generating |
| speaking | full multicolour | Kokoro is playing |

Transitions are tweened, not switched — idle→listening is the slowest at 0.85s.

Three changes from the code you gave me, all of which were load-bearing:

1. **Shaders are inlined** instead of read with `document.getElementById()`. The
   original lookup ran in `useEffect` against `<script>` tags in the same
   component's return, so on first mount they didn't exist yet.
2. **The scene is built once.** Every prop was in the `useEffect` dependency
   array, which tore down and rebuilt the renderer, geometry and shader program
   on every colour change — several times a second while talking.
3. **The `gsap` randomiser is gone.** It tweened `RGBr/g/b/n/m` toward
   `Math.random()`, which are exactly the uniforms carrying state colour, so
   blue-idle would have drifted to something arbitrary within a second.

Two uniforms were added: `tint`/`tintMix` (the noise colours are emergent, so
there was no way to say "be blue now"), and `gain` (the raw noise spends much of
its range negative and clamps to black). At `tintMix = 0, gain = 1` you get your
original output exactly — which is what the speaking state uses.

Preview all four states without a microphone:

```bash
cd ui && npm run build && npm i --no-save playwright ws && node state-test.mjs
```

---

## Layout

```
core/jarvis/
  daemon.py          wires everything together
  bus.py             WebSocket to the UI; the speech/thinking boundary
  state.py           idle / listening / thinking / speaking
  audio/             wake word, VAD, Parakeet, Kokoro, the loop, hotkey
  llm/               ollama, openrouter, the router, prompts
  agent/             tool loop, tool registry, the capability ledger
  skills/            macOS control, A/B schedule, tool definitions
  memory/            local SQLite — tabs, apps, facts, history
  doctor.py          checks every dependency and permission

ui/src/
  components/AbstractBall.tsx   the glob
  lib/shaders.ts                GLSL
  lib/globStates.ts             colour per state
  hooks/useJarvis.ts            WebSocket to the daemon
```

## Commands

```bash
python -m jarvis run          # start everything
python -m jarvis doctor       # check deps, permissions, credentials
python -m jarvis say "hello"  # audition a voice
python -m jarvis ask "..."    # one request through the agent, no microphone
```

## Roadmap

**Phase 1 — done.** Voice loop, glob UI, capability ledger, memory, macOS app
and Chrome control, A/B schedule logic, doctor.

**Phase 2 — Notion.** Assignments (read, add, update), notes search across page
bodies and the Drive database, live class schedule from Courses, Days Off
feeding the A/B rotation.

**Phase 3 — Messages and LectureSynth.** iMessage read/send via `chat.db` and
AppleScript. LectureSynth capture → auto-added assignments.

**Phase 4 — Proactivity and onboarding.** The autonomous daemon that speaks up
unprompted, and the guided first-run that learns your projects and habits.

## Known limits

- **macOS only.** App control, Messages and the hotkey are all AppleScript and
  macOS APIs.
- **Apple Silicon for the fast path.** The MLX speech models need it. On Intel,
  set `stt_engine = "whisper"` and expect it to be slower.
- **`canary-qwen-2.5b` isn't used.** It's a NeMo/CUDA batch model — it won't run
  on a Mac, and even on NVIDIA its batch design fights the latency this needs.
  Parakeet-TDT is the Apple Silicon equivalent. If you ever move to an NVIDIA
  box, add a `CanaryEngine` to `audio/stt.py` implementing the same two methods.
- **I could not run any of this end to end.** It was built in a Linux container
  with no microphone, no Ollama and no macOS. The pure logic is tested (see
  below); the hardware paths are not. `jarvis doctor` exists for exactly this
  reason — run it first.

## What's verified

Tested here: the A/B rotation (alternation, weekends, days off, backwards from
the anchor), the capability ledger (one prompt per capability across many
arguments, persistence across restart, revoke and re-ask), config parsing, the
UI typecheck and build, GLSL compilation, and all four glob states rendered
through a fake daemon.

Not tested here: microphone capture, wake word, Parakeet, Kokoro, Ollama, the
hotkey, and AppleScript. Those need your machine.
