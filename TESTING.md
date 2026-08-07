# Testing JARVIS the first time

Everything here was written in a Linux container with no microphone, no macOS,
no Ollama and no Notion token. The pure logic is tested; every integration with
your actual hardware and with third-party APIs is written from documentation
and has never been executed.

So assume the first run breaks somewhere. The point of this page is that when
it does, you know exactly which piece — not "the voice assistant doesn't work".

Work down the list. **Don't skip ahead** — later stages depend on earlier ones,
and a failure at stage 3 caused by stage 1 is miserable to debug.

---

## 0. Install and check

```bash
./scripts/setup.sh
cd core && source .venv/bin/activate
python -m jarvis doctor
```

`doctor` answers *"is it installed and permitted"*. Fix every red row before
going further. Yellow rows are features that stay off — fine for now.

## 1. Non-interactive sweep

```bash
python -m jarvis test all
```

Runs everything that doesn't need you at the microphone: Kokoro, the LLM
router, AppleScript, Notion, and reading Messages. Gives you a pass/fail summary
so you know the shape of the damage before touching audio.

Then work through whatever failed, one at a time.

## 2. Audio

```bash
python -m jarvis test audio
```

Records three seconds and plays it back. Proves the mic and speakers work
independently of any model.

**Recorded silence** → wrong input device or a muted mic:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## 3. Speech

```bash
python -m jarvis test tts
python -m jarvis test stt
```

`test stt` also prints how far ahead of realtime it ran. Below about 3x on a
5-second clip means something is falling back to CPU — check which engine
actually loaded in the output.

Don't like the voice? Change `tts_voice` in `config.toml` and run
`python -m jarvis say "testing one two three"`. `bm_george` is the British one.

## 4. Wake word

```bash
python -m jarvis test wake
```

Fifteen seconds to say "hey Jarvis" a few times. It prints each detection.

**Never fires** → lower `wake_threshold` to `0.4`.
**Fires constantly** → raise it to `0.6`.

## 5. Brain

```bash
python -m jarvis test llm
```

Asks two questions that can only be answered by calling a tool — this is where
8B models most often fall down. It also checks no reasoning leaked into the
spoken reply.

It also reports how long each answer took, and warns if it's too slow.

**If it says 8+ seconds**, `qwen3:8b` is too big for your machine — on a
MacBook Air especially, an 8B model spends most of its time swapping. Switch to
the 4B:

```bash
ollama pull qwen3:4b
echo 'OLLAMA_MODEL=qwen3:4b' >> ../.env
python -m jarvis test llm
```

Roughly twice as fast, and the router escalates anything genuinely hard to
OpenRouter regardless — so the local model mostly needs to be quick, not clever.

You can talk to it without a microphone at all:

```bash
python -m jarvis ask "what do I have due this week"
python -m jarvis ask "what day is it"
```

## 6. Notion

```bash
python -m jarvis test notion
```

Authenticates, lists your courses, reads assignments, loads Days Off, then
crawls your notes and does a test search.

**"no courses returned"** → the integration can't see your databases. Open the
Dashboard page in Notion → `•••` → **Connections** → add your integration.
Children inherit, so that one share covers everything.

## 7. Messages

```bash
python -m jarvis test messages
```

Read-only — this never sends anything. It shows your last few messages and
reports how many bodies it managed to decode.

**"every message body came back empty"** → the `attributedBody` decoder doesn't
match your macOS version. Send me the output; it's a contained fix.

**"can't read chat.db"** → Full Disk Access for your terminal, then **fully
quit it** (⌘Q, not just the window) and reopen. macOS caches this per-process.

## 8. The whole thing

Terminal 1:

```bash
cd core
source .venv/bin/activate
python -m jarvis run -v
```

Terminal 2:

```bash
cd ui
npm run dev
```

Open <http://localhost:5173> in Chrome.

`-v` matters on the first real run — it logs every state transition, which
route each request took, and every tool call.

**On first launch it starts the onboarding conversation.** Say "skip" to
anything you don't want to answer. Re-run later with `python -m jarvis onboard`.

### Watch for, in order

1. Glob goes from grey to **blue** — the UI connected to the daemon
2. Say "hey Jarvis" → **green** within a beat
3. Say something → **amber** while it thinks
4. It answers → **multicolour** while speaking, then back to blue

If the glob never leaves grey, the daemon isn't running or something else is on
port 8765 (`lsof -i :8765`).

---

## Things to actually try

Ordered so each one exercises something new.

| Say | What it proves |
|---|---|
| "what time is it" | the loop works end to end |
| "is today an A day" | schedule logic — **check the letter is right** |
| "what do I have due" | Notion reads, priority ranking |
| "open YouTube" | first capability prompt — it should ask **once** |
| "open Instagram" | it should **not** ask again |
| "find my notes about limits" | full-text search over note bodies |
| "add an essay for calc due Friday" | course matching, date parsing, writing |
| "any messages" | chat.db read |
| "text Mom I'm running late" | cancel window — **let it send** |
| "text Mom testing" then "stop" | cancel window — **nothing should send** |
| "stop letting yourself send messages" | revoking a capability |

---

## The one thing to check on day one

Ask **"is today an A day or a B day"** and confirm it's right.

Notion won't expose the code inside your `Class today?` formula, so I
implemented the standard rule — A and B alternate across school days, and days
in your Days Off database don't advance the rotation. If the letter is wrong,
fix two lines in `config.toml`:

```toml
[schedule]
anchor_date = 2026-09-08      # any date you're certain about
anchor_day_type = "A"         # what that date actually was
```

Everything re-derives from it, forwards and backwards.

---

## Known gotcha

**`RuntimeError: There is no Stream(gpu, 1) in current thread`** — MLX binds
its Metal stream to whichever thread created it, so the model has to be loaded
and used on the same one. `STT` and `TTS` each own a single-worker thread pool
for exactly this; if you add another GPU-backed model, give it the same
treatment rather than reaching for `asyncio.to_thread`.

Worth knowing because it hides well: sequential `asyncio.to_thread` calls reuse
one pooled thread, so this works fine in isolated tests and only breaks in the
daemon, where models load concurrently and land on different threads.

## When something breaks

`var/jarvis.log` has everything it thought, including every tool call and every
decision to stay quiet. Run with `-v` for full detail.

Useful to send me:

```bash
python -m jarvis doctor
python -m jarvis test all
tail -100 var/jarvis.log
```

## Resetting

```bash
rm var/jarvis.db
rm var/notes.db
```
