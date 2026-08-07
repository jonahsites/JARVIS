#!/usr/bin/env bash
# One-shot setup. Safe to re-run — every step is idempotent.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33m    %s\033[0m\n' "$1"; }

[[ "$(uname)" == "Darwin" ]] || { echo "macOS only."; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || warn "Not Apple Silicon — MLX speech models won't run. Set stt_engine=\"whisper\" in config.toml."

say "Homebrew packages"
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
for pkg in portaudio ollama python@3.11; do
  brew list "$pkg" >/dev/null 2>&1 && echo "    $pkg already installed" || brew install "$pkg"
done

say "Ollama"
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  warn "Not running. Starting it in the background..."
  brew services start ollama || (ollama serve >/dev/null 2>&1 &)
  sleep 3
fi
if curl -sf http://127.0.0.1:11434/api/tags 2>/dev/null | grep -q 'qwen3:8b'; then
  echo "    qwen3:8b already pulled"
else
  echo "    pulling qwen3:8b (~5 GB, this takes a while)"
  ollama pull qwen3:8b
fi

say "Python environment"
cd "$ROOT/core"
[[ -d .venv ]] || python3.11 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e .
echo "    installed into core/.venv"

say "UI"
cd "$ROOT/ui"
if command -v npm >/dev/null; then
  npm install --silent --no-audit --no-fund
  echo "    node modules installed"
else
  warn "npm not found — install Node, then run: cd ui && npm install"
fi

say "Credentials"
cd "$ROOT"
if [[ -f .env ]]; then
  echo "    .env already exists, leaving it alone"
else
  cp .env.example .env
  warn "Created .env — add NOTION_TOKEN and OPENROUTER_API_KEY before running."
fi

say "Checking everything"
cd "$ROOT/core"
source .venv/bin/activate
python -m jarvis doctor || true

cat <<'EOF'

Next:

  1. Fix anything red above.
  2. Add your keys to .env
  3. Start it:

       cd core && source .venv/bin/activate && python -m jarvis run

     and in another terminal:

       cd ui && npm run dev

  4. Open http://localhost:5173 in Chrome, say "hey Jarvis".

  First run downloads ~1 GB of speech models. Be patient once.

EOF
