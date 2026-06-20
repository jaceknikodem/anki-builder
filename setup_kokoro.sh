#!/usr/bin/env bash
# setup_kokoro.sh — install kokoro-onnx and download model files
#
# Run once before using audio generation:
#   bash setup_kokoro.sh
#
# Model files (~120 MB total) are cached in ~/.cache/kotoba-ai/ and shared
# with the lang-app, so they won't be re-downloaded if already present.

set -euo pipefail

CACHE_DIR="$HOME/.cache/kotoba-ai"
MODEL_FILE="$CACHE_DIR/kokoro-v1.0.int8.onnx"
VOICES_FILE="$CACHE_DIR/voices-v1.0.bin"
MODEL_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# ── venv ──────────────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

PIP=".venv/bin/pip"

# ── base deps (Gemini + Anki) ─────────────────────────────────────────────────
echo "Installing base dependencies..."
"$PIP" install -q -r requirements.txt

# ── audio deps ────────────────────────────────────────────────────────────────
echo "Installing audio dependencies..."
"$PIP" install -q \
  "kokoro-onnx>=0.5.0" \
  "soundfile>=0.14.0" \
  "misaki[ja]>=0.7.4"

# ── model files ───────────────────────────────────────────────────────────────
mkdir -p "$CACHE_DIR"

if [ -f "$MODEL_FILE" ]; then
  echo "Kokoro ONNX model already present — skipping download"
else
  echo "Downloading Kokoro ONNX model (~80 MB)..."
  curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
fi

if [ -f "$VOICES_FILE" ]; then
  echo "Kokoro voices already present — skipping download"
else
  echo "Downloading Kokoro voices (~40 MB)..."
  curl -L --progress-bar -o "$VOICES_FILE" "$VOICES_URL"
fi

# ── smoke test ────────────────────────────────────────────────────────────────
echo ""
echo "Verifying kokoro-onnx loads correctly..."
.venv/bin/python - <<'EOF'
from kokoro_onnx import Kokoro
from pathlib import Path
import os

model  = Path.home() / ".cache/kotoba-ai/kokoro-v1.0.int8.onnx"
voices = Path.home() / ".cache/kotoba-ai/voices-v1.0.bin"
k = Kokoro(str(model), str(voices))
audio, sr = k.create("Hello.", voice="af_heart", lang="en-us")
assert len(audio) > 0, "empty audio"
print(f"  OK — generated {len(audio)} samples at {sr} Hz")
EOF

echo ""
echo "Setup complete. Generate a test deck:"
echo "  GEMINI_API_KEY=... .venv/bin/python anki_export.py \\"
echo "      --words \"hello,thank you\" --language english --proficiency a1"
