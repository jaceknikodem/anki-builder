# anki-builder

Generate Anki flashcards from a word list. Calls Gemini for natural example sentences and Kokoro ONNX for local TTS audio.

**Front** — sentence with the target word highlighted, audio plays automatically  
**Back** — English translation of the sentence + word meaning

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

For audio (optional):

```bash
bash setup_kokoro.sh
```

This installs `kokoro-onnx` and downloads the model files (~120 MB) to `~/.cache/kotoba-ai/`. The files are shared with the lang-app so they won't be re-downloaded if already present.

## Usage

```bash
export GEMINI_API_KEY="..."

# Basic
uv run python anki_export.py \
    --words "comprare,vendere,mangiare" \
    --language italian --proficiency a2

# With topic and custom deck name
uv run python anki_export.py \
    --words-file words.txt \
    --language japanese --proficiency b1 \
    --topic travel \
    --deck-name "Japanese B1 — Travel"

# 3 sentences per word, no audio
uv run python anki_export.py \
    --words "hola,gracias,por favor" \
    --language spanish --proficiency newbie \
    --sentences 3 --no-audio

# Custom output path
uv run python anki_export.py \
    --words "bonjour,merci" --language french \
    --output ~/Desktop/french.apkg
```

> `uv run` ensures the project's virtualenv is used. Alternatively, activate it first with `source .venv/bin/activate` and use `python` directly.

## Flags

| Flag | Description |
|------|-------------|
| `--words WORD,WORD,...` | Comma-separated word list |
| `--words-file FILE` | One word per line |
| `--language` | Target language (spanish, japanese, italian, french, portuguese, polish, indonesian, english) |
| `--proficiency` | `newbie` / `a1` / `a2` / `b1` — shapes grammar complexity in sentences |
| `--topic` | Topic context for sentence generation (e.g. food, travel, work) |
| `--sentences` | `2` or `3` sentences per word (default: 2) |
| `--deck-name` | Full Anki deck name. Default: `Kotoba::<Language>` or `Kotoba::<Language>::<Topic>` |
| `--output` | Output `.apkg` path. Default: `<language>_anki.apkg` |
| `--gemini-api-key` | Gemini API key (or set `GEMINI_API_KEY` env var) |
| `--no-audio` | Skip TTS; generates text-only cards. No kokoro-onnx required. |

## Languages and voices

| Language | Voice |
|----------|-------|
| Japanese | jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo |
| English | af_heart, af_bella, af_nicole, am_fenrir, am_michael |
| Spanish | ef_dora |
| French | ff_siwis |
| Italian | if_sara |
| Portuguese | pf_dora |
| Chinese | zf_xiaobei, zf_xiaoni, zm_yunxi |
| Korean | kf_aria, km_junho |

## Dependencies

**Core** (`requirements.txt`):
- `google-genai` — Gemini API with Pydantic structured output
- `genanki` — Anki `.apkg` builder
- `pydantic` — response schema validation (pulled in by google-genai)

**Audio** (installed by `setup_kokoro.sh`):
- `kokoro-onnx` — local TTS inference
- `soundfile` — WAV encoding
- `misaki[ja]` — Japanese phonemization (optional; degrades gracefully if absent)
