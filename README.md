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
| `--language` | Target language (japanese, spanish, french, italian, portuguese, chinese) |
| `--proficiency` | `newbie` / `a1` / `a2` / `b1` — shapes grammar complexity in sentences |
| `--topic` | Topic context for sentence generation (e.g. food, travel, work) |
| `--sentences` | `2` or `3` sentences per word (default: 2) |
| `--deck-name` | Full Anki deck name. Default: `Kotoba::<Language>` or `Kotoba::<Language>::<Topic>` |
| `--output` | Output `.apkg` path. Default: `<language>_anki.apkg` |
| `--gemini-api-key` | Gemini API key (or set `GEMINI_API_KEY` env var) |
| `--tts` | TTS backend: `kokoro` (local, default) or `elevenlabs` (cloud) |
| `--elevenlabs-api-key` | ElevenLabs API key (or set `ELEVENLABS_API_KEY` env var) |
| `--no-audio` | Skip TTS; generates text-only cards. No kokoro-onnx required. |

## Personal phrase cards

Drill sentences you'd actually say. Write one per line, Japanese first, then English
(`#` comments and blank lines are ignored, `1.`-style numbering is stripped):

```
大阪に六年住んでいます。 I've lived in Osaka for six years.
もう一度お願いします。 Once more, please.
```

```bash
uv run python anki_export.py --language japanese --phrases-file ~/Documents/japanese/my_phrases.txt
```

The list can live anywhere — keep it outside the repo, since a personal phrase list tends
to contain names and details you don't want committed. No Gemini key needed.

If the Japanese contains spaces, or you just want to be explicit, separate the two
halves with `|` (`駅は どこですか。 | Where is the station?`). Otherwise the English
starts at the first space-separated word with no Japanese characters.

Each note has one card: the English meaning is the cue, and the
back shows the Japanese with ruby furigana, romaji and audio (auto-played). Notes keep
file order and are identified by their Japanese text, so re-importing an edited list adds
new sentences without touching existing review history. Default deck is
`Kotoba::Japanese::Phrases`; override with `--deck-name`.

Romaji is generated from UniDic readings with a few corrections (日本 → nihon, 私 →
watashi, 十 in 四十メートル, rendaku in 和菓子). If a reading is wrong, add it to
`_KANA_OVERRIDES` or `_COMPOUND_READINGS` in `japanese.py`.

## ElevenLabs audio

```bash
export ELEVENLABS_API_KEY="..."
uv run python anki_export.py --words-file words.txt --language japanese --tts elevenlabs
```

Uses the same voices and models as the lang-app (Japanese, Spanish, Italian, Portuguese;
no French or Chinese voice is configured). kokoro-onnx doesn't need to be installed.
Audio is cached under `~/.cache/kotoba-ai/elevenlabs/`, so re-running an export doesn't
spend credits on sentences you already have.

## Languages and voices

| Language | Voice |
|----------|-------|
| Japanese | jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo |
| Spanish | ef_dora |
| French | ff_siwis |
| Italian | if_sara |
| Portuguese | pf_dora |
| Chinese | zf_xiaobei, zf_xiaoni, zm_yunxi |

## Dependencies

**Core** (`requirements.txt`):
- `google-genai` — Gemini API with Pydantic structured output
- `genanki` — Anki `.apkg` builder
- `pydantic` — response schema validation (pulled in by google-genai)

**Audio** (installed by `setup_kokoro.sh`):
- `kokoro-onnx` — local TTS inference
- `soundfile` — WAV encoding
- `misaki[ja]` — Japanese phonemization (optional; degrades gracefully if absent)
