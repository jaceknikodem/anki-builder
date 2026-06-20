#!/usr/bin/env python3
"""
anki_export.py — Generate Anki flashcards from a word list.

Hits Gemini for 2-3 example sentences (+ translations) per word, then Kokoro
TTS locally for audio, and bundles everything into an importable .apkg file.

Front card: target-language sentence with the word highlighted + audio
Back card: English translation of the sentence + word translation

Requirements:
    pip install -r requirements.txt

Usage:
    python anki_export.py --words "comprare,vendere,mangiare" \\
        --language italian --proficiency a2

    python anki_export.py --words-file words.txt \\
        --language japanese --proficiency b1 --topic travel

    # skip audio if Kokoro is not running
    python anki_export.py --words "hola,gracias" --language spanish \\
        --proficiency newbie --no-audio
"""

import argparse
import base64
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
import time
from typing import Optional

try:
    import genanki
except ImportError:
    print("Missing dependency: pip install genanki", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Missing dependency: pip install requests", file=sys.stderr)
    sys.exit(1)


# ── Language / voice config ────────────────────────────────────────────────────

VOICES_BY_LANGUAGE: dict[str, list[str]] = {
    "japanese":   ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "english":    ["af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael"],
    "spanish":    ["ef_dora"],
    "french":     ["ff_siwis"],
    "italian":    ["if_sara"],
    "portuguese": ["pf_dora"],
    "chinese":    ["zf_xiaobei", "zf_xiaoni", "zm_yunxi"],
    "korean":     ["kf_aria", "km_junho"],
}
DEFAULT_VOICE = "af_heart"

# Grammar notes per language+level — fed into the Gemini prompt.
GRAMMAR_GUIDANCE: dict[str, dict[str, str]] = {
    "italian": {
        "newbie": "presente (essere/avere/regular), simple S-V and S-V-O, fixed chunks (posso/devo/voglio)",
        "a1":     "presente all persons, passato prossimo recognition, modal + infinitive, basic imperatives",
        "a2":     "productive passato prossimo, imperfetto/futuro recognition, gerundio progressivo (sto + gerundio)",
        "b1":     "productive imperfetto, full condizionale presente, congiuntivo recognition, clitic combinations",
    },
    "spanish": {
        "newbie": "presente (ser/estar/tener/haber/ir/hacer + regular), simple S-V patterns",
        "a1":     "presente all persons, basic reflexives, modal + infinitive, periphrastic future (ir a + inf)",
        "a2":     "pretérito perfecto compuesto, estar+gerundio, basic condicional, gustar-type verbs",
        "b1":     "present subjunctive, advanced connectors (sin embargo, a pesar de), conversational idioms",
    },
    "portuguese": {
        "newbie": "presente (ser/estar/ter/haver/regular), simple S-V patterns",
        "a1":     "presente all persons, pretérito perfeito recognition, poder/precisar/querer + infinitive",
        "a2":     "pretérito perfeito, estar+gerúndio, basic condicional (gostaria/poderia)",
        "b1":     "pretérito imperfeito, futuro do presente, full condicional, subjuntivo recognition",
    },
    "japanese": {
        "newbie": "copula forms (です/ではありません), simple verb polite forms, basic particles (は・が・を・に)",
        "a1":     "polite past (〜ました), te-form recognition, existence verbs (あります/います)",
        "a2":     "te-form usage, informal/plain forms, potential form (〜られる/〜できる), basic conditionals (〜たら)",
        "b1":     "full plain-form conjugation, extended te-forms (〜てしまう), passive/causative recognition",
    },
    "polish": {
        "newbie": "basic present tense (być/mieć), S-V-O patterns, simple negation, noun gender (m/f/n)",
        "a1":     "present tense all persons, past tense recognition, modals (mogę/chcę/muszę + inf), basic cases",
        "a2":     "past tense all genders, future (będę + inf), accusative/dative/locative in predictable patterns",
        "b1":     "perfective/imperfective contrast, conditional forms, instrumental/genitive, subordinate clauses",
    },
    "french": {
        "newbie": "présent (être/avoir/aller/regular -er), simple S-V-O, basic negation (ne…pas)",
        "a1":     "présent all persons, passé composé recognition, modal + inf (pouvoir/vouloir/devoir), imperatives",
        "a2":     "productive passé composé, imparfait recognition, futur proche (aller + inf), pronoms COD",
        "b1":     "imparfait vs passé composé, futur simple, conditionnel présent, subjonctif recognition",
    },
    "indonesian": {
        "newbie": "simple S-V and S-V-O, basic stative adjectives, simple negation (tidak/bukan), common pronouns",
        "a1":     "me- and ber- prefixes in common verbs, negation patterns, basic question words, possessives with -nya",
        "a2":     "di- passives, modal verbs (bisa/harus/mau), aspect markers (sedang/sudah/belum/akan), yang clauses",
        "b1":     "varied affix combinations, full passive system (di-/ter-), complex clause structures (kalau/meskipun)",
    },
}


def get_voice(language: str) -> str:
    return VOICES_BY_LANGUAGE.get(language.lower(), [DEFAULT_VOICE])[0]


# ── Gemini API ─────────────────────────────────────────────────────────────────

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/gemini-2.5-flash:generateContent"
)


def _gemini_call(prompt: str, api_key: str, temperature: float = 0.7) -> str:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 4096,
        },
    }
    resp = requests.post(f"{GEMINI_URL}?key={api_key}", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("No candidates in Gemini response")
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise ValueError("Empty parts in Gemini response")
    return parts[0]["text"]


def _clean_json(text: str) -> str:
    """Strip markdown fences and return the bare JSON string."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _proficiency_note(proficiency: Optional[str], language: str) -> str:
    if not proficiency:
        return ""
    guidance = GRAMMAR_GUIDANCE.get(language.lower(), {}).get(proficiency.lower())
    if guidance:
        return (
            f"\nIMPORTANT: Proficiency level is {proficiency.upper()}. "
            f"Use grammar appropriate for this level: {guidance}"
        )
    return f"\nIMPORTANT: Proficiency level is {proficiency.upper()}. Adjust sentence complexity accordingly."


def fetch_word_data(
    word: str,
    language: str,
    proficiency: Optional[str],
    topic: Optional[str],
    sentence_count: int,
    api_key: str,
) -> dict:
    """
    Returns {"word_translation": str, "sentences": [{"sentence": str, "translation": str}, ...]}.
    Retries up to 3 times on transient failures.
    """
    lang = language.lower()
    script_note = (
        "\nCRITICAL: Write ALL text in kanji/kana (Japanese script), NOT romaji."
        if lang == "japanese"
        else ""
    )
    proficiency_note = _proficiency_note(proficiency, language)
    topic_note = f'\nIMPORTANT: Sentences must relate to the topic: "{topic}"' if topic else ""

    prompt = f"""CRITICAL: Return ONLY a JSON object — no markdown, no explanation.{script_note}

Task: For the {language} word '{word}', provide its English translation and exactly {sentence_count} natural example sentences in {language}.{proficiency_note}{topic_note}

Expected JSON format:
{{
  "word_translation": "concise English meaning of '{word}'",
  "sentences": [
    {{"sentence": "example sentence using {word} (or its conjugated form)", "translation": "English translation of the sentence"}},
    ... ({sentence_count} items total)
  ]
}}

Rules:
1. word_translation: 1-4 words, concise English definition
2. sentences: exactly {sentence_count} items, each different
3. Every sentence must contain '{word}' or an appropriate conjugated/inflected form
4. Keep sentences short (5-15 words) and conversational
5. Return ONLY the raw JSON object"""

    last_err: Exception = RuntimeError("unknown error")
    for attempt in range(3):
        try:
            raw = _gemini_call(prompt, api_key, temperature=0.7)
            data = json.loads(_clean_json(raw))
            if "word_translation" not in data or not data.get("sentences"):
                raise ValueError("Missing required fields in Gemini response")
            return data
        except Exception as exc:
            last_err = exc
            if attempt < 2:
                wait = (attempt + 1) * 4
                print(f"    Retry {attempt + 1}/3 for '{word}': {exc} (waiting {wait}s)")
                time.sleep(wait)
    raise last_err


# ── Kokoro TTS ─────────────────────────────────────────────────────────────────

def fetch_audio_batch(
    sentences: list[str],
    language: str,
    kokoro_url: str,
) -> list[Optional[bytes]]:
    """
    POST to /tts-batch without output_path so the server returns base64 audio_data.
    Returns list of WAV bytes (None where generation failed or server unreachable).
    """
    voice = get_voice(language)
    payload = [{"text": s, "language": language, "voice": voice} for s in sentences]
    try:
        resp = requests.post(f"{kokoro_url}/tts-batch", json=payload, timeout=120)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        out: list[Optional[bytes]] = []
        for r in results:
            if r.get("success") and r.get("audio_data"):
                out.append(base64.b64decode(r["audio_data"]))
            else:
                out.append(None)
        return out
    except Exception as exc:
        print(f"  Warning: Kokoro TTS failed: {exc}", file=sys.stderr)
        return [None] * len(sentences)


# ── Highlighting ───────────────────────────────────────────────────────────────

def highlight_word(sentence: str, word: str) -> str:
    """Wrap the first occurrence of `word` in a highlight span (HTML-safe)."""
    escaped = html.escape(sentence)
    escaped_word = html.escape(word)
    if not escaped_word or escaped_word not in escaped:
        return escaped
    return escaped.replace(escaped_word, f'<span class="kw">{escaped_word}</span>', 1)


# ── Anki card model ────────────────────────────────────────────────────────────

_MODEL_ID = 1758600000003  # distinct from the Electron app's model id

_MODEL_CSS = """\
.card {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 20px;
  text-align: center;
  color: #1c1c1e;
  background: #ffffff;
}
.sentence { font-size: 26px; margin: 16px 0; line-height: 1.6; }
.sentence .kw { color: #2563eb; font-weight: 700; }
.translation { margin-top: 12px; color: #444; }
.word { color: #888; font-size: 16px; margin-top: 14px; font-style: italic; }
hr#answer { margin: 18px 0; border: none; border-top: 1px solid #e0e0e0; }
"""

SENTENCE_MODEL = genanki.Model(
    _MODEL_ID,
    "Kotoba Sentence Export",
    fields=[
        {"name": "Sentence"},     # HTML with word highlighted
        {"name": "Translation"},  # English sentence translation
        {"name": "Audio"},        # [sound:xxx.wav]
        {"name": "Word"},         # "word — english meaning"
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": '{{Audio}}<div class="sentence">{{Sentence}}</div>',
            "afmt": (
                "{{FrontSide}}\n<hr id=answer>\n"
                '<div class="translation">{{Translation}}</div>'
                '{{#Word}}<div class="word">{{Word}}</div>{{/Word}}'
            ),
        }
    ],
    css=_MODEL_CSS,
)


def _stable_deck_id(name: str) -> int:
    digest = hashlib.sha256(name.encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


# ── Build .apkg ────────────────────────────────────────────────────────────────

def build_apkg(
    words_data: list[dict],
    language: str,
    topic: Optional[str],
    output_path: str,
) -> int:
    """
    Each entry in `words_data`:
        {word, word_translation, sentences: [{sentence, translation, audio_bytes}]}

    Returns the number of cards written.
    """
    deck_name = f"Kotoba::{language.capitalize()}"
    if topic:
        deck_name += f"::{topic.capitalize()}"

    deck = genanki.Deck(_stable_deck_id(deck_name), deck_name)

    tmp_dir = tempfile.mkdtemp(prefix="kotoba-anki-")
    media_paths: list[str] = []

    try:
        for word_info in words_data:
            word = word_info["word"]
            word_translation = word_info["word_translation"]

            for idx, sent in enumerate(word_info["sentences"]):
                audio_bytes: Optional[bytes] = sent.get("audio_bytes")

                audio_field = ""
                if audio_bytes:
                    slug = hashlib.md5(sent["sentence"].encode()).hexdigest()[:10]
                    filename = f"kotoba_{language}_{slug}.wav"
                    tmp_path = os.path.join(tmp_dir, filename)
                    with open(tmp_path, "wb") as fh:
                        fh.write(audio_bytes)
                    media_paths.append(tmp_path)
                    audio_field = f"[sound:{filename}]"

                note = genanki.Note(
                    model=SENTENCE_MODEL,
                    fields=[
                        highlight_word(sent["sentence"], word),
                        html.escape(sent["translation"]),
                        audio_field,
                        html.escape(f"{word} — {word_translation}"),
                    ],
                    guid=genanki.guid_for(f"kotoba-export:{language}:{word}:{idx}"),
                )
                deck.add_note(note)

        pkg = genanki.Package(deck)
        pkg.media_files = media_paths
        pkg.write_to_file(output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return len(deck.notes)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Anki flashcards from a word list using Gemini + Kokoro TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    word_src = parser.add_mutually_exclusive_group(required=True)
    word_src.add_argument(
        "--words", metavar="WORD,WORD,...",
        help="Comma-separated list of target-language words",
    )
    word_src.add_argument(
        "--words-file", metavar="FILE",
        help="Plain-text file with one word per line",
    )

    parser.add_argument("--language", required=True,
                        help="Target language, e.g. spanish, japanese, italian")
    parser.add_argument("--proficiency", choices=["newbie", "a1", "a2", "b1"],
                        help="Learner proficiency level (optional but recommended)")
    parser.add_argument("--topic",
                        help="Optional topic context for sentence generation")
    parser.add_argument("--sentences", type=int, default=2, choices=[2, 3],
                        help="Sentences to generate per word (default: 2)")
    parser.add_argument("--output", metavar="FILE",
                        help="Output .apkg path (default: <language>_anki.apkg)")
    parser.add_argument("--gemini-api-key", metavar="KEY",
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--kokoro-url", default="http://localhost:8000",
                        help="Kokoro TTS server URL (default: http://localhost:8000)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip Kokoro TTS; generate text-only cards")

    args = parser.parse_args()

    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        parser.error("Gemini API key required: --gemini-api-key or GEMINI_API_KEY env var")

    if args.words:
        words = [w.strip() for w in args.words.split(",") if w.strip()]
    else:
        with open(args.words_file, encoding="utf-8") as fh:
            words = [line.strip() for line in fh if line.strip()]

    if not words:
        parser.error("No words provided")

    output_path = args.output or f"{args.language}_anki.apkg"

    print(f"Language   : {args.language}")
    print(f"Words      : {len(words)}  ({', '.join(words[:6])}{'...' if len(words) > 6 else ''})")
    if args.proficiency:
        print(f"Proficiency: {args.proficiency.upper()}")
    if args.topic:
        print(f"Topic      : {args.topic}")
    print(f"Sentences  : {args.sentences} per word")
    print(f"Audio      : {'disabled (--no-audio)' if args.no_audio else args.kokoro_url}")
    print(f"Output     : {output_path}")
    print()

    # ── Step 1: Sentences via Gemini ──────────────────────────────────────────
    print("── Generating sentences ────────────────────────────────────────")
    words_data: list[dict] = []
    for word in words:
        print(f"  {word}...", end=" ", flush=True)
        try:
            data = fetch_word_data(
                word=word,
                language=args.language,
                proficiency=args.proficiency,
                topic=args.topic,
                sentence_count=args.sentences,
                api_key=api_key,
            )
            sentences = [
                {"sentence": s["sentence"], "translation": s["translation"], "audio_bytes": None}
                for s in data["sentences"][: args.sentences]
            ]
            words_data.append({
                "word": word,
                "word_translation": data["word_translation"],
                "sentences": sentences,
            })
            print(f'→ "{data["word_translation"]}" ({len(sentences)} sentences)')
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            print(f"  Skipping '{word}'")

    if not words_data:
        print("No words processed. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: Audio via Kokoro ──────────────────────────────────────────────
    total_sentences = sum(len(w["sentences"]) for w in words_data)

    if args.no_audio:
        print(f"\n── Audio skipped ───────────────────────────────────────────────")
    else:
        print(f"\n── Generating audio ({total_sentences} sentences) ────────────────────────")

        flat_sentences: list[str] = []
        index_map: list[tuple[int, int]] = []
        for wi, wi_data in enumerate(words_data):
            for si, sent in enumerate(wi_data["sentences"]):
                flat_sentences.append(sent["sentence"])
                index_map.append((wi, si))

        audio_results = fetch_audio_batch(flat_sentences, args.language, args.kokoro_url)

        ok = 0
        for i, audio_bytes in enumerate(audio_results):
            wi, si = index_map[i]
            words_data[wi]["sentences"][si]["audio_bytes"] = audio_bytes
            if audio_bytes:
                ok += 1

        print(f"  {ok}/{total_sentences} audio files generated")

    # ── Step 3: Build .apkg ───────────────────────────────────────────────────
    print(f"\n── Building {output_path} ─────────────────────────────────────────")
    card_count = build_apkg(words_data, args.language, args.topic, output_path)
    print(f"Done — {card_count} cards written to {output_path}")


if __name__ == "__main__":
    main()
