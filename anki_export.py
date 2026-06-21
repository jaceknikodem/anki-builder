#!/usr/bin/env python3
"""
anki_export.py — Generate Anki flashcards from a word list.

Calls Gemini for 2-3 natural sentences (+ translations) per word, optionally
generates audio via kokoro-onnx locally, and writes a ready-to-import .apkg.

Front: sentence with the target word highlighted  + audio
Back:  English translation of the sentence  +  word meaning

Setup:
    uv pip install -r requirements.txt     # Gemini + Anki
    bash setup_kokoro.sh                   # audio (optional)

Usage:
    python anki_export.py \\
        --words "comprare,vendere,mangiare" \\
        --language italian --proficiency a2 --topic food

    python anki_export.py \\
        --words-file words.txt \\
        --language japanese --proficiency b1 \\
        --deck-name "Japanese B1 — verbs"

    python anki_export.py \\
        --words "hola,gracias" --language spanish \\
        --proficiency newbie --no-audio
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ── required deps ──────────────────────────────────────────────────────────────

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    sys.exit("Missing dep: pip install google-genai")

try:
    from pydantic import BaseModel
except ImportError:
    sys.exit("Missing dep: pip install pydantic")

try:
    import genanki
except ImportError:
    sys.exit("Missing dep: pip install genanki")

# ── audio deps — only imported when audio is enabled (see _load_audio_deps) ───

_sf = None           # soundfile module
_Kokoro = None       # kokoro_onnx.Kokoro class
_kokoro = None       # loaded Kokoro instance
_ja_g2p = None       # misaki Japanese G2P (optional)
_ja_tagger = None    # fugashi MeCab tagger (optional, Japanese furigana)

CACHE_DIR  = Path.home() / ".cache" / "kotoba-ai"
MODEL_PATH = CACHE_DIR / "kokoro-v1.0.int8.onnx"
VOICES_PATH = CACHE_DIR / "voices-v1.0.bin"


def _load_audio_deps() -> None:
    """Import audio libraries and verify model files exist. Exits on failure."""
    global _sf, _Kokoro, _ja_g2p

    missing = []
    if not MODEL_PATH.exists():
        missing.append(f"model file: {MODEL_PATH}")
    if not VOICES_PATH.exists():
        missing.append(f"voices file: {VOICES_PATH}")
    if missing:
        print("Kokoro model files not found:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("Run:  bash setup_kokoro.sh", file=sys.stderr)
        sys.exit(1)

    try:
        import soundfile as sf
        _sf = sf
    except ImportError:
        sys.exit("Missing audio dep: run  bash setup_kokoro.sh")

    try:
        from kokoro_onnx import Kokoro
        _Kokoro = Kokoro
    except ImportError:
        sys.exit("Missing audio dep: run  bash setup_kokoro.sh")

    try:
        from misaki import ja as misaki_ja
        _ja_g2p = misaki_ja.JAG2P()
    except ImportError:
        pass  # Japanese phonemization degrades gracefully


# ── language / voice config ────────────────────────────────────────────────────

VOICES: dict[str, list[str]] = {
    "japanese":   ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "english":    ["af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael"],
    "spanish":    ["ef_dora"],
    "french":     ["ff_siwis"],
    "italian":    ["if_sara"],
    "portuguese": ["pf_dora"],
    "chinese":    ["zf_xiaobei", "zf_xiaoni", "zm_yunxi"],
    "korean":     ["kf_aria", "km_junho"],
}

LANG_TO_ESPEAK: dict[str, str] = {
    "english":    "en-us",
    "spanish":    "es",
    "french":     "fr-fr",
    "italian":    "it",
    "portuguese": "pt-br",
    "chinese":    "cmn",
    "korean":     "ko",
    "polish":     "pl",
    "indonesian": "id",
}

GRAMMAR_GUIDANCE: dict[str, dict[str, str]] = {
    "italian": {
        "newbie": "presente (essere/avere/regular), simple S-V patterns, fixed chunks (posso/devo/voglio)",
        "a1":     "presente all persons, passato prossimo recognition, modal + infinitive, imperatives",
        "a2":     "productive passato prossimo, imperfetto/futuro recognition, gerundio progressivo",
        "b1":     "productive imperfetto, full condizionale, congiuntivo recognition, clitics",
    },
    "spanish": {
        "newbie": "presente (ser/estar/tener + regular), simple S-V patterns",
        "a1":     "presente all persons, basic reflexives, modal + infinitive, periphrastic future",
        "a2":     "pretérito perfecto compuesto, estar+gerundio, basic condicional, gustar verbs",
        "b1":     "present subjunctive, advanced connectors, conversational idioms",
    },
    "portuguese": {
        "newbie": "presente (ser/estar/ter/regular), simple S-V patterns",
        "a1":     "presente all persons, pretérito perfeito recognition, poder/precisar + infinitive",
        "a2":     "pretérito perfeito, estar+gerúndio, basic condicional (gostaria/poderia)",
        "b1":     "pretérito imperfeito, futuro do presente, full condicional, subjuntivo recognition",
    },
    "japanese": {
        "newbie": "copula (です/ではありません), simple polite verb forms, basic particles (は・が・を・に)",
        "a1":     "polite past (〜ました), te-form recognition, あります/います",
        "a2":     "te-form usage, plain forms, potential (〜られる/〜できる), basic conditionals (〜たら)",
        "b1":     "full plain-form conjugation, extended te-forms (〜てしまう), passive/causative",
    },
    "polish": {
        "newbie": "basic present tense (być/mieć), S-V-O, simple negation, noun gender",
        "a1":     "present tense all persons, past recognition, modals (mogę/chcę/muszę), basic cases",
        "a2":     "past tense all genders, future (będę + inf), accusative/dative in predictable patterns",
        "b1":     "perfective/imperfective, conditional forms, instrumental/genitive, subordinate clauses",
    },
    "french": {
        "newbie": "présent (être/avoir/aller/regular -er), S-V-O, basic negation (ne…pas)",
        "a1":     "présent all persons, passé composé recognition, modal + inf (pouvoir/vouloir/devoir)",
        "a2":     "productive passé composé, imparfait recognition, futur proche, pronoms COD",
        "b1":     "imparfait vs passé composé, futur simple, conditionnel, subjonctif recognition",
    },
    "indonesian": {
        "newbie": "simple S-V-O, basic stative adjectives, negation (tidak/bukan), common pronouns",
        "a1":     "me-/ber- prefixes, question words, possessives -nya, reduplication",
        "a2":     "di- passives, modals (bisa/harus/mau), aspect markers (sedang/sudah/belum/akan)",
        "b1":     "varied affix combinations, full passive system, complex clauses (kalau/meskipun)",
    },
}


def _voice(language: str) -> str:
    return VOICES.get(language.lower(), ["af_heart"])[0]


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class SentenceItem(BaseModel):
    sentence: str
    translation: str


class WordData(BaseModel):
    word_translation: str
    sentences: list[SentenceItem]


# ── Gemini ─────────────────────────────────────────────────────────────────────

def _proficiency_note(proficiency: Optional[str], language: str) -> str:
    if not proficiency:
        return ""
    guidance = GRAMMAR_GUIDANCE.get(language.lower(), {}).get(proficiency.lower())
    if guidance:
        return (
            f"\nIMPORTANT: Proficiency is {proficiency.upper()}. "
            f"Use grammar appropriate for this level: {guidance}"
        )
    return f"\nIMPORTANT: Proficiency is {proficiency.upper()}. Adjust complexity accordingly."


def fetch_word_data(
    word: str,
    language: str,
    proficiency: Optional[str],
    topic: Optional[str],
    sentence_count: int,
    client: genai.Client,
) -> WordData:
    """
    Calls Gemini with structured output to get word translation + example sentences.
    Retries up to 3 times on transient errors.
    """
    script_note = (
        "\nCRITICAL: Write ALL text in kanji/kana (Japanese script), NOT romaji."
        if language.lower() == "japanese"
        else ""
    )
    proficiency_note = _proficiency_note(proficiency, language)
    topic_note = f'\nIMPORTANT: Sentences must relate to the topic: "{topic}"' if topic else ""

    prompt = (
        f"For the {language} word '{word}', provide its English translation and "
        f"exactly {sentence_count} natural, short (5-15 words) example sentences.{script_note}"
        f"{proficiency_note}{topic_note}\n\n"
        f"Every sentence must contain '{word}' or its conjugated/inflected form. "
        f"Each sentence must be different and conversational."
    )

    last_err: Exception = RuntimeError("unknown")
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=WordData,
                    temperature=0.8,
                ),
            )
            data: WordData = (
                response.parsed
                if response.parsed is not None
                else WordData.model_validate_json(response.text)
            )
            return data
        except Exception as exc:
            last_err = exc
            if attempt < 2:
                wait = (attempt + 1) * 4
                print(f"    retry {attempt + 1}/3: {exc} (waiting {wait}s)")
                time.sleep(wait)
    raise last_err


# ── Kokoro TTS (direct, no HTTP) ───────────────────────────────────────────────

def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        print("  Loading Kokoro model...", flush=True)
        _kokoro = _Kokoro(str(MODEL_PATH), str(VOICES_PATH))
        print("  Kokoro ready", flush=True)
    return _kokoro


def generate_audio(text: str, language: str) -> bytes:
    """Synthesise WAV audio and return raw bytes."""
    voice = _voice(language)
    kokoro = _get_kokoro()
    lang = language.lower()

    if lang == "japanese" and _ja_g2p is not None:
        ipa, _ = _ja_g2p(text)
        audio, sr = kokoro.create(ipa, voice=voice, is_phonemes=True)
    else:
        espeak_lang = LANG_TO_ESPEAK.get(lang, "en-us")
        audio, sr = kokoro.create(text, voice=voice, lang=espeak_lang)

    buf = io.BytesIO()
    _sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


# ── Highlighting ───────────────────────────────────────────────────────────────

def highlight_word(sentence: str, word: str) -> str:
    """Return HTML-escaped sentence with the first match of `word` highlighted."""
    escaped = html.escape(sentence)
    escaped_word = html.escape(word)
    if escaped_word and escaped_word in escaped:
        escaped = escaped.replace(
            escaped_word, f'<span class="kw">{escaped_word}</span>', 1
        )
    return escaped


# ── Furigana (Japanese only) ───────────────────────────────────────────────────

def _get_ja_tagger():
    global _ja_tagger
    if _ja_tagger is None:
        try:
            import fugashi
            _ja_tagger = fugashi.Tagger()
        except Exception:
            _ja_tagger = False  # disable gracefully
    return _ja_tagger if _ja_tagger else None


def _kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ン" else c for c in s)


def _has_kanji(s: str) -> bool:
    return any("一" <= c <= "鿿" or "㐀" <= c <= "䶿" for c in s)


def furigana_highlight_html(sentence: str, word: str) -> str:
    """Return sentence as HTML with ruby furigana; the target word is highlighted."""
    tagger = _get_ja_tagger()
    if tagger is None:
        return highlight_word(sentence, word)

    morphemes = list(tagger(sentence))
    surface_text = "".join(m.surface for m in morphemes)
    word_start = surface_text.find(word)
    word_end = word_start + len(word) if word_start != -1 else -1

    parts: list[str] = []
    pos = 0
    for m in morphemes:
        surface = m.surface
        m_end = pos + len(surface)
        in_word = word_start != -1 and pos >= word_start and m_end <= word_end

        kana = getattr(m.feature, "kana", None)
        if _has_kanji(surface) and kana and kana != "*" and kana != surface:
            reading = _kata_to_hira(kana)
            inner = f"<ruby>{html.escape(surface)}<rt>{html.escape(reading)}</rt></ruby>"
        else:
            inner = html.escape(surface)

        parts.append(f'<span class="kw">{inner}</span>' if in_word else inner)
        pos = m_end

    return "".join(parts)


# ── Anki model ─────────────────────────────────────────────────────────────────

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
ruby { ruby-align: center; }
rt { font-size: 0.5em; color: #555; }
"""

SENTENCE_MODEL = genanki.Model(
    1758600000003,
    "Kotoba Sentence Export",
    fields=[
        {"name": "Sentence"},     # HTML, word highlighted
        {"name": "Translation"},  # English sentence translation
        {"name": "Audio"},        # [sound:xxx.wav]  or empty
        {"name": "Word"},         # "word — meaning"
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


def _deck_id(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") & 0x7FFF_FFFF


# ── Build .apkg ────────────────────────────────────────────────────────────────

def build_apkg(
    words_data: list[dict],
    deck_name: str,
    output_path: str,
    language: str = "",
) -> int:
    """
    Each entry in words_data:
        {word, word_translation, sentences: [{sentence, translation, audio_bytes?}]}
    Returns card count.
    """
    deck = genanki.Deck(_deck_id(deck_name), deck_name)
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
                    filename = f"kotoba_{slug}.wav"
                    tmp_path = os.path.join(tmp_dir, filename)
                    with open(tmp_path, "wb") as fh:
                        fh.write(audio_bytes)
                    media_paths.append(tmp_path)
                    audio_field = f"[sound:{filename}]"

                sentence_html = (
                    furigana_highlight_html(sent["sentence"], word)
                    if language.lower() == "japanese"
                    else highlight_word(sent["sentence"], word)
                )
                note = genanki.Note(
                    model=SENTENCE_MODEL,
                    fields=[
                        sentence_html,
                        html.escape(sent["translation"]),
                        audio_field,
                        html.escape(f"{word} — {word_translation}"),
                    ],
                    guid=genanki.guid_for(f"kotoba-export:{deck_name}:{word}:{idx}"),
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
        description="Generate Anki flashcards from a word list (Gemini + Kokoro TTS)",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--words", metavar="WORD,WORD,...",
                     help="Comma-separated list of words")
    src.add_argument("--words-file", metavar="FILE",
                     help="One word per line")

    parser.add_argument("--language", required=True,
                        help="Target language, e.g. spanish, japanese, italian")
    parser.add_argument("--proficiency", choices=["newbie", "a1", "a2", "b1"],
                        help="Learner proficiency level (recommended)")
    parser.add_argument("--topic",
                        help="Topic context for sentence generation")
    parser.add_argument("--sentence-count", type=int, default=2, choices=range(1, 6),
                        help="Sentences per word, 1–5 (default: 2)")
    parser.add_argument("--deck-name", metavar="NAME",
                        help="Anki deck name (default: Kotoba::<Language>[::<Topic>])")
    parser.add_argument("--output", metavar="FILE",
                        help="Output .apkg path (default: <language>_anki.apkg)")
    parser.add_argument("--gemini-api-key", metavar="KEY",
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip audio generation; text-only cards")

    args = parser.parse_args()

    # ── API key ──────────────────────────────────────────────────────────────
    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        parser.error("Gemini API key required: --gemini-api-key or GEMINI_API_KEY env var")

    client = genai.Client(api_key=api_key)

    # ── Audio deps (lazy) ─────────────────────────────────────────────────────
    if not args.no_audio:
        _load_audio_deps()

    # ── Word list ─────────────────────────────────────────────────────────────
    if args.words:
        words = [w.strip() for w in args.words.split(",") if w.strip()]
    else:
        with open(args.words_file, encoding="utf-8") as fh:
            words = [line.strip() for line in fh if line.strip()]

    if not words:
        parser.error("No words found")

    # ── Names ─────────────────────────────────────────────────────────────────
    deck_name = args.deck_name or (
        f"Kotoba::{args.language.capitalize()}"
        + (f"::{args.topic.capitalize()}" if args.topic else "")
    )
    output_path = args.output or f"{args.language}_anki.apkg"

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"Language   : {args.language}")
    print(f"Words      : {len(words)}  ({', '.join(words[:6])}{'...' if len(words) > 6 else ''})")
    if args.proficiency:
        print(f"Proficiency: {args.proficiency.upper()}")
    if args.topic:
        print(f"Topic      : {args.topic}")
    print(f"Sentences  : {args.sentence_count} per word")
    print(f"Deck       : {deck_name}")
    print(f"Audio      : {'off' if args.no_audio else 'kokoro-onnx'}")
    print(f"Output     : {output_path}")
    print()

    # ── Step 1: Sentences ─────────────────────────────────────────────────────
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
                sentence_count=args.sentence_count,
                client=client,
            )
            sentences = [
                {"sentence": s.sentence, "translation": s.translation, "audio_bytes": None}
                for s in data.sentences[: args.sentence_count]
            ]
            words_data.append({
                "word": word,
                "word_translation": data.word_translation,
                "sentences": sentences,
            })
            print(f'→ "{data.word_translation}" ({len(sentences)} sentences)')
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            print(f"  skipping '{word}'")

    if not words_data:
        sys.exit("No words processed.")

    # ── Step 2: Audio ─────────────────────────────────────────────────────────
    total = sum(len(w["sentences"]) for w in words_data)

    if args.no_audio:
        print(f"\n── Audio skipped ───────────────────────────────────────────────")
    else:
        print(f"\n── Generating audio ({total} sentences via kokoro-onnx) ─────────")
        ok = 0
        for word_info in words_data:
            for sent in word_info["sentences"]:
                try:
                    sent["audio_bytes"] = generate_audio(sent["sentence"], args.language)
                    ok += 1
                    print(".", end="", flush=True)
                except Exception as exc:
                    print(f"\n  Warning: audio failed for '{sent['sentence'][:40]}': {exc}",
                          file=sys.stderr)
        print(f"\n  {ok}/{total} audio files generated")

    # ── Step 3: Package ───────────────────────────────────────────────────────
    print(f"\n── Writing {output_path} ──────────────────────────────────────────")
    card_count = build_apkg(words_data, deck_name, output_path, language=args.language)
    print(f"Done — {card_count} cards → {output_path}")


if __name__ == "__main__":
    main()
