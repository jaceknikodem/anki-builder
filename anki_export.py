#!/usr/bin/env python3
"""Generate Anki flashcards from a word list. See README.md for usage."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import fugashi
import genanki
import soundfile as sf
from google import genai
from google.genai import types as genai_types
from kokoro_onnx import Kokoro
from misaki import ja as misaki_ja
from pydantic import BaseModel

_ja_g2p = None       # misaki Japanese G2P instance
_ja_g2p_lock = threading.Lock()  # misaki is not documented thread-safe
_ja_tagger = None    # fugashi MeCab tagger (Japanese furigana)

CACHE_DIR  = Path.home() / ".cache" / "kotoba-ai"
MODEL_PATH = CACHE_DIR / "kokoro-v1.0.int8.onnx"
VOICES_PATH = CACHE_DIR / "voices-v1.0.bin"

_kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))


def _load_audio_deps() -> None:
    """Initialize Japanese language processing deps."""
    global _ja_g2p, _ja_tagger
    _ja_g2p = misaki_ja.JAG2P()
    _ja_tagger = fugashi.Tagger()


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

GRAMMAR_GUIDANCE: dict[str, dict[str, str]] = json.loads(
    (Path(__file__).parent / "grammar_guidance.json").read_text(encoding="utf-8")
)


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

def generate_audio(text: str, language: str) -> bytes:
    """Synthesise WAV audio and return raw bytes."""
    voice = _voice(language)
    lang = language.lower()

    if lang == "japanese" and _ja_g2p is not None:
        with _ja_g2p_lock:
            ipa, _ = _ja_g2p(text)
        audio, sr = _kokoro.create(ipa, voice=voice, is_phonemes=True)
    else:
        espeak_lang = LANG_TO_ESPEAK.get(lang, "en-us")
        audio, sr = _kokoro.create(text, voice=voice, lang=espeak_lang)

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
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

def _kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ン" else c for c in s)


def _has_kanji(s: str) -> bool:
    return any("一" <= c <= "鿿" or "㐀" <= c <= "䶿" for c in s)


def furigana_highlight_html(sentence: str, word: str) -> str:
    """Return sentence as HTML with ruby furigana; the target word is highlighted."""
    tagger = _ja_tagger
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


def sentence_to_hiragana(sentence: str) -> str:
    """Return a flat hiragana transcription of a Japanese sentence using fugashi readings."""
    tagger = _ja_tagger
    if tagger is None:
        return _kata_to_hira(sentence)
    parts: list[str] = []
    for m in tagger(sentence):
        kana = getattr(m.feature, "kana", None)
        parts.append(_kata_to_hira(kana) if kana and kana != "*" else m.surface)
    return "".join(parts)


# ── Anki model ─────────────────────────────────────────────────────────────────

# SentencesJSON field: [{s:<html>, t:<plain>, a:<filename|"">, p:<plain>, h?:<hiragana>}, ...]
# Audios field:        [sound:f1.wav][sound:f2.wav]... — kept in field values so Anki's
#                      media manager counts the files as used; NOT rendered in templates.
# JS plays the selected sentence's audio directly via the HTML5 Audio API.

_TMPL_DIR = Path(__file__).parent / "templates"

def _tmpl(name: str) -> str:
    return (_TMPL_DIR / name).read_text(encoding="utf-8")

_MODEL_CSS  = _tmpl("card.css")
_FRONT_TMPL = _tmpl("front.html")
_BACK_TMPL  = _tmpl("back.html")

SENTENCE_MODEL = genanki.Model(
    1758600000006,
    "Kotoba Sentence v2",
    fields=[
        {"name": "SentencesJSON"},  # JSON array of sentence objects
        {"name": "Audios"},         # [sound:f1.wav][sound:f2.wav]... for native Anki audio
        {"name": "Word"},           # "word — meaning"
    ],
    templates=[{"name": "Card 1", "qfmt": _FRONT_TMPL, "afmt": _BACK_TMPL}],
    css=_MODEL_CSS,
)

# ── Pronunciation card model ────────────────────────────────────────────────────
# SentencesJSON items carry two extra fields used only by this template:
#   p — plain sentence text (no HTML), for scoring against kanji input
#   h — flat hiragana transcription, for scoring against kana input (Japanese only)

# Pronunciation cards use {{type:SentencePlain}} solely for iOS keyboard/mic input.
# Anki's own comparison display (#typeans) is hidden; we extract the typed text from
# it on the back and run our own Levenshtein scoring against both kanji and hiragana.
# Each pronunciation note always uses sentence index 0 (fixed target for {{type:}}).

_PRON_CSS        = _MODEL_CSS + _tmpl("pron_extra.css")
_PRON_FRONT_TMPL = _tmpl("pron_front.html")
_PRON_BACK_TMPL  = _tmpl("pron_back.html")

PRONUNCIATION_MODEL = genanki.Model(
    1758600000008,
    "Kotoba Pronunciation v2",
    fields=[
        {"name": "SentencesJSON"},
        {"name": "Audios"},
        {"name": "SentencePlain"},  # plain text of sentence 0, target for {{type:}}
        {"name": "Word"},
    ],
    templates=[{"name": "Pronunciation", "qfmt": _PRON_FRONT_TMPL, "afmt": _PRON_BACK_TMPL}],
    css=_PRON_CSS,
)


def _deck_id(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") & 0x7FFF_FFFF


# ── Build .apkg ────────────────────────────────────────────────────────────────

def build_apkg(
    words_data: list[dict],
    deck_name: str,
    output_path: str,
    language: str = "",
    pronunciation_cards: bool = False,
) -> int:
    """
    Each entry in words_data:
        {word, word_translation, sentences: [{sentence, translation, audio_bytes?}]}
    pronunciation_cards=True → only pronunciation cards (no reading cards).
    Returns card count.
    """
    deck = genanki.Deck(_deck_id(deck_name), deck_name)
    tmp_dir = tempfile.mkdtemp(prefix="kotoba-anki-")
    media_paths: list[str] = []
    is_japanese = language.lower() == "japanese"

    try:
        shuffled = list(words_data)
        random.shuffle(shuffled)

        for word_info in shuffled:
            word = word_info["word"]
            word_translation = word_info["word_translation"]

            items: list[dict] = []
            audio_queue: list[str] = []  # filenames in order, for the Audios field
            for sent in word_info["sentences"]:
                audio_filename = ""
                audio_bytes: Optional[bytes] = sent.get("audio_bytes")
                if audio_bytes:
                    slug = hashlib.md5(sent["sentence"].encode()).hexdigest()[:10]
                    filename = f"kotoba_{slug}.wav"
                    tmp_path = os.path.join(tmp_dir, filename)
                    with open(tmp_path, "wb") as fh:
                        fh.write(audio_bytes)
                    media_paths.append(tmp_path)
                    audio_filename = filename

                sentence_html = (
                    furigana_highlight_html(sent["sentence"], word)
                    if is_japanese
                    else highlight_word(sent["sentence"], word)
                )
                item: dict = {
                    "s": sentence_html,
                    "t": sent["translation"],
                    "a": audio_filename,
                    "p": sent["sentence"],  # plain text for pronunciation scoring
                }
                if audio_filename:
                    item["qi"] = len(audio_queue)  # index in Anki's sound queue
                    audio_queue.append(audio_filename)
                if is_japanese:
                    item["h"] = sentence_to_hiragana(sent["sentence"])
                items.append(item)

            # Unicode-escape < so the JSON is safe inside any HTML/script context.
            # < is decoded correctly by JSON.parse(); avoids genanki HTML warnings.
            sentences_json = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
            audios_field = "".join(f"[sound:{f}]" for f in audio_queue)
            word_field = html.escape(f"{word} — {word_translation}")

            if not pronunciation_cards:
                note = genanki.Note(
                    model=SENTENCE_MODEL,
                    fields=[sentences_json, audios_field, word_field],
                    guid=genanki.guid_for(f"kotoba-export-v3:{deck_name}:{word}"),
                )
                deck.add_note(note)
            else:
                pron_note = genanki.Note(
                    model=PRONUNCIATION_MODEL,
                    fields=[sentences_json, audios_field, items[0]["p"], word_field],
                    guid=genanki.guid_for(f"kotoba-pron-v3:{deck_name}:{word}"),
                )
                deck.add_note(pron_note)

        pkg = genanki.Package(deck)
        pkg.media_files = media_paths
        pkg.write_to_file(output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return len(deck.notes)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--pronunciation-cards", action="store_true",
                        help="Generate pronunciation cards instead of reading cards")
    return parser


def _init(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[genai.Client, list[str], str, str]:
    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        parser.error("Gemini API key required: --gemini-api-key or GEMINI_API_KEY env var")

    client = genai.Client(api_key=api_key)

    if not args.no_audio:
        _load_audio_deps()

    if args.words:
        words = [w.strip() for w in args.words.split(",") if w.strip()]
    else:
        with open(args.words_file, encoding="utf-8") as fh:
            words = [line.strip() for line in fh if line.strip()]

    if not words:
        parser.error("No words found")

    deck_name = args.deck_name or (
        f"Kotoba::{args.language.capitalize()}"
        + (f"::{args.topic.capitalize()}" if args.topic else "")
    )
    output_path = args.output or f"{args.language}_anki.apkg"

    return client, words, deck_name, output_path


def _print_params(args: argparse.Namespace, words: list[str], deck_name: str, output_path: str) -> None:
    print(f"Language   : {args.language}")
    print(f"Words      : {len(words)}  ({', '.join(words[:6])}{'...' if len(words) > 6 else ''})")
    if args.proficiency:
        print(f"Proficiency: {args.proficiency.upper()}")
    if args.topic:
        print(f"Topic      : {args.topic}")
    print(f"Sentences  : {args.sentence_count} per word")
    print(f"Deck       : {deck_name}")
    print(f"Audio      : {'off' if args.no_audio else 'kokoro-onnx'}")
    print(f"Cards      : {'pronunciation' if args.pronunciation_cards else 'reading'}")
    print(f"Output     : {output_path}")
    print()


def _generate_sentences(args: argparse.Namespace, words: list[str], client: genai.Client) -> list[dict]:
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

    return words_data


def _generate_audio(words_data: list[dict], language: str) -> None:
    total = sum(len(w["sentences"]) for w in words_data)
    print(f"\n── Generating audio ({total} sentences via kokoro-onnx) ─────────")

    all_sents = [sent for word_info in words_data for sent in word_info["sentences"]]
    ok = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(generate_audio, sent["sentence"], language): sent
            for sent in all_sents
        }
        for future in as_completed(futures):
            sent = futures[future]
            try:
                sent["audio_bytes"] = future.result()
                ok += 1
                print(".", end="", flush=True)
            except Exception as exc:
                print(f"\n  Warning: audio failed for '{sent['sentence'][:40]}': {exc}",
                      file=sys.stderr)
    print(f"\n  {ok}/{total} audio files generated")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    client, words, deck_name, output_path = _init(args, parser)
    _print_params(args, words, deck_name, output_path)
    words_data = _generate_sentences(args, words, client)

    if args.no_audio:
        print("── Audio skipped ───────────────────────────────────────────────")
    else:
        _generate_audio(words_data, args.language)

    print(f"\n── Writing {output_path} ──────────────────────────────────────────")
    card_count = build_apkg(
        words_data, deck_name, output_path,
        language=args.language,
        pronunciation_cards=args.pronunciation_cards,
    )
    print(f"Done — {card_count} cards → {output_path}")


if __name__ == "__main__":
    main()
