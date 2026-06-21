#!/usr/bin/env python3
"""Generate Anki flashcards from a word list. See README.md for usage."""

from __future__ import annotations

import hashlib
import html
import io
import json
import os
import random
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Optional

import fugashi
import genanki
import soundfile as sf
import typer
from google import genai
from google.genai import types as genai_types
from kokoro_onnx import Kokoro
from misaki import ja as misaki_ja
from pydantic import BaseModel
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

console = Console()
err_console = Console(stderr=True)

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
    "spanish":    ["ef_dora"],
    "french":     ["ff_siwis"],
    "italian":    ["if_sara"],
    "portuguese": ["pf_dora"],
    "chinese":    ["zf_xiaobei", "zf_xiaoni", "zm_yunxi"],
}

LANG_TO_ESPEAK: dict[str, str] = {
    "spanish":    "es",
    "french":     "fr-fr",
    "italian":    "it",
    "portuguese": "pt-br",
    "chinese":    "cmn",
}

GRAMMAR_GUIDANCE: dict[str, dict[str, str]] = json.loads(
    (Path(__file__).parent / "grammar_guidance.json").read_text(encoding="utf-8")
)


def _voice(language: str) -> str:
    return VOICES.get(language.lower(), ["af_heart"])[0]


class Proficiency(str, Enum):
    newbie = "newbie"
    a1 = "a1"
    a2 = "a2"
    b1 = "b1"


app = typer.Typer()


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
                console.print(f"    [yellow]retry {attempt + 1}/3:[/yellow] {exc} (waiting {wait}s)")
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

def _print_params(
    language: str,
    words: list[str],
    proficiency: Optional[Proficiency],
    topic: Optional[str],
    sentence_count: int,
    deck_name: str,
    output_path: str,
    no_audio: bool,
    pronunciation_cards: bool,
) -> None:
    console.print(f"[bold]Language[/bold]   : {language}")
    console.print(f"[bold]Words[/bold]      : {len(words)}  ({', '.join(words[:6])}{'...' if len(words) > 6 else ''})")
    if proficiency:
        console.print(f"[bold]Proficiency[/bold]: {proficiency.upper()}")
    if topic:
        console.print(f"[bold]Topic[/bold]      : {topic}")
    console.print(f"[bold]Sentences[/bold]  : {sentence_count} per word")
    console.print(f"[bold]Deck[/bold]       : {deck_name}")
    console.print(f"[bold]Audio[/bold]      : {'off' if no_audio else 'kokoro-onnx'}")
    console.print(f"[bold]Cards[/bold]      : {'pronunciation' if pronunciation_cards else 'reading'}")
    console.print(f"[bold]Output[/bold]     : {output_path}")
    console.print()


def _generate_sentences(
    words: list[str],
    language: str,
    proficiency: Optional[Proficiency],
    topic: Optional[str],
    sentence_count: int,
    client: genai.Client,
) -> list[dict]:
    console.rule("[bold]Generating sentences[/bold]")
    words_data: list[dict] = []

    for word in words:
        console.print(f"  {word}...", end=" ")
        try:
            data = fetch_word_data(
                word=word,
                language=language,
                proficiency=proficiency,
                topic=topic,
                sentence_count=sentence_count,
                client=client,
            )
            sentences = [
                {"sentence": s.sentence, "translation": s.translation, "audio_bytes": None}
                for s in data.sentences[:sentence_count]
            ]
            words_data.append({
                "word": word,
                "word_translation": data.word_translation,
                "sentences": sentences,
            })
            console.print(f'[green]→ "{data.word_translation}"[/green] ({len(sentences)} sentences)')
        except Exception as exc:
            err_console.print(f"[red]FAILED:[/red] {exc}")
            console.print(f"  [yellow]skipping '{word}'[/yellow]")

    if not words_data:
        err_console.print("[red]No words processed.[/red]")
        raise typer.Exit(1)

    return words_data


def _generate_audio(words_data: list[dict], language: str) -> None:
    total = sum(len(w["sentences"]) for w in words_data)
    console.rule(f"[bold]Generating audio[/bold] ({total} sentences via kokoro-onnx)")

    all_sents = [sent for word_info in words_data for sent in word_info["sentences"]]
    ok = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Synthesising...", total=total)
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
                except Exception as exc:
                    err_console.print(f"[yellow]Warning: audio failed for '{sent['sentence'][:40]}': {exc}[/yellow]")
                finally:
                    progress.advance(task)

    console.print(f"  {ok}/{total} audio files generated")


@app.command()
def main(
    language: str = typer.Option(..., help="Target language, e.g. spanish, japanese, italian"),
    words: Optional[str] = typer.Option(None, metavar="WORD,WORD,...", help="Comma-separated list of words"),
    words_file: Optional[Path] = typer.Option(None, metavar="FILE", help="One word per line"),
    proficiency: Optional[Proficiency] = typer.Option(None, help="Learner proficiency level (recommended)"),
    topic: Optional[str] = typer.Option(None, help="Topic context for sentence generation"),
    sentence_count: int = typer.Option(2, min=1, max=5, help="Sentences per word, 1–5 (default: 2)"),
    deck_name: Optional[str] = typer.Option(None, metavar="NAME", help="Anki deck name (default: Kotoba::<Language>[::<Topic>])"),
    output: Optional[str] = typer.Option(None, metavar="FILE", help="Output .apkg path (default: <language>_anki.apkg)"),
    gemini_api_key: Optional[str] = typer.Option(None, metavar="KEY", envvar="GEMINI_API_KEY", help="Gemini API key (or set GEMINI_API_KEY env var)"),
    no_audio: bool = typer.Option(False, "--no-audio/--audio", help="Skip audio generation; text-only cards"),
    pronunciation_cards: bool = typer.Option(False, "--pronunciation-cards/--reading-cards", help="Generate pronunciation cards instead of reading cards"),
) -> None:
    if not words and not words_file:
        err_console.print("[red]Provide --words or --words-file (exactly one required)[/red]")
        raise typer.Exit(1)
    if words and words_file:
        err_console.print("[red]--words and --words-file are mutually exclusive[/red]")
        raise typer.Exit(1)

    if not gemini_api_key:
        err_console.print("[red]Gemini API key required: --gemini-api-key or GEMINI_API_KEY env var[/red]")
        raise typer.Exit(1)

    client = genai.Client(api_key=gemini_api_key)

    if not no_audio:
        _load_audio_deps()

    if words:
        word_list = [w.strip() for w in words.split(",") if w.strip()]
    else:
        with open(words_file, encoding="utf-8") as fh:
            word_list = [line.strip() for line in fh if line.strip()]

    if not word_list:
        err_console.print("[red]No words found[/red]")
        raise typer.Exit(1)

    final_deck_name = deck_name or (
        f"Kotoba::{language.capitalize()}"
        + (f"::{topic.capitalize()}" if topic else "")
    )
    output_path = output or f"{language}_anki.apkg"

    _print_params(language, word_list, proficiency, topic, sentence_count, final_deck_name, output_path, no_audio, pronunciation_cards)
    words_data = _generate_sentences(word_list, language, proficiency, topic, sentence_count, client)

    if no_audio:
        console.rule("[bold]Audio skipped[/bold]")
    else:
        _generate_audio(words_data, language)

    console.rule(f"[bold]Writing {output_path}[/bold]")
    card_count = build_apkg(
        words_data, final_deck_name, output_path,
        language=language,
        pronunciation_cards=pronunciation_cards,
    )
    console.print(f"\n[green bold]Done[/green bold] — {card_count} cards → {output_path}")


if __name__ == "__main__":
    app()
