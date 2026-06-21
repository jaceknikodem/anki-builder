from __future__ import annotations

import hashlib
import html
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import genanki

from japanese import furigana_highlight_html, highlight_word, sentence_to_hiragana

_TMPL_DIR = Path(__file__).parent / "templates"


def _tmpl(name: str) -> str:
    return (_TMPL_DIR / name).read_text(encoding="utf-8")


_MODEL_CSS  = _tmpl("card.css")
_FRONT_TMPL = _tmpl("front.html")
_BACK_TMPL  = _tmpl("back.html")

# SentencesJSON field: [{s:<html>, t:<plain>, a:<filename|"">, p:<plain>, h?:<hiragana>}, ...]
# Audios field:        [sound:f1.wav][sound:f2.wav]... — kept in field values so Anki's
#                      media manager counts the files as used; NOT rendered in templates.
# JS plays the selected sentence's audio directly via the HTML5 Audio API.

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

# SentencesJSON items carry two extra fields used only by the pronunciation template:
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


def _build_sentence_items(
    word_info: dict,
    is_japanese: bool,
    tmp_dir: str,
    media_paths: list[str],
) -> tuple[list[dict], list[str]]:
    word = word_info["word"]
    items: list[dict] = []
    audio_queue: list[str] = []

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
            "p": sent["sentence"],
        }
        if audio_filename:
            item["qi"] = len(audio_queue)
            audio_queue.append(audio_filename)
        if is_japanese:
            item["h"] = sentence_to_hiragana(sent["sentence"])
        items.append(item)

    return items, audio_queue


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
            items, audio_queue = _build_sentence_items(word_info, is_japanese, tmp_dir, media_paths)

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
