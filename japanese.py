from __future__ import annotations

import html
from typing import Optional

import fugashi

_ja_tagger: Optional[fugashi.Tagger] = None


def init_tagger() -> None:
    global _ja_tagger
    _ja_tagger = fugashi.Tagger()


def _kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ン" else c for c in s)


def _has_kanji(s: str) -> bool:
    return any("一" <= c <= "鿿" or "㐀" <= c <= "䶿" for c in s)


def highlight_word(sentence: str, word: str) -> str:
    """Return HTML-escaped sentence with the first match of `word` highlighted."""
    escaped = html.escape(sentence)
    escaped_word = html.escape(word)
    if escaped_word and escaped_word in escaped:
        escaped = escaped.replace(
            escaped_word, f'<span class="kw">{escaped_word}</span>', 1
        )
    return escaped


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
