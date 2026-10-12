from __future__ import annotations

import html
from typing import Optional

import fugashi

from romaji import kana_to_romaji

_ja_tagger: Optional[fugashi.Tagger] = None


def init_tagger() -> None:
    global _ja_tagger
    _ja_tagger = fugashi.Tagger()


def _kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ン" else c for c in s)


def _is_katakana(s: str) -> bool:
    return bool(s) and all("ァ" <= c <= "ヺ" or c == "ー" or c == "・" for c in s)


# UniDic's default readings differ from everyday speech for these.
_KANA_OVERRIDES = {
    "日本": "ニホン",      # not ニッポン
    "私": "ワタシ",        # not ワタクシ
    "本当": "ホントウ",    # UniDic drops the long vowel
}

# UniDic splits 和菓子 into 和 + 菓子 and loses the rendaku (わがし).
_COMPOUND_READINGS = {("和", "菓子"): "ガシ"}

# Written with は but pronounced わ; the particle rule can't catch these because
# UniDic tags them as a single interjection token.
_IRREGULAR_ROMAJI = {"こんにちは": "konnichiwa", "こんばんは": "konbanwa"}

_PUNCTUATION_ROMAJI = {
    "。": ".", "、": ",", "？": "?", "！": "!", "…": "...", "・": " ",
    "「": "“", "」": "”", "『": "“", "』": "”", "（": "(", "）": ")",
}
# Brackets that open a span attach to the *next* word, not the previous one.
_OPENING_BRACKETS = {"「", "『", "（", "(", "[", "［", "“", "‘"}

# Token classes that attach to the preceding word when spacing romaji.
_ATTACH_POS1 = {"助動詞", "接尾辞"}
_ATTACH_PARTICLES = {"て", "で", "ば"}
_ATTACH_AFTER_POS1 = {"動詞", "助動詞", "形容詞"}
# Copula forms (簡単 な, 上手 に, 学生 です) read better as separate words.
_STANDALONE_AUX = {"です", "でし", "でしょ", "だ", "だっ", "じゃ", "では", "で", "な", "に"}

# 十 is ジッ in UniDic (as in 十回); before these sounds it is really ジュウ (四十メートル).
_JU_GEMINATING_ONSETS = set("カキクケコサシスセソタチツテトハヒフヘホパピプペポ")


def _readings(morphemes: list) -> list[Optional[str]]:
    """Katakana reading per token, corrected for UniDic quirks. None if unknown."""
    readings: list[Optional[str]] = []
    for i, m in enumerate(morphemes):
        kana = getattr(m.feature, "kana", None)
        if kana is None or kana == "*":
            kana = m.surface if _is_katakana(m.surface) else None
        if m.surface in _KANA_OVERRIDES:
            kana = _KANA_OVERRIDES[m.surface]
        prev_surface = morphemes[i - 1].surface if i > 0 else ""
        if (prev_surface, m.surface) in _COMPOUND_READINGS:
            kana = _COMPOUND_READINGS[(prev_surface, m.surface)]
        if m.surface.endswith("十") and kana and kana.endswith("ジッ"):
            nxt = getattr(morphemes[i + 1].feature, "kana", None) if i + 1 < len(morphemes) else None
            if not nxt or nxt == "*" or nxt[0] not in _JU_GEMINATING_ONSETS:
                kana = kana[:-2] + "ジュウ"
        readings.append(kana)
    return readings


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


def furigana_html(sentence: str) -> str:
    """Return sentence as HTML with ruby furigana and no highlighting."""
    return furigana_highlight_html(sentence, "")


def furigana_highlight_html(sentence: str, word: str) -> str:
    """Return sentence as HTML with ruby furigana; the target word is highlighted."""
    tagger = _ja_tagger
    if tagger is None:
        return highlight_word(sentence, word)

    morphemes = list(tagger(sentence))
    readings = _readings(morphemes)
    surface_text = "".join(m.surface for m in morphemes)
    word_start = surface_text.find(word) if word else -1
    word_end = word_start + len(word) if word_start != -1 else -1

    parts: list[str] = []
    pos = 0
    for m, kana in zip(morphemes, readings):
        surface = m.surface
        m_end = pos + len(surface)
        in_word = word_start != -1 and pos >= word_start and m_end <= word_end

        if _has_kanji(surface) and kana and kana != surface:
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
    morphemes = list(tagger(sentence))
    return "".join(
        _kata_to_hira(kana) if kana else m.surface
        for m, kana in zip(morphemes, _readings(morphemes))
    )


def _attaches_to_previous(m, prev) -> bool:
    """Whether token `m` belongs to the same romaji word as `prev`."""
    if prev is None:
        return False
    pos1, pos2 = m.feature.pos1, m.feature.pos2
    if (prev.surface, m.surface) in _COMPOUND_READINGS:
        return True
    if pos1 == "助動詞" and m.surface in _STANDALONE_AUX:
        return False
    if pos1 in _ATTACH_POS1:
        return True
    # Only the te/de of a verb's te-form; case particle で (家で) stays separate.
    if (
        pos1 == "助詞"
        and pos2 == "接続助詞"
        and m.surface in _ATTACH_PARTICLES
        and prev.feature.pos1 in _ATTACH_AFTER_POS1
    ):
        return True
    if prev.feature.pos1 == "接頭辞":
        return True
    # Compound nouns: counters and short suffix-like nouns (年, 語, 日) stick to what
    # precedes them; longer nouns (出身, 戦略) stay separate words. Digits only join
    # other digits (十+八), so 毎日 + 一時間 stays two words.
    if pos1 == "名詞" and prev.feature.pos1 == "名詞":
        if pos2 == "数詞":
            return prev.feature.pos2 == "数詞"
        return len(m.surface) == 1 or prev.feature.pos2 == "数詞"
    return False


def _token_romaji(m, kana: Optional[str]) -> str:
    if m.surface in _IRREGULAR_ROMAJI:
        return _IRREGULAR_ROMAJI[m.surface]
    pos1 = m.feature.pos1
    if pos1 == "助詞" and m.surface in ("は", "へ"):
        return {"は": "wa", "へ": "e"}[m.surface]
    if kana is None:
        return m.surface
    return kana_to_romaji(kana, contract_long_vowels=pos1 != "動詞")


def sentence_to_romaji(sentence: str) -> str:
    """Return word-spaced Hepburn romaji for a Japanese sentence."""
    tagger = _ja_tagger
    if tagger is None:
        raise RuntimeError("init_tagger() must be called before sentence_to_romaji()")

    morphemes = list(tagger(sentence))
    words: list[str] = []
    prev = None
    join_next = False
    geminate_next = False  # previous token ended in っ (手伝っ + て → tetsudatte)
    pending_open = ""      # opening brackets waiting for the next word
    for m, kana in zip(morphemes, _readings(morphemes)):
        if m.feature.pos1 == "補助記号":
            symbol = _PUNCTUATION_ROMAJI.get(m.surface, m.surface)
            if m.surface == "・":
                prev = None
            elif m.surface in _OPENING_BRACKETS:
                pending_open += symbol
                prev = None
            elif words and symbol == "-":
                words[-1] += "-"
                join_next = True
            elif words:
                words[-1] += symbol
                prev = None
            continue

        trailing_geminate = bool(kana) and kana.endswith("ッ")
        if trailing_geminate:
            kana = kana[:-1]
        piece = _token_romaji(m, kana)
        if geminate_next and piece and piece[0].isalpha():
            piece = ("t" if piece.startswith("ch") else piece[0]) + piece
        geminate_next = trailing_geminate

        if pending_open:
            words.append(pending_open + piece)
            pending_open = ""
        elif words and (join_next or _attaches_to_previous(m, prev)):
            words[-1] += piece
        else:
            words.append(piece)
        join_next = False
        prev = m
    return " ".join(words)
