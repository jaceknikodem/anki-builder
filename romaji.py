"""Katakana/hiragana → Hepburn romaji (macrons for long vowels)."""
from __future__ import annotations

_MONOGRAPHS = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヲ": "o",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "ヴ": "vu",
}

# Consonant stems for i-row kana + small ya/yu/yo (キャ → kya, シャ → sha, ...).
_PALATAL_STEMS = {
    "キ": "ky", "ギ": "gy", "シ": "sh", "ジ": "j", "ヂ": "j", "チ": "ch",
    "ニ": "ny", "ヒ": "hy", "ビ": "by", "ピ": "py", "ミ": "my", "リ": "ry",
}
_SMALL_Y = {"ャ": "a", "ュ": "u", "ョ": "o"}

# Two-kana sequences used mostly for loanwords.
_LOAN_DIGRAPHS = {
    "ファ": "fa", "フィ": "fi", "フェ": "fe", "フォ": "fo",
    "ウィ": "wi", "ウェ": "we", "ウォ": "wo", "イェ": "ye",
    "ヴァ": "va", "ヴィ": "vi", "ヴェ": "ve", "ヴォ": "vo",
    "ティ": "ti", "ディ": "di", "トゥ": "tu", "ドゥ": "du",
    "テュ": "tyu", "デュ": "dyu",
    "シェ": "she", "ジェ": "je", "チェ": "che",
    "ツァ": "tsa", "ツィ": "tsi", "ツェ": "tse", "ツォ": "tso",
}

_MACRONS = {"a": "ā", "i": "ī", "u": "ū", "e": "ē", "o": "ō"}
_VOWEL_KANA = set("アイウエオヤユヨ")
_CONTRACTIONS = (("ou", "ō"), ("oo", "ō"), ("uu", "ū"), ("aa", "ā"))


def _to_katakana(text: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def _mora_at(kana: str, i: int) -> tuple[str, int] | None:
    """Return (romaji, kana_length) for the mora starting at i, or None."""
    pair = kana[i : i + 2]
    if pair in _LOAN_DIGRAPHS:
        return _LOAN_DIGRAPHS[pair], 2
    if len(pair) == 2 and pair[0] in _PALATAL_STEMS and pair[1] in _SMALL_Y:
        return _PALATAL_STEMS[pair[0]] + _SMALL_Y[pair[1]], 2
    if kana[i] in _MONOGRAPHS:
        return _MONOGRAPHS[kana[i]], 1
    return None


def kana_to_romaji(kana: str, contract_long_vowels: bool = True) -> str:
    """Convert kana to Hepburn romaji.

    Katakana `ー` always becomes a macron. With `contract_long_vowels`, orthographic
    おう/おお/うう/ああ also become ō/ū/ā (so 東京 → tōkyō) while えい/いい stay
    `ei`/`ii`. Turn it off for verbs, where 思う is `omou`, not `omō`.
    """
    kana = _to_katakana(kana)
    out = ""
    geminate = False
    i = 0
    while i < len(kana):
        c = kana[i]
        if c == "ッ":
            geminate = True
            i += 1
            continue
        if c == "ー":
            if out and out[-1] in _MACRONS:
                out = out[:-1] + _MACRONS[out[-1]]
            i += 1
            continue
        if c == "ン":
            nxt = kana[i + 1] if i + 1 < len(kana) else ""
            out += "n'" if nxt in _VOWEL_KANA else "n"
            i += 1
            continue
        mora = _mora_at(kana, i)
        if mora is None:
            out += c
            i += 1
            continue
        romaji, length = mora
        if geminate:
            romaji = ("t" if romaji.startswith("ch") else romaji[0]) + romaji
            geminate = False
        out += romaji
        i += length

    if contract_long_vowels:
        for plain, long_form in _CONTRACTIONS:
            out = out.replace(plain, long_form)
    return out
