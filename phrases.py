"""Parse a personal phrase list into (Japanese, English) pairs."""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBERING = re.compile(r"^\d+[.．)]\s*")
_JAPANESE_CHAR = re.compile(r"[぀-ヿ㐀-鿿]")


@dataclass(frozen=True)
class Phrase:
    japanese: str
    english: str


def _split_line(line: str) -> tuple[str, str] | None:
    """Split at `|` if present, else before the first whitespace-separated word
    with no Japanese characters (so spaces inside the Japanese are fine)."""
    if "|" in line:
        jp, _, en = line.partition("|")
    else:
        words = list(re.finditer(r"\S+", line))  # \S/\s also cover full-width spaces
        start = next(
            (w.start() for w in words[1:] if not _JAPANESE_CHAR.search(w.group())), None
        )
        if start is None:
            return None
        jp, en = line[:start], line[start:]
    jp, en = jp.strip(), en.strip()
    return (jp, en) if jp and en else None


def parse_phrases(text: str) -> list[Phrase]:
    """Parse lines of `[N. ]日本語 English` or `日本語 | English`.

    Blank lines and `#` comments are skipped; repeated Japanese lines keep the first.
    Malformed lines fail with their line number rather than being silently dropped.
    """
    phrases: list[Phrase] = []
    seen: set[str] = set()
    for number, raw in enumerate(text.splitlines(), start=1):
        line = _NUMBERING.sub("", raw.strip())
        if not line or line.startswith("#"):
            continue
        parts = _split_line(line)
        if parts is None:
            raise ValueError(f"line {number}: expected '<Japanese> <English>', got {raw!r}")
        jp, en = parts
        if not _JAPANESE_CHAR.search(jp):
            raise ValueError(f"line {number}: first field is not Japanese: {jp!r}")
        if jp in seen:
            continue
        seen.add(jp)
        phrases.append(Phrase(jp, en))
    if not phrases:
        raise ValueError("No phrases found")
    return phrases
