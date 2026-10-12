"""Tests for the phrase note type and .apkg build."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import japanese  # noqa: E402
from cards import PHRASE_MODEL, build_phrase_apkg  # noqa: E402


def _phrase(jp: str, en: str, audio: bytes | None = None) -> dict:
    return {"sentence": jp, "translation": en, "audio_bytes": audio}


def test_model_shape():
    assert [f["name"] for f in PHRASE_MODEL.fields] == ["Japanese", "Romaji", "English", "Audio"]
    (template,) = PHRASE_MODEL.templates
    assert "{{English}}" in template["qfmt"]
    assert "{{Audio}}" not in template["qfmt"]  # front must not give the answer away
    for field in ("Japanese", "Romaji", "Audio"):
        assert "{{%s}}" % field in template["afmt"]


def test_build_writes_notes_and_media(tmp_path):
    japanese.init_tagger()
    out = tmp_path / "phrases.apkg"
    count = build_phrase_apkg(
        [
            _phrase("どこですか。", "Where is it?", b"RIFFfake"),
            _phrase("大阪に住んでいます。", "I live in Osaka.", None),
        ],
        "Kotoba::Japanese::Phrases",
        str(out),
    )
    assert count == 2
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "collection.anki2" in names
        assert "media" in names
        media_manifest = z.read("media").decode()
    assert media_manifest.count(".wav") == 1  # only the phrase that has audio


def test_note_guid_is_stable_across_builds(tmp_path):
    japanese.init_tagger()
    phrases = [_phrase("どこですか。", "Where is it?")]
    a, b = tmp_path / "a.apkg", tmp_path / "b.apkg"
    build_phrase_apkg(phrases, "Deck", str(a))
    build_phrase_apkg(phrases, "Deck", str(b))

    def guids(path: Path) -> list[str]:
        import sqlite3
        import tempfile
        with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as d:
            z.extract("collection.anki2", d)
            con = sqlite3.connect(Path(d) / "collection.anki2")
            try:
                return [r[0] for r in con.execute("select guid from notes")]
            finally:
                con.close()

    assert guids(a) == guids(b) != []
