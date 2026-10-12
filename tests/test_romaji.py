"""Tests for kana → romaji and sentence-level romaji/ruby."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import japanese  # noqa: E402
from romaji import kana_to_romaji  # noqa: E402


@pytest.mark.parametrize("kana,expected", [
    ("スシ", "sushi"),
    ("ツキ", "tsuki"),
    ("チャ", "cha"),
    ("ジャ", "ja"),
    ("キョウ", "kyō"),
    ("トウキョウ", "tōkyō"),
    ("コーヒー", "kōhī"),
    ("メートル", "mētoru"),
    ("ガッコウ", "gakkō"),
    ("マッチャ", "matcha"),
    ("ザッシ", "zasshi"),
    ("キンエン", "kin'en"),
    ("ホンヤ", "hon'ya"),
    ("ホン", "hon"),
    ("ファイル", "fairu"),
    ("ティー", "tī"),
    ("センセイ", "sensei"),
    ("イイ", "ii"),
    ("オオキイ", "ōkii"),
])
def test_kana_to_romaji(kana, expected):
    assert kana_to_romaji(kana) == expected


def test_verbs_do_not_contract_long_vowels():
    assert kana_to_romaji("オモウ", contract_long_vowels=False) == "omou"


@pytest.fixture(scope="module", autouse=True)
def _tagger():
    japanese.init_tagger()


@pytest.mark.parametrize("sentence,expected", [
    ("どこですか。", "doko desu ka."),
    ("ポーランド出身です。", "pōrando shusshin desu."),
    ("彼は「こんにちは」と言いました。", "kare wa “konnichiwa” to iimashita."),
    ("（東京）に行きます。", "(tōkyō) ni ikimasu."),
    ("大阪に六年住んでいます。", "ōsaka ni rokunen sunde imasu."),
    ("日本語はまだ上手じゃありません。", "nihongo wa mada jōzu ja arimasen."),
    ("もう一度お願いします。", "mō ichido onegai shimasu."),
    ("今日はいい天気ですね。", "kyō wa ii tenki desu ne."),
    ("はじめまして。ジャックです。", "hajimemashite. jakku desu."),
    ("こんにちは。", "konnichiwa."),
    ("兄はトムで、妹はメアリーです。", "ani wa tomu de, imōto wa mearī desu."),
    ("Wi-Fiのパスワードは何ですか。", "Wi-Fi no pasuwādo wa nan desu ka."),
])
def test_sentence_to_romaji(sentence, expected):
    assert japanese.sentence_to_romaji(sentence) == expected


def test_nihon_is_not_nippon_in_ruby_and_romaji():
    html = japanese.furigana_html("日本語")
    assert "<rt>にほん</rt>" in html
    assert "nihon" in japanese.sentence_to_romaji("日本の包丁を持っています。")


def test_furigana_html_has_no_highlight():
    html = japanese.furigana_html("大阪に住んでいます。")
    assert 'class="kw"' not in html
    assert "<ruby>大阪<rt>おおさか</rt></ruby>" in html
