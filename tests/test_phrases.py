"""Tests for phrase-file parsing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phrases import Phrase, parse_phrases  # noqa: E402


def test_plain_and_numbered_lines():
    text = (
        "はじめまして。ジャックです。 Nice to meet you. I’m Jack.\n"
        "21. もう一度お願いします。 Once more, please.\n"
    )
    assert parse_phrases(text) == [
        Phrase("はじめまして。ジャックです。", "Nice to meet you. I’m Jack."),
        Phrase("もう一度お願いします。", "Once more, please."),
    ]


def test_pipe_format():
    assert parse_phrases("駅はどこですか。 | Where is the station?") == [
        Phrase("駅はどこですか。", "Where is the station?")
    ]


def test_spaces_inside_japanese_and_full_width_space():
    text = "はじめまして。 ジャックです。 Nice to meet you.\nありがとう。\u3000Thanks.\n"
    assert parse_phrases(text) == [
        Phrase("はじめまして。 ジャックです。", "Nice to meet you."),
        Phrase("ありがとう。", "Thanks."),
    ]


def test_blank_lines_and_comments_skipped():
    text = "\n# greetings\nおはようございます。 Good morning.\n\n"
    assert parse_phrases(text) == [Phrase("おはようございます。", "Good morning.")]


def test_duplicates_removed_keeping_first():
    text = "ありがとう。 Thanks.\nありがとう。 Thank you.\n"
    assert parse_phrases(text) == [Phrase("ありがとう。", "Thanks.")]


def test_line_without_english_fails_with_line_number():
    with pytest.raises(ValueError, match="line 2"):
        parse_phrases("おはよう。 Good morning.\nありがとう。\n")


def test_non_japanese_first_field_fails():
    with pytest.raises(ValueError, match="line 1"):
        parse_phrases("Good morning. おはよう。\n")


def test_empty_input_fails():
    with pytest.raises(ValueError, match="No phrases"):
        parse_phrases("\n\n")
