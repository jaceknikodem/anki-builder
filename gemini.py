from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel
from rich.console import Console

console = Console()

GRAMMAR_GUIDANCE: dict[str, dict[str, str]] = json.loads(
    (Path(__file__).parent / "grammar_guidance.json").read_text(encoding="utf-8")
)


class SentenceItem(BaseModel):
    sentence: str
    translation: str


class WordData(BaseModel):
    word_translation: str
    sentences: list[SentenceItem]


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
    model: str = "gemini-3.1-flash-lite",
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
                model=model,
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
