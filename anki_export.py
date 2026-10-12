#!/usr/bin/env python3
"""Generate Anki flashcards from a word list. See README.md for usage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from google import genai
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

import audio
import japanese
from cards import build_apkg, build_phrase_apkg
from gemini import fetch_word_data
from phrases import parse_phrases

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


class Proficiency(str, Enum):
    newbie = "newbie"
    a1 = "a1"
    a2 = "a2"
    b1 = "b1"


class Tts(str, Enum):
    kokoro = "kokoro"
    elevenlabs = "elevenlabs"


def _load_audio_deps(tts: Tts, elevenlabs_api_key: Optional[str]) -> None:
    """Initialise the chosen TTS backend and the Japanese tagger (used for ruby/romaji)."""
    if tts is Tts.elevenlabs:
        if not elevenlabs_api_key:
            err_console.print("[red]ElevenLabs API key required: --elevenlabs-api-key or ELEVENLABS_API_KEY env var[/red]")
            raise typer.Exit(1)
        audio.init_audio("elevenlabs", api_key=elevenlabs_api_key)
    else:
        audio.init_audio("kokoro")
    japanese.init_tagger()


def _print_params(
    language: str,
    words: list[str],
    proficiency: Optional[Proficiency],
    topic: Optional[str],
    sentence_count: int,
    deck_name: str,
    model: str,
    output_path: str,
    no_audio: bool,
    tts: Tts,
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
    console.print(f"[bold]Model[/bold]      : {model}")
    console.print(f"[bold]Audio[/bold]      : {'off' if no_audio else tts.value}")
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
    model: str,
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
                model=model,
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


def _generate_audio(words_data: list[dict], language: str, tts: Tts) -> None:
    _synthesize([sent for word_info in words_data for sent in word_info["sentences"]], language, tts)


def _synthesize(sentences: list[dict], language: str, tts: Tts) -> None:
    """Fill `audio_bytes` on each {sentence, ...} dict; failures leave it None."""
    total = len(sentences)
    console.rule(f"[bold]Generating audio[/bold] ({total} sentences via {tts.value})")

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
                pool.submit(audio.generate_audio, sent["sentence"], language): sent
                for sent in sentences
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


def _run_phrases(
    phrases_file: Path,
    language: str,
    deck_name: Optional[str],
    output: Optional[str],
    no_audio: bool,
    tts: Tts,
    elevenlabs_api_key: Optional[str],
) -> None:
    """Build phrase cards from a personal sentence list. No LLM involved."""
    if language.lower() != "japanese":
        err_console.print("[red]--phrases-file only supports --language japanese (ruby and romaji)[/red]")
        raise typer.Exit(1)

    try:
        parsed = parse_phrases(phrases_file.read_text(encoding="utf-8"))
    except ValueError as exc:
        err_console.print(f"[red]{phrases_file}: {exc}[/red]")
        raise typer.Exit(1)

    final_deck_name = deck_name or "Kotoba::Japanese::Phrases"
    output_path = output or "japanese_phrases.apkg"
    console.print(f"[bold]Phrases[/bold]    : {len(parsed)}  ({phrases_file})")
    console.print(f"[bold]Deck[/bold]       : {final_deck_name}")
    console.print(f"[bold]Audio[/bold]      : {'off' if no_audio else tts.value}")
    console.print(f"[bold]Output[/bold]     : {output_path}\n")

    sentences = [
        {"sentence": p.japanese, "translation": p.english, "audio_bytes": None}
        for p in parsed
    ]
    if no_audio:
        japanese.init_tagger()  # ruby and romaji still need it
        console.rule("[bold]Audio skipped[/bold]")
    else:
        _load_audio_deps(tts, elevenlabs_api_key)
        _synthesize(sentences, language, tts)

    console.rule(f"[bold]Writing {output_path}[/bold]")
    card_count = build_phrase_apkg(sentences, final_deck_name, output_path)
    console.print(f"\n[green bold]Done[/green bold] — {card_count} cards → {output_path}")


@app.command()
def main(
    language: str = typer.Option(..., help="Target language, e.g. spanish, japanese, italian"),
    words: Optional[str] = typer.Option(None, metavar="WORD,WORD,...", help="Comma-separated list of words"),
    words_file: Optional[Path] = typer.Option(None, metavar="FILE", help="One word per line"),
    phrases_file: Optional[Path] = typer.Option(None, metavar="FILE", help="Personal phrase list ('日本語 English' per line); builds phrase cards, no Gemini needed"),
    proficiency: Optional[Proficiency] = typer.Option(None, help="Learner proficiency level (recommended)"),
    topic: Optional[str] = typer.Option(None, help="Topic context for sentence generation"),
    sentence_count: int = typer.Option(2, min=1, max=5, help="Sentences per word, 1–5 (default: 2)"),
    deck_name: Optional[str] = typer.Option(None, metavar="NAME", help="Anki deck name (default: Kotoba::<Language>[::<Topic>])"),
    output: Optional[str] = typer.Option(None, metavar="FILE", help="Output .apkg path (default: <language>_anki.apkg)"),
    gemini_api_key: Optional[str] = typer.Option(None, metavar="KEY", envvar="GEMINI_API_KEY", help="Gemini API key (or set GEMINI_API_KEY env var)"),
    gemini_model: str = typer.Option("gemini-3.1-flash-lite", "--model", metavar="MODEL", help="Gemini model to use for sentence generation"),
    no_audio: bool = typer.Option(False, "--no-audio/--audio", help="Skip audio generation; text-only cards"),
    tts: Tts = typer.Option(Tts.kokoro, "--tts", help="TTS backend: kokoro (local) or elevenlabs (cloud)"),
    elevenlabs_api_key: Optional[str] = typer.Option(None, metavar="KEY", envvar="ELEVENLABS_API_KEY", help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)"),
    pronunciation_cards: bool = typer.Option(False, "--pronunciation-cards/--reading-cards", help="Generate pronunciation cards instead of reading cards"),
) -> None:
    if phrases_file:
        if words or words_file:
            err_console.print("[red]--phrases-file can't be combined with --words/--words-file[/red]")
            raise typer.Exit(1)
        _run_phrases(phrases_file, language, deck_name, output, no_audio, tts, elevenlabs_api_key)
        return

    if not words and not words_file:
        err_console.print("[red]Provide --words, --words-file or --phrases-file[/red]")
        raise typer.Exit(1)
    if words and words_file:
        err_console.print("[red]--words and --words-file are mutually exclusive[/red]")
        raise typer.Exit(1)

    if not gemini_api_key:
        err_console.print("[red]Gemini API key required: --gemini-api-key or GEMINI_API_KEY env var[/red]")
        raise typer.Exit(1)

    client = genai.Client(api_key=gemini_api_key)

    if not no_audio:
        _load_audio_deps(tts, elevenlabs_api_key)

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

    _print_params(language, word_list, proficiency, topic, sentence_count, final_deck_name, gemini_model, output_path, no_audio, tts, pronunciation_cards)
    words_data = _generate_sentences(word_list, language, proficiency, topic, sentence_count, client, gemini_model)

    if no_audio:
        console.rule("[bold]Audio skipped[/bold]")
    else:
        _generate_audio(words_data, language, tts)

    console.rule(f"[bold]Writing {output_path}[/bold]")
    card_count = build_apkg(
        words_data, final_deck_name, output_path,
        language=language,
        pronunciation_cards=pronunciation_cards,
    )
    console.print(f"\n[green bold]Done[/green bold] — {card_count} cards → {output_path}")


if __name__ == "__main__":
    app()
