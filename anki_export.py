#!/usr/bin/env python3
"""
anki_export.py — Generate Anki flashcards from a word list.

Calls Gemini for 2-3 natural sentences (+ translations) per word, optionally
generates audio via kokoro-onnx locally, and writes a ready-to-import .apkg.

Front: sentence with the target word highlighted  + audio
Back:  English translation of the sentence  +  word meaning

Setup:
    uv pip install -r requirements.txt     # Gemini + Anki
    bash setup_kokoro.sh                   # audio (optional)

Usage:
    python anki_export.py \\
        --words "comprare,vendere,mangiare" \\
        --language italian --proficiency a2 --topic food

    python anki_export.py \\
        --words-file words.txt \\
        --language japanese --proficiency b1 \\
        --deck-name "Japanese B1 — verbs"

    python anki_export.py \\
        --words "hola,gracias" --language spanish \\
        --proficiency newbie --no-audio
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import fugashi
import genanki
import soundfile as sf
from google import genai
from google.genai import types as genai_types
from kokoro_onnx import Kokoro
from misaki import ja as misaki_ja
from pydantic import BaseModel

_ja_g2p = None       # misaki Japanese G2P instance
_ja_g2p_lock = threading.Lock()  # misaki is not documented thread-safe
_ja_tagger = None    # fugashi MeCab tagger (Japanese furigana)

CACHE_DIR  = Path.home() / ".cache" / "kotoba-ai"
MODEL_PATH = CACHE_DIR / "kokoro-v1.0.int8.onnx"
VOICES_PATH = CACHE_DIR / "voices-v1.0.bin"

_kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))


def _load_audio_deps() -> None:
    """Initialize Japanese language processing deps."""
    global _ja_g2p, _ja_tagger
    _ja_g2p = misaki_ja.JAG2P()
    _ja_tagger = fugashi.Tagger()


# ── language / voice config ────────────────────────────────────────────────────

VOICES: dict[str, list[str]] = {
    "japanese":   ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "english":    ["af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael"],
    "spanish":    ["ef_dora"],
    "french":     ["ff_siwis"],
    "italian":    ["if_sara"],
    "portuguese": ["pf_dora"],
    "chinese":    ["zf_xiaobei", "zf_xiaoni", "zm_yunxi"],
    "korean":     ["kf_aria", "km_junho"],
}

LANG_TO_ESPEAK: dict[str, str] = {
    "english":    "en-us",
    "spanish":    "es",
    "french":     "fr-fr",
    "italian":    "it",
    "portuguese": "pt-br",
    "chinese":    "cmn",
    "korean":     "ko",
    "polish":     "pl",
    "indonesian": "id",
}

GRAMMAR_GUIDANCE: dict[str, dict[str, str]] = {
    "italian": {
        "newbie": (
            "USE: presente of essere/avere/stare + regular -are/-ere verbs in 1st–3rd sg only; "
            "fixed modal chunks (posso/devo/voglio/ho bisogno di + infinitive). "
            "AVOID: passato prossimo, imperfetto, futuro, object pronouns, subordinate clauses, "
            "irregular verbs beyond essere/avere/stare/fare. "
            "STRUCTURE: bare S-V or S-V-O only; concrete high-frequency vocabulary; ≤8 words per sentence."
        ),
        "a1": (
            "USE: presente all 6 persons including key irregulars (fare/andare/dire/venire/sapere/uscire); "
            "modal + infinitive productively; tu/Lei imperative; basic reflexive verbs (mi chiamo/mi sveglio/mi alzo); "
            "definite and indefinite articles; negation non; connectors e/ma/però/perché/quando. "
            "AVOID: passato prossimo in productive output (recognition only), clitics, congiuntivo, "
            "relative clauses with cui. "
            "STRUCTURE: simple S-V-O with one optional adjunct; ≤12 words per sentence."
        ),
        "a2": (
            "USE: passato prossimo productively (essere vs. avere auxiliary, past participle agreement); "
            "stare + gerundio (sto leggendo); reflexive verbs fully; ne/ci as locative/partitive; "
            "direct object pronouns (lo/la/li/le); adverbs of frequency (sempre/mai/spesso/ancora/già); "
            "relative clauses with che. "
            "AVOID: congiuntivo, combined clitics (glielo), passive voice, imperfetto in productive output. "
            "STRUCTURE: may include one simple relative clause or temporal clause; 8–14 words per sentence."
        ),
        "b1": (
            "USE: imperfetto productively for habitual past and background description; "
            "futuro semplice; condizionale presente (vorrei/potrei/dovrei/sarei); "
            "congiuntivo presente in formulaic triggers (penso/credo/spero/voglio che + subj); "
            "direct + indirect clitics productively; si passivante; "
            "connectors (quindi/però/tuttavia/nonostante/sebbene). "
            "AVOID: congiuntivo imperfetto, congiuntivo trapassato, periodo ipotetico III type. "
            "STRUCTURE: one subordinate clause allowed; idiomatic expressions welcome; 10–18 words per sentence."
        ),
    },
    "spanish": {
        "newbie": (
            "USE: presente of ser/estar/tener + regular -ar/-er/-ir verbs (yo/tú/él forms only); "
            "gender-agreeing adjectives; negation no; basic interrogatives (¿qué?/¿dónde?/¿cómo?); "
            "fixed phrases (me llamo…/tengo…años). "
            "AVOID: preterite, reflexives, object pronouns, stem-changing verbs, ser vs. estar nuance. "
            "STRUCTURE: S-V or S-V-O only; concrete everyday vocabulary; ≤8 words per sentence."
        ),
        "a1": (
            "USE: presente all 6 persons including stem-changers (e→ie/o→ue/e→i) and irregulars "
            "(ser/estar/ir/tener/venir/hacer/saber/poder); reflexive verbs (levantarse/llamarse); "
            "modal periphrasis (querer/poder/deber + inf); periphrastic future (ir a + inf); "
            "direct object pronouns (lo/la/los/las); basic connectors (y/pero/porque/cuando). "
            "AVOID: preterite, imperfecto, subjunctive, indirect object clitic clusters. "
            "STRUCTURE: simple S-V-O with one optional adverbial; ≤12 words per sentence."
        ),
        "a2": (
            "USE: pretérito perfecto compuesto (he/has/ha… + participio) for recent actions; "
            "pretérito indefinido for completed past events; estar + gerundio productively; "
            "gustar-type verbs (encantar/doler/parecer/interesar) with indirect object pronouns; "
            "basic condicional (me gustaría/podría/debería); demonstratives; comparatives (más…que/tan…como); "
            "por vs. para in high-frequency patterns. "
            "AVOID: present subjunctive, imperfecto, pluscuamperfecto. "
            "STRUCTURE: may include one relative clause (que) or causal clause; 8–14 words per sentence."
        ),
        "b1": (
            "USE: present subjunctive in common triggers (querer que/es importante que/cuando+future action); "
            "pretérito imperfecto vs. indefinido contrast; pluscuamperfecto for recognition; "
            "complex connectors (aunque/sin embargo/a pesar de/por lo tanto/mientras que); "
            "ser vs. estar nuanced distinctions; long-range clitic placement; "
            "idiomatic expressions and colloquial register where natural. "
            "AVOID: imperfect subjunctive, conditional perfect, vosotros unless targeting Spain specifically. "
            "STRUCTURE: one subordinate clause; 10–18 words per sentence."
        ),
    },
    "portuguese": {
        "newbie": (
            "USE: presente of ser/estar/ter/ficar + regular -ar/-er/-ir verbs (eu/você/ele forms); "
            "gender-agreeing adjectives; negation não; basic interrogatives (o que/onde/como); "
            "fixed phrases (me chamo…/tenho…anos). "
            "AVOID: preterite, reflexives, object pronouns, irregular verbs beyond ser/estar/ter. "
            "STRUCTURE: S-V or S-V-O; concrete vocabulary; ≤8 words per sentence. "
            "Note: default to Brazilian Portuguese (você, a gente) unless European specified."
        ),
        "a1": (
            "USE: presente all persons including irregulars (ir/vir/fazer/saber/poder/querer); "
            "poder/precisar/querer + infinitive; reflexive pronouns (me/se); "
            "definite and indefinite articles; negation não; connectors e/mas/porque/quando; "
            "pretérito perfeito for recognition only (vi/fui/fiz). "
            "AVOID: pretérito perfeito in productive output, imperfeito, futuro, object clitic clusters. "
            "STRUCTURE: S-V-O with one optional adjunct; ≤12 words per sentence."
        ),
        "a2": (
            "USE: pretérito perfeito productively (regular and key irregular: fui/vi/fiz/vim/trouxe); "
            "estar + gerúndio (estou comendo — Brazilian) or a + infinitive (European); "
            "direct object pronouns; condicional de cortesia (gostaria/poderia/precisaria); "
            "quantifiers and comparatives; relative clauses with que. "
            "AVOID: pretérito imperfeito in productive output, subjunctive, personal infinitive. "
            "STRUCTURE: may include one relative or causal clause; 8–14 words per sentence."
        ),
        "b1": (
            "USE: pretérito imperfeito productively (habitual/background past, descriptions); "
            "futuro do presente (falarei/irá — or ir a + inf colloquially); "
            "full condicional; presente do subjuntivo in formulaic triggers (quero que/é importante que); "
            "personal infinitive; complex connectors (embora/apesar de/portanto/enquanto). "
            "AVOID: futuro do subjuntivo (except in set phrases), imperfeito do subjuntivo. "
            "STRUCTURE: one subordinate clause allowed; idiomatic expressions welcome; 10–18 words per sentence."
        ),
    },
    "japanese": {
        "newbie": (
            "USE: copula (です/ではありません/じゃありません); present-tense polite verb forms (~ます/~ません) "
            "for common verbs (たべます/のみます/いきます/きます/します); "
            "particles は・が・を・に・で・も; demonstratives (これ/それ/あれ/ここ/そこ). "
            "AVOID: て-form, た-form, plain/dictionary form, conditionals, relative clauses, keigo. "
            "STRUCTURE: topic-comment or S-O-V; short, concrete, single-clause sentences; ≤10 words. "
            "Write in kana + common kanji (JLPT N5 kanji only); add furigana on unfamiliar kanji."
        ),
        "a1": (
            "USE: polite past (~ました/~ませんでした); て-form for sequential actions (〜てから) and requests (〜てください); "
            "あります/います (existence/possession distinction); "
            "〜たいです (want to); 〜ましょう/〜ましょうか (let's/shall we); "
            "time expressions and frequency adverbs (毎日/よく/たまに). "
            "AVOID: plain form in productive output, potential form, conditional, passive. "
            "STRUCTURE: up to two linked clauses via て-form; 10–14 words. JLPT N5–N4 kanji."
        ),
        "a2": (
            "USE: て-form compounds (〜ている progressive/state; 〜てみる; 〜てしまう); "
            "plain/dictionary form productively; potential form (~られる/~できる); "
            "conditionals (〜たら for sequenced/hypothetical); "
            "noun modification with plain-form relative clauses; "
            "のです/んです for explanation; ために/ように purpose clauses. "
            "AVOID: passive, causative, keigo beyond ます/です, て-form causative. "
            "STRUCTURE: one embedded clause; 12–18 words. JLPT N4–N3 kanji."
        ),
        "b1": (
            "USE: full plain-form conjugation across tenses; passive (〜られる) and causative (〜させる); "
            "keigo basics (〜ていただく/〜てさしあげる/〜ていただけますか); "
            "extended predicate nominalizers (こと/の); conjunctions (〜ので/〜のに/〜ても/〜ながら); "
            "indirect speech (〜と言っていた/〜と思う); "
            "colloquial contractions (〜ている→〜てる; 〜てしまう→〜ちゃう). "
            "AVOID: literary/archaic forms, overly complex keigo chains. "
            "STRUCTURE: up to two embedded clauses; 14–22 words. JLPT N3–N2 kanji with furigana on N2."
        ),
    },
    "polish": {
        "newbie": (
            "USE: present tense być/mieć/chcieć + regular verbs (1st/2nd/3rd sg); "
            "nominative and accusative cases for common masculine/feminine/neuter nouns; "
            "basic adjective agreement (nominative only); negation nie; "
            "simple S-V-O; common adverbs (tu/tam/teraz/bardzo). "
            "AVOID: past tense, genitive, instrumental, dative, reflexives, aspect distinction. "
            "STRUCTURE: S-V or S-V-O; concrete everyday vocabulary; ≤8 words per sentence."
        ),
        "a1": (
            "USE: present tense all persons for all verb classes; "
            "past tense for recognition (był/była/było/byli); "
            "modals (mogę/chcę/muszę/powinienem + inf); "
            "accusative case productively; basic genitive (negation/possession); "
            "question words (kto/co/gdzie/kiedy/jak/dlaczego); connectors i/ale/bo/że/kiedy. "
            "AVOID: instrumental/locative/dative in productive output, perfective aspect distinction, "
            "conditional, passive. "
            "STRUCTURE: S-V-O with one optional adverbial; ≤12 words per sentence."
        ),
        "a2": (
            "USE: past tense productively all genders/persons; "
            "future with będę + infinitive (imperfective); "
            "accusative and genitive in predictable patterns (negation, quantity, prepositions: do/z/bez/od/dla); "
            "dative with common verbs (dać/powiedzieć/pomóc); "
            "reflexive się productively; locative with w/na for place; "
            "aspect pairs in high-frequency verbs. "
            "AVOID: instrumental beyond fixed phrases, complex subordinate clauses, conditional. "
            "STRUCTURE: may include one subordinate clause (że/kiedy/żeby); 8–14 words per sentence."
        ),
        "b1": (
            "USE: perfective/imperfective aspect contrast productively across tenses; "
            "conditional (chciałbym/mogłabym); "
            "instrumental productively (z kimś/być kimś/czymś); "
            "genitive plural; verbal nouns; subordinate clauses (żeby/chociaż/mimo że/dopóki); "
            "passive constructions with być + past participle. "
            "AVOID: archaic or highly formal constructions, rare case collocations. "
            "STRUCTURE: one complex subordinate clause; idiomatic expressions welcome; 10–18 words per sentence."
        ),
    },
    "french": {
        "newbie": (
            "USE: présent of être/avoir/aller + regular -er verbs (je/tu/il forms); "
            "basic articles (le/la/les/un/une/des); negation ne…pas; "
            "basic interrogatives (qu'est-ce que/où/comment/qui); "
            "fixed expressions (je m'appelle/j'ai…ans/il y a). "
            "AVOID: passé composé, imparfait, reflexives, object pronouns, liaisons explained explicitly. "
            "STRUCTURE: S-V or S-V-O; concrete high-frequency vocabulary; ≤8 words per sentence."
        ),
        "a1": (
            "USE: présent all persons including key irregulars (faire/pouvoir/vouloir/devoir/venir/savoir/prendre); "
            "modal + infinitive productively; reflexive verbs (se lever/se coucher/s'appeler); "
            "passé composé with avoir/être for recognition; "
            "basic adjective agreement (position and gender); "
            "connectors et/mais/parce que/quand/alors. "
            "AVOID: imparfait, futur simple, subjunctive, object pronouns in productive output. "
            "STRUCTURE: S-V-O with one optional adverbial; ≤12 words per sentence."
        ),
        "a2": (
            "USE: passé composé productively (avoir vs être auxiliary, past participle agreement with être verbs); "
            "imparfait for recognition/comprehension framing; futur proche (aller + inf) productively; "
            "direct object pronouns (le/la/les/me/te) in correct pre-verb position; "
            "comparatives (plus/moins/aussi…que); "
            "depuis + présent for ongoing states; relative clauses with qui/que. "
            "AVOID: futur simple, conditionnel, subjunctive, combined pronoun clusters. "
            "STRUCTURE: may include one relative clause or causal clause; 8–14 words per sentence."
        ),
        "b1": (
            "USE: imparfait vs passé composé contrast productively; "
            "futur simple productively; conditionnel présent (je voudrais/je pourrais/il faudrait); "
            "subjonctif présent in formulaic triggers (il faut que/je veux que/bien que + subj); "
            "indirect object and adverbial pronouns (y/en) productively; "
            "plus-que-parfait for recognition; connectors (cependant/néanmoins/pourtant/donc/afin de). "
            "AVOID: subjonctif passé, conditionnel passé, passive with faire causatif. "
            "STRUCTURE: one subordinate clause; idiomatic register welcome; 10–18 words per sentence."
        ),
    },
    "indonesian": {
        "newbie": (
            "USE: base (bare) verb forms; simple S-V-O; "
            "common pronouns (saya/aku/kamu/dia/kami/kita/mereka); "
            "stative adjectives as predicates; negation tidak (verbs/adjectives) vs bukan (nouns); "
            "topic-fronting; common nouns without affixes. "
            "AVOID: me-/ber-/di- affixes, reduplication, formal register, aspect markers. "
            "STRUCTURE: bare S-V or S-V-O; concrete everyday vocabulary; ≤8 words per sentence."
        ),
        "a1": (
            "USE: me- prefix verbs (makan/minum/pergi base; membeli/membaca/menulis affixed); "
            "ber- verbs (berbicara/bekerja/berjalan); "
            "question words (apa/siapa/di mana/kapan/bagaimana/mengapa/berapa); "
            "possessive suffix -nya; simple reduplication (noun plurals: buku-buku); "
            "connectors dan/atau/tapi/karena/kalau. "
            "AVOID: di- passive, ke-…-an/-an nominalizations, complex affix stacking. "
            "STRUCTURE: S-V-O with one optional adjunct; ≤12 words per sentence."
        ),
        "a2": (
            "USE: di- passive productively (buku itu dibeli oleh dia); "
            "modals (bisa/harus/mau/boleh/perlu + base verb); "
            "aspect markers (sedang for progressive; sudah for completion; belum for not-yet; akan for future); "
            "me-…-kan and me-…-i causative/directional verbs; "
            "ke-…-an and pe-…-an nominalizations in common words; "
            "relative clauses with yang. "
            "AVOID: complex affix stacking (memper-…-kan), formal register vocabulary. "
            "STRUCTURE: may include one relative clause (yang) or conditional; 8–14 words per sentence."
        ),
        "b1": (
            "USE: varied affix combinations (memper-…-kan; ke-…-an; pe-N-…-an) in productive output; "
            "full passive system (di- and ter- for accidental/involuntary); "
            "complex subordinating conjunctions (walaupun/meskipun/sehingga/agar/supaya/padahal/setelah); "
            "formal vs. colloquial register contrast (saya vs. aku; tidak vs. nggak; "
            "mengapa vs. kenapa); "
            "idiomatic expressions and proverbs. "
            "AVOID: archaic or highly literary forms (adapun/barang siapa). "
            "STRUCTURE: one complex subordinate clause; 10–18 words per sentence."
        ),
    },
}


def _voice(language: str) -> str:
    return VOICES.get(language.lower(), ["af_heart"])[0]


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class SentenceItem(BaseModel):
    sentence: str
    translation: str


class WordData(BaseModel):
    word_translation: str
    sentences: list[SentenceItem]


# ── Gemini ─────────────────────────────────────────────────────────────────────

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
                model="gemini-2.5-flash",
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
                print(f"    retry {attempt + 1}/3: {exc} (waiting {wait}s)")
                time.sleep(wait)
    raise last_err


# ── Kokoro TTS (direct, no HTTP) ───────────────────────────────────────────────

def generate_audio(text: str, language: str) -> bytes:
    """Synthesise WAV audio and return raw bytes."""
    voice = _voice(language)
    lang = language.lower()

    if lang == "japanese" and _ja_g2p is not None:
        with _ja_g2p_lock:
            ipa, _ = _ja_g2p(text)
        audio, sr = _kokoro.create(ipa, voice=voice, is_phonemes=True)
    else:
        espeak_lang = LANG_TO_ESPEAK.get(lang, "en-us")
        audio, sr = _kokoro.create(text, voice=voice, lang=espeak_lang)

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


# ── Highlighting ───────────────────────────────────────────────────────────────

def highlight_word(sentence: str, word: str) -> str:
    """Return HTML-escaped sentence with the first match of `word` highlighted."""
    escaped = html.escape(sentence)
    escaped_word = html.escape(word)
    if escaped_word and escaped_word in escaped:
        escaped = escaped.replace(
            escaped_word, f'<span class="kw">{escaped_word}</span>', 1
        )
    return escaped


# ── Furigana (Japanese only) ───────────────────────────────────────────────────

def _kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ン" else c for c in s)


def _has_kanji(s: str) -> bool:
    return any("一" <= c <= "鿿" or "㐀" <= c <= "䶿" for c in s)


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


# ── Anki model ─────────────────────────────────────────────────────────────────

_MODEL_CSS = """\
.card {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 20px;
  text-align: center;
  color: #1c1c1e;
  background: #ffffff;
}
.sentence { font-size: 26px; margin: 16px 0; line-height: 1.6; }
.sentence .kw { color: #2563eb; font-weight: 700; }
.translation { margin-top: 12px; color: #444; }
.word { color: #888; font-size: 16px; margin-top: 14px; font-style: italic; }
hr#answer { margin: 18px 0; border: none; border-top: 1px solid #e0e0e0; }
ruby { ruby-align: center; }
rt { font-size: 0.5em; color: #555; }
.kotoba-replay {
  background: none; border: 2px solid #2563eb; border-radius: 50%;
  color: #2563eb; cursor: pointer; font-size: 22px;
  width: 52px; height: 52px; margin-top: 8px;
  display: inline-flex; align-items: center; justify-content: center;
}
.kotoba-replay:hover { background: #eff6ff; }
"""

# SentencesJSON field: [{s:<html>, t:<plain>, a:<filename|"">, p:<plain>, h?:<hiragana>}, ...]
# Audios field:        [sound:f1.wav][sound:f2.wav]... — kept in field values so Anki's
#                      media manager counts the files as used; NOT rendered in templates.
# JS plays the selected sentence's audio directly via the HTML5 Audio API.

_AUDIO_JS = """\
  if (items[idx].a) {
    var _audio = new Audio(items[idx].a);
    _audio.play();
    window._kotobaReplay = function () { _audio.currentTime = 0; _audio.play(); };
  }"""

_FRONT_TMPL = """\
<script type="application/json" id="kotoba-data">{{SentencesJSON}}</script>
<div id="kotoba-sentence" class="sentence"></div>
<button class="kotoba-replay" onclick="window._kotobaReplay&&window._kotobaReplay()">&#9654;</button>
<script>
(function () {
  var items = JSON.parse(document.getElementById('kotoba-data').textContent);
  var idx = Math.floor(Math.random() * items.length);
  sessionStorage.setItem('kotobaIdx', String(idx));
  document.getElementById('kotoba-sentence').innerHTML = items[idx].s;
""" + _AUDIO_JS + """
}());
</script>"""

_BACK_TMPL = """\
<script type="application/json" id="kotoba-data">{{SentencesJSON}}</script>
<div id="kotoba-sentence" class="sentence"></div>
<button class="kotoba-replay" onclick="window._kotobaReplay&&window._kotobaReplay()">&#9654;</button>
<hr id=answer>
<div id="kotoba-translation" class="translation"></div>
{{#Word}}<div class="word">{{Word}}</div>{{/Word}}
<script>
(function () {
  var items = JSON.parse(document.getElementById('kotoba-data').textContent);
  var idx = parseInt(sessionStorage.getItem('kotobaIdx') || '0', 10);
  if (idx < 0 || idx >= items.length) idx = 0;
  document.getElementById('kotoba-sentence').innerHTML = items[idx].s;
  document.getElementById('kotoba-translation').textContent = items[idx].t;
  if (items[idx].a) {
    window._kotobaReplay = function () { new Audio(items[idx].a).play(); };
  }
}());
</script>"""

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

# ── Pronunciation card model ────────────────────────────────────────────────────
# SentencesJSON items carry two extra fields used only by this template:
#   p — plain sentence text (no HTML), for scoring against kanji input
#   h — flat hiragana transcription, for scoring against kana input (Japanese only)

_PRON_CSS = _MODEL_CSS + """\
#typeans { display: none !important; }
.pron-label { color: #888; font-size: 14px; margin-bottom: 4px; }
.score-box { font-size: 52px; font-weight: 700; margin: 14px 0 4px; }
.score-excellent { color: #16a34a; }
.score-ok        { color: #d97706; }
.score-poor      { color: #dc2626; }
.score-verdict   { font-size: 17px; color: #555; margin-bottom: 10px; }
.you-said        { font-size: 15px; color: #777; margin: 6px 0 14px; }
.you-said em     { color: #1c1c1e; font-style: normal; }
"""

# Pronunciation cards use {{type:SentencePlain}} solely for iOS keyboard/mic input.
# Anki's own comparison display (#typeans) is hidden; we extract the typed text from
# it on the back and run our own Levenshtein scoring against both kanji and hiragana.
# Each pronunciation note always uses sentence index 0 (fixed target for {{type:}}).

_PRON_FRONT_TMPL = """\
<script type="application/json" id="kotoba-data">{{SentencesJSON}}</script>
<div class="pron-label">Listen, then show answer to speak:</div>
<div id="kotoba-sentence" class="sentence"></div>
<button class="kotoba-replay" onclick="window._kotobaReplay&&window._kotobaReplay()">&#9654;</button>
{{type:SentencePlain}}
<script>
(function () {
  var items = JSON.parse(document.getElementById('kotoba-data').textContent);
  document.getElementById('kotoba-sentence').innerHTML = items[0].s;
""" + _AUDIO_JS.replace("items[idx]", "items[0]") + """
}());
</script>"""

_PRON_BACK_TMPL = """\
<script type="application/json" id="kotoba-data">{{SentencesJSON}}</script>
<div id="kotoba-sentence" class="sentence"></div>
<button class="kotoba-replay" onclick="window._kotobaReplay&&window._kotobaReplay()">&#9654;</button>
{{type:SentencePlain}}
<hr id=answer>
<div id="score-box" class="score-box"></div>
<div id="score-verdict" class="score-verdict"></div>
<div id="you-said" class="you-said"></div>
<div id="kotoba-translation" class="translation"></div>
{{#Word}}<div class="word">{{Word}}</div>{{/Word}}
<script>
(function () {
  function lev(a, b) {
    var m = a.length, n = b.length, i, j;
    var dp = [];
    for (i = 0; i <= m; i++) { dp[i] = [i]; }
    for (j = 1; j <= n; j++) { dp[0][j] = j; }
    for (i = 1; i <= m; i++) {
      for (j = 1; j <= n; j++) {
        dp[i][j] = a[i-1] === b[j-1] ? dp[i-1][j-1]
          : 1 + Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
      }
    }
    return dp[m][n];
  }
  function sim(a, b) {
    if (!a && !b) return 1;
    if (!a || !b) return 0;
    return 1 - lev(a, b) / Math.max(a.length, b.length);
  }
  function norm(s) {
    return s.trim().replace(/[\\u30A1-\\u30F6]/g, function (c) {
      return String.fromCharCode(c.charCodeAt(0) - 0x60);
    });
  }
  function esc(s) {
    return s.replace(/[<>&"]/g, function (c) {
      return {'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'}[c];
    });
  }
  var item = JSON.parse(document.getElementById('kotoba-data').textContent)[0];
  document.getElementById('kotoba-sentence').innerHTML = item.s;
  document.getElementById('kotoba-translation').textContent = item.t;
  if (item.a) {
    window._kotobaReplay = function () { new Audio(item.a).play(); };
  }
  function extractTyped() {
    var typeans = document.getElementById('typeans');
    if (!typeans) return '';
    // Anki marks user's chars as typeGood (correct) or typeBad (wrong); typeMissed = not typed.
    var spans = typeans.querySelectorAll('.typeGood, .typeBad');
    if (spans.length > 0) {
      var t = ''; spans.forEach(function (el) { t += el.textContent; }); return t;
    }
    // Fallback: full text minus missed chars (different Anki versions).
    var clone = typeans.cloneNode(true);
    clone.querySelectorAll('.typeMissed').forEach(function (el) { el.remove(); });
    return clone.textContent.trim();
  }
  function showScore() {
    var userRaw  = extractTyped();
    var userNorm = norm(userRaw);
    var score = Math.max(
      sim(userNorm, norm(item.p || '')),
      item.h ? sim(userNorm, norm(item.h)) : 0
    );
    var pct = Math.round(score * 100);
    var box     = document.getElementById('score-box');
    var verdict = document.getElementById('score-verdict');
    box.textContent = pct + '%';
    if (pct >= 85) {
      box.className = 'score-box score-excellent';
      verdict.textContent = 'Excellent!';
    } else if (pct >= 60) {
      box.className = 'score-box score-ok';
      verdict.textContent = 'Almost there';
    } else {
      box.className = 'score-box score-poor';
      verdict.textContent = 'Keep practising';
    }
    document.getElementById('you-said').innerHTML =
      userRaw ? 'You said: <em>' + esc(userRaw) + '</em>' : '<em>(nothing recorded)</em>';
  }
  // #typeans is populated asynchronously by Anki after the template renders.
  var typeans = document.getElementById('typeans');
  if (typeans && typeans.children.length > 0) {
    showScore();
  } else {
    new MutationObserver(function (mutations, obs) {
      var ta = document.getElementById('typeans');
      if (ta && (ta.children.length > 0 || ta.textContent.trim())) {
        obs.disconnect();
        showScore();
      }
    }).observe(document.body, { childList: true, subtree: true });
  }
  // Fallback: if observer never fires (e.g. user typed nothing), show after 1.5 s.
  setTimeout(function () {
    if (!document.getElementById('score-box').textContent) showScore();
  }, 1500);
}());
</script>"""

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


# ── Build .apkg ────────────────────────────────────────────────────────────────

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

            items: list[dict] = []
            audio_queue: list[str] = []  # filenames in order, for the Audios field
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
                    "p": sent["sentence"],  # plain text for pronunciation scoring
                }
                if audio_filename:
                    item["qi"] = len(audio_queue)  # index in Anki's sound queue
                    audio_queue.append(audio_filename)
                if is_japanese:
                    item["h"] = sentence_to_hiragana(sent["sentence"])
                items.append(item)

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


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Anki flashcards from a word list (Gemini + Kokoro TTS)",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--words", metavar="WORD,WORD,...",
                     help="Comma-separated list of words")
    src.add_argument("--words-file", metavar="FILE",
                     help="One word per line")
    parser.add_argument("--language", required=True,
                        help="Target language, e.g. spanish, japanese, italian")
    parser.add_argument("--proficiency", choices=["newbie", "a1", "a2", "b1"],
                        help="Learner proficiency level (recommended)")
    parser.add_argument("--topic",
                        help="Topic context for sentence generation")
    parser.add_argument("--sentence-count", type=int, default=2, choices=range(1, 6),
                        help="Sentences per word, 1–5 (default: 2)")
    parser.add_argument("--deck-name", metavar="NAME",
                        help="Anki deck name (default: Kotoba::<Language>[::<Topic>])")
    parser.add_argument("--output", metavar="FILE",
                        help="Output .apkg path (default: <language>_anki.apkg)")
    parser.add_argument("--gemini-api-key", metavar="KEY",
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip audio generation; text-only cards")
    parser.add_argument("--pronunciation-cards", action="store_true",
                        help="Generate pronunciation cards instead of reading cards")
    return parser


def _init(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[genai.Client, list[str], str, str]:
    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        parser.error("Gemini API key required: --gemini-api-key or GEMINI_API_KEY env var")

    client = genai.Client(api_key=api_key)

    if not args.no_audio:
        _load_audio_deps()

    if args.words:
        words = [w.strip() for w in args.words.split(",") if w.strip()]
    else:
        with open(args.words_file, encoding="utf-8") as fh:
            words = [line.strip() for line in fh if line.strip()]

    if not words:
        parser.error("No words found")

    deck_name = args.deck_name or (
        f"Kotoba::{args.language.capitalize()}"
        + (f"::{args.topic.capitalize()}" if args.topic else "")
    )
    output_path = args.output or f"{args.language}_anki.apkg"

    return client, words, deck_name, output_path


def _print_params(args: argparse.Namespace, words: list[str], deck_name: str, output_path: str) -> None:
    print(f"Language   : {args.language}")
    print(f"Words      : {len(words)}  ({', '.join(words[:6])}{'...' if len(words) > 6 else ''})")
    if args.proficiency:
        print(f"Proficiency: {args.proficiency.upper()}")
    if args.topic:
        print(f"Topic      : {args.topic}")
    print(f"Sentences  : {args.sentence_count} per word")
    print(f"Deck       : {deck_name}")
    print(f"Audio      : {'off' if args.no_audio else 'kokoro-onnx'}")
    print(f"Cards      : {'pronunciation' if args.pronunciation_cards else 'reading'}")
    print(f"Output     : {output_path}")
    print()


def _generate_sentences(args: argparse.Namespace, words: list[str], client: genai.Client) -> list[dict]:
    print("── Generating sentences ────────────────────────────────────────")
    words_data: list[dict] = []

    for word in words:
        print(f"  {word}...", end=" ", flush=True)
        try:
            data = fetch_word_data(
                word=word,
                language=args.language,
                proficiency=args.proficiency,
                topic=args.topic,
                sentence_count=args.sentence_count,
                client=client,
            )
            sentences = [
                {"sentence": s.sentence, "translation": s.translation, "audio_bytes": None}
                for s in data.sentences[: args.sentence_count]
            ]
            words_data.append({
                "word": word,
                "word_translation": data.word_translation,
                "sentences": sentences,
            })
            print(f'→ "{data.word_translation}" ({len(sentences)} sentences)')
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            print(f"  skipping '{word}'")

    if not words_data:
        sys.exit("No words processed.")

    return words_data


def _generate_audio(words_data: list[dict], language: str) -> None:
    total = sum(len(w["sentences"]) for w in words_data)
    print(f"\n── Generating audio ({total} sentences via kokoro-onnx) ─────────")

    all_sents = [sent for word_info in words_data for sent in word_info["sentences"]]
    ok = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(generate_audio, sent["sentence"], language): sent
            for sent in all_sents
        }
        for future in as_completed(futures):
            sent = futures[future]
            try:
                sent["audio_bytes"] = future.result()
                ok += 1
                print(".", end="", flush=True)
            except Exception as exc:
                print(f"\n  Warning: audio failed for '{sent['sentence'][:40]}': {exc}",
                      file=sys.stderr)
    print(f"\n  {ok}/{total} audio files generated")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    client, words, deck_name, output_path = _init(args, parser)
    _print_params(args, words, deck_name, output_path)
    words_data = _generate_sentences(args, words, client)

    if args.no_audio:
        print("── Audio skipped ───────────────────────────────────────────────")
    else:
        _generate_audio(words_data, args.language)

    print(f"\n── Writing {output_path} ──────────────────────────────────────────")
    card_count = build_apkg(
        words_data, deck_name, output_path,
        language=args.language,
        pronunciation_cards=args.pronunciation_cards,
    )
    print(f"Done — {card_count} cards → {output_path}")


if __name__ == "__main__":
    main()
