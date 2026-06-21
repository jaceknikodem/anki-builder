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

# ── required deps ──────────────────────────────────────────────────────────────

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    sys.exit("Missing dep: uv pip install google-genai")

try:
    from pydantic import BaseModel
except ImportError:
    sys.exit("Missing dep: uv pip install pydantic")

try:
    import genanki
except ImportError:
    sys.exit("Missing dep: uv pip install genanki")

# ── audio deps — only imported when audio is enabled (see _load_audio_deps) ───

_sf = None           # soundfile module
_Kokoro = None       # kokoro_onnx.Kokoro class
_kokoro = None       # loaded Kokoro instance
_ja_g2p = None       # misaki Japanese G2P (optional)
_ja_g2p_lock = threading.Lock()  # misaki is not documented thread-safe
_ja_tagger = None    # fugashi MeCab tagger (optional, Japanese furigana)

CACHE_DIR  = Path.home() / ".cache" / "kotoba-ai"
MODEL_PATH = CACHE_DIR / "kokoro-v1.0.int8.onnx"
VOICES_PATH = CACHE_DIR / "voices-v1.0.bin"


def _load_audio_deps() -> None:
    """Import audio libraries and verify model files exist. Exits on failure."""
    global _sf, _Kokoro, _ja_g2p

    missing = []
    if not MODEL_PATH.exists():
        missing.append(f"model file: {MODEL_PATH}")
    if not VOICES_PATH.exists():
        missing.append(f"voices file: {VOICES_PATH}")
    if missing:
        print("Kokoro model files not found:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("Run:  bash setup_kokoro.sh", file=sys.stderr)
        sys.exit(1)

    try:
        import soundfile as sf
        _sf = sf
    except ImportError:
        sys.exit("Missing audio dep: run  bash setup_kokoro.sh")

    try:
        from kokoro_onnx import Kokoro
        _Kokoro = Kokoro
    except ImportError:
        sys.exit("Missing audio dep: run  bash setup_kokoro.sh")

    try:
        from misaki import ja as misaki_ja
        _ja_g2p = misaki_ja.JAG2P()
    except ImportError:
        pass  # Japanese phonemization degrades gracefully


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
        "newbie": "presente (essere/avere/regular), simple S-V patterns, fixed chunks (posso/devo/voglio)",
        "a1":     "presente all persons, passato prossimo recognition, modal + infinitive, imperatives",
        "a2":     "productive passato prossimo, imperfetto/futuro recognition, gerundio progressivo",
        "b1":     "productive imperfetto, full condizionale, congiuntivo recognition, clitics",
    },
    "spanish": {
        "newbie": "presente (ser/estar/tener + regular), simple S-V patterns",
        "a1":     "presente all persons, basic reflexives, modal + infinitive, periphrastic future",
        "a2":     "pretérito perfecto compuesto, estar+gerundio, basic condicional, gustar verbs",
        "b1":     "present subjunctive, advanced connectors, conversational idioms",
    },
    "portuguese": {
        "newbie": "presente (ser/estar/ter/regular), simple S-V patterns",
        "a1":     "presente all persons, pretérito perfeito recognition, poder/precisar + infinitive",
        "a2":     "pretérito perfeito, estar+gerúndio, basic condicional (gostaria/poderia)",
        "b1":     "pretérito imperfeito, futuro do presente, full condicional, subjuntivo recognition",
    },
    "japanese": {
        "newbie": "copula (です/ではありません), simple polite verb forms, basic particles (は・が・を・に)",
        "a1":     "polite past (〜ました), te-form recognition, あります/います",
        "a2":     "te-form usage, plain forms, potential (〜られる/〜できる), basic conditionals (〜たら)",
        "b1":     "full plain-form conjugation, extended te-forms (〜てしまう), passive/causative",
    },
    "polish": {
        "newbie": "basic present tense (być/mieć), S-V-O, simple negation, noun gender",
        "a1":     "present tense all persons, past recognition, modals (mogę/chcę/muszę), basic cases",
        "a2":     "past tense all genders, future (będę + inf), accusative/dative in predictable patterns",
        "b1":     "perfective/imperfective, conditional forms, instrumental/genitive, subordinate clauses",
    },
    "french": {
        "newbie": "présent (être/avoir/aller/regular -er), S-V-O, basic negation (ne…pas)",
        "a1":     "présent all persons, passé composé recognition, modal + inf (pouvoir/vouloir/devoir)",
        "a2":     "productive passé composé, imparfait recognition, futur proche, pronoms COD",
        "b1":     "imparfait vs passé composé, futur simple, conditionnel, subjonctif recognition",
    },
    "indonesian": {
        "newbie": "simple S-V-O, basic stative adjectives, negation (tidak/bukan), common pronouns",
        "a1":     "me-/ber- prefixes, question words, possessives -nya, reduplication",
        "a2":     "di- passives, modals (bisa/harus/mau), aspect markers (sedang/sudah/belum/akan)",
        "b1":     "varied affix combinations, full passive system, complex clauses (kalau/meskipun)",
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

def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        print("  Loading Kokoro model...", flush=True)
        _kokoro = _Kokoro(str(MODEL_PATH), str(VOICES_PATH))
        print("  Kokoro ready", flush=True)
    return _kokoro


def generate_audio(text: str, language: str) -> bytes:
    """Synthesise WAV audio and return raw bytes."""
    voice = _voice(language)
    kokoro = _get_kokoro()
    lang = language.lower()

    if lang == "japanese" and _ja_g2p is not None:
        with _ja_g2p_lock:
            ipa, _ = _ja_g2p(text)
        audio, sr = kokoro.create(ipa, voice=voice, is_phonemes=True)
    else:
        espeak_lang = LANG_TO_ESPEAK.get(lang, "en-us")
        audio, sr = kokoro.create(text, voice=voice, lang=espeak_lang)

    buf = io.BytesIO()
    _sf.write(buf, audio, sr, format="WAV")
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

def _get_ja_tagger():
    global _ja_tagger
    if _ja_tagger is None:
        try:
            import fugashi
            _ja_tagger = fugashi.Tagger()
        except Exception:
            _ja_tagger = False  # disable gracefully
    return _ja_tagger if _ja_tagger else None


def _kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ン" else c for c in s)


def _has_kanji(s: str) -> bool:
    return any("一" <= c <= "鿿" or "㐀" <= c <= "䶿" for c in s)


def furigana_highlight_html(sentence: str, word: str) -> str:
    """Return sentence as HTML with ruby furigana; the target word is highlighted."""
    tagger = _get_ja_tagger()
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
    tagger = _get_ja_tagger()
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
  background: none; border: 1.5px solid #2563eb; border-radius: 50%;
  color: #2563eb; cursor: pointer; font-size: 16px;
  width: 34px; height: 34px; margin-top: 8px;
  display: inline-flex; align-items: center; justify-content: center;
}
.kotoba-replay:hover { background: #eff6ff; }
"""

# SentencesJSON field: [{s:<html>, t:<plain>, a:<filename|"">, qi:<audio-queue-idx>?}, ...]
# Audios field:        [sound:f1.wav][sound:f2.wav]... — processed by Anki for native audio.
# JS picks one entry randomly on the front, stores the index in sessionStorage, and
# intercepts Anki's pycmd auto-play to redirect it to the selected sentence's audio.
# pycmd is set up asynchronously (QWebChannel), so we poll until it's available.

_AUDIO_JS = """\
  var _qi = items[idx].qi;
  if (_qi !== undefined) {
    var _fired = false, _n = 0;
    (function poll() {
      if (typeof pycmd !== 'function') { if (++_n < 100) setTimeout(poll, 20); return; }
      var orig = pycmd;
      window._kotobaReplay = function () { orig('play:q:' + _qi); };
      pycmd = function (cmd) {
        if (/^play:q:\\d+$/.test(cmd)) {
          if (!_fired) { _fired = true; orig('play:q:' + _qi); }
          return;
        }
        orig(cmd);
      };
    }());
  }"""

_FRONT_TMPL = """\
<script type="application/json" id="kotoba-data">{{SentencesJSON}}</script>
<div id="kotoba-sentence" class="sentence"></div>
<button class="kotoba-replay" onclick="window._kotobaReplay&&window._kotobaReplay()">&#9654;</button>
<span style="display:none">{{Audios}}</span>
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
  var qi = items[idx].qi;
  if (qi !== undefined) {
    window._kotobaReplay = function () { if (typeof pycmd === 'function') pycmd('play:q:' + qi); };
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
<span style="display:none">{{Audios}}</span>
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
  if (item.qi !== undefined) {
    window._kotobaReplay = function () { if (typeof pycmd === 'function') pycmd('play:q:' + item.qi); };
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

def main() -> None:
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

    args = parser.parse_args()

    # ── API key ──────────────────────────────────────────────────────────────
    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        parser.error("Gemini API key required: --gemini-api-key or GEMINI_API_KEY env var")

    client = genai.Client(api_key=api_key)

    # ── Audio deps (lazy) ─────────────────────────────────────────────────────
    if not args.no_audio:
        _load_audio_deps()

    # ── Word list ─────────────────────────────────────────────────────────────
    if args.words:
        words = [w.strip() for w in args.words.split(",") if w.strip()]
    else:
        with open(args.words_file, encoding="utf-8") as fh:
            words = [line.strip() for line in fh if line.strip()]

    if not words:
        parser.error("No words found")

    # ── Names ─────────────────────────────────────────────────────────────────
    deck_name = args.deck_name or (
        f"Kotoba::{args.language.capitalize()}"
        + (f"::{args.topic.capitalize()}" if args.topic else "")
    )
    output_path = args.output or f"{args.language}_anki.apkg"

    # ── Banner ────────────────────────────────────────────────────────────────
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

    # ── Step 1: Sentences ─────────────────────────────────────────────────────
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

    # ── Step 2: Audio ─────────────────────────────────────────────────────────
    total = sum(len(w["sentences"]) for w in words_data)

    if args.no_audio:
        print(f"\n── Audio skipped ───────────────────────────────────────────────")
    else:
        print(f"\n── Generating audio ({total} sentences via kokoro-onnx) ─────────")
        _get_kokoro()  # load model once before spawning threads

        # Flatten to (sent_dict, text, language) triples so threads can work independently
        all_sents = [
            sent
            for word_info in words_data
            for sent in word_info["sentences"]
        ]

        ok = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(generate_audio, sent["sentence"], args.language): sent
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

    # ── Step 3: Package ───────────────────────────────────────────────────────
    print(f"\n── Writing {output_path} ──────────────────────────────────────────")
    card_count = build_apkg(
        words_data, deck_name, output_path,
        language=args.language,
        pronunciation_cards=args.pronunciation_cards,
    )
    print(f"Done — {card_count} cards → {output_path}")


if __name__ == "__main__":
    main()
