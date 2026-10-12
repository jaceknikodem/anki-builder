"""Kokoro ONNX TTS backend (local, offline)."""
from __future__ import annotations

import io
import threading
from pathlib import Path

import soundfile as sf
from kokoro_onnx import Kokoro
from misaki import ja as misaki_ja

CACHE_DIR = Path.home() / ".cache" / "kotoba-ai"
MODEL_PATH = CACHE_DIR / "kokoro-v1.0.int8.onnx"
VOICES_PATH = CACHE_DIR / "voices-v1.0.bin"

VOICES: dict[str, list[str]] = {
    "japanese":   ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "spanish":    ["ef_dora"],
    "french":     ["ff_siwis"],
    "italian":    ["if_sara"],
    "portuguese": ["pf_dora"],
    "chinese":    ["zf_xiaobei", "zf_xiaoni", "zm_yunxi"],
}

LANG_TO_ESPEAK: dict[str, str] = {
    "spanish":    "es",
    "french":     "fr-fr",
    "italian":    "it",
    "portuguese": "pt-br",
    "chinese":    "cmn",
}

_kokoro: Kokoro | None = None
_ja_g2p = None
_ja_g2p_lock = threading.Lock()


def init() -> None:
    global _kokoro, _ja_g2p
    _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
    _ja_g2p = misaki_ja.JAG2P()


def _voice(language: str) -> str:
    return VOICES.get(language.lower(), ["af_heart"])[0]


def generate(text: str, language: str) -> bytes:
    """Synthesise WAV audio and return raw bytes."""
    if _kokoro is None:
        raise RuntimeError("tts_kokoro.init() must be called before generate()")
    voice = _voice(language)
    lang = language.lower()

    if lang == "japanese" and _ja_g2p is not None:
        with _ja_g2p_lock:
            ipa, _ = _ja_g2p(text)
        samples, sr = _kokoro.create(ipa, voice=voice, is_phonemes=True)
    else:
        espeak_lang = LANG_TO_ESPEAK.get(lang, "en-us")
        samples, sr = _kokoro.create(text, voice=voice, lang=espeak_lang)

    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()
