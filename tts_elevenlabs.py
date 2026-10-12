"""ElevenLabs TTS backend (cloud).

Requests raw PCM and wraps it as WAV so the rest of the pipeline (which names
media files `.wav`) is unaffected. Results are cached on disk by
(model, voice, text) so re-exports don't spend credits again.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
import uuid
import wave
from pathlib import Path

import httpx

CACHE_DIR = Path.home() / ".cache" / "kotoba-ai" / "elevenlabs"
API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
SAMPLE_RATE = 24000
REQUEST_TIMEOUT_S = 60.0
DEFAULT_MODEL = "eleven_flash_v2_5"

# Same voices/models the lang-app uses (config.toml). No French or Chinese
# voice is configured there, so those fail with a clear error.
VOICES: dict[str, list[str]] = {
    "japanese":   ["sRYzP8TwEiiqAWebdYPJ"],
    "spanish":    ["Nh2zY9kknu6z4pZy6FhD", "P951amuWPNCJ0L15rFyC"],
    "italian":    ["oCS6WHyqobqW2UapCSHl", "CiwzbDpaN3pQXjTgx3ML", "P951amuWPNCJ0L15rFyC"],
    "portuguese": ["GDzHdQOi6jjf8zaXhCYD", "9pDzHy2OpOgeXM8SeL0t"],
}
MODELS: dict[str, str] = {"japanese": "eleven_v3"}
LANGUAGE_CODES: dict[str, str] = {
    "japanese": "ja", "spanish": "es", "italian": "it",
    "portuguese": "pt", "french": "fr", "chinese": "zh",
}

_api_key: str | None = None
# The lang-app limits ElevenLabs to one concurrent request; do the same.
_request_lock = threading.Lock()


def init(api_key: str) -> None:
    global _api_key
    if not api_key:
        raise ValueError("ElevenLabs API key required (--elevenlabs-api-key or ELEVENLABS_API_KEY)")
    _api_key = api_key


def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def generate(text: str, language: str) -> bytes:
    """Synthesise WAV audio and return raw bytes."""
    if _api_key is None:
        raise RuntimeError("tts_elevenlabs.init() must be called before generate()")
    lang = language.lower()
    voices = VOICES.get(lang)
    if not voices:
        raise ValueError(f"No ElevenLabs voice configured for {lang}")
    voice_id = voices[0]
    model = MODELS.get(lang, DEFAULT_MODEL)

    # Spaces cause audible pauses in Japanese TTS.
    tts_text = text.replace(" ", "") if lang == "japanese" else text

    cache_key = hashlib.sha256(f"{model}|{voice_id}|{tts_text}".encode()).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.wav"
    if cache_path.exists():
        return cache_path.read_bytes()

    with _request_lock:
        response = httpx.post(
            API_URL.format(voice_id=voice_id),
            headers={"xi-api-key": _api_key, "Content-Type": "application/json"},
            params={"output_format": f"pcm_{SAMPLE_RATE}"},
            json={
                "text": tts_text,
                "model_id": model,
                "language_code": LANGUAGE_CODES.get(lang, lang),
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
            },
            timeout=REQUEST_TIMEOUT_S,
        )
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs API error {response.status_code}: {response.text[:200]}")

    wav = _pcm_to_wav(response.content)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a concurrent reader never sees a half-written file.
    tmp_path = cache_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp_path.write_bytes(wav)
    os.replace(tmp_path, cache_path)
    return wav
