"""Tests for the ElevenLabs TTS backend and the audio dispatcher."""
from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audio  # noqa: E402
import tts_elevenlabs  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_elevenlabs, "CACHE_DIR", tmp_path / "cache")
    tts_elevenlabs.init(api_key="test-key")
    calls: list[dict] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return _FakeResponse(content=b"\x00\x01" * 100)

    monkeypatch.setattr(tts_elevenlabs.httpx, "post", fake_post)
    return calls


def test_returns_valid_wav(backend):
    data = tts_elevenlabs.generate("どこですか？", "japanese")
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == tts_elevenlabs.SAMPLE_RATE
        assert w.getnframes() == 100


def test_request_shape(backend):
    tts_elevenlabs.generate("どこ です か", "japanese")
    (call,) = backend
    assert call["url"].endswith("/sRYzP8TwEiiqAWebdYPJ")
    assert call["headers"]["xi-api-key"] == "test-key"
    assert call["params"] == {"output_format": f"pcm_{tts_elevenlabs.SAMPLE_RATE}"}
    body = call["json"]
    assert body["text"] == "どこですか"  # spaces cause TTS pauses in Japanese
    assert body["model_id"] == "eleven_v3"
    assert body["language_code"] == "ja"


def test_second_call_hits_cache(backend):
    tts_elevenlabs.generate("どこですか？", "japanese")
    tts_elevenlabs.generate("どこですか？", "japanese")
    assert len(backend) == 1


def test_api_error_includes_status(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_elevenlabs, "CACHE_DIR", tmp_path / "cache")
    tts_elevenlabs.init(api_key="test-key")
    monkeypatch.setattr(
        tts_elevenlabs.httpx, "post",
        lambda *a, **k: _FakeResponse(status_code=401, text="invalid key"),
    )
    with pytest.raises(RuntimeError, match="401"):
        tts_elevenlabs.generate("こんにちは", "japanese")


def test_language_without_voice_fails_clearly(backend):
    with pytest.raises(ValueError, match="french"):
        tts_elevenlabs.generate("Bonjour", "french")


def test_init_requires_api_key():
    with pytest.raises(ValueError, match="API key"):
        tts_elevenlabs.init(api_key="")


def test_dispatcher_rejects_unknown_backend():
    with pytest.raises(ValueError, match="nope"):
        audio.init_audio("nope")


def test_dispatcher_routes_to_elevenlabs(backend):
    audio.init_audio("elevenlabs", api_key="test-key")
    assert audio.generate_audio("こんにちは", "japanese")
    assert len(backend) == 1
