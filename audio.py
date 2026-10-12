"""TTS dispatcher. Backends live in tts_<name>.py and are imported on demand,
so e.g. ElevenLabs runs without kokoro-onnx installed."""
from __future__ import annotations

import importlib
from types import ModuleType

BACKENDS = ("kokoro", "elevenlabs")

_backend: ModuleType | None = None


def init_audio(backend: str = "kokoro", **options: str) -> None:
    """Load and initialise a TTS backend. `options` are passed to its `init()`."""
    global _backend
    if backend not in BACKENDS:
        raise ValueError(f"Unknown TTS backend '{backend}' (choose from {', '.join(BACKENDS)})")
    module = importlib.import_module(f"tts_{backend}")
    module.init(**options)
    _backend = module


def generate_audio(text: str, language: str) -> bytes:
    """Synthesise WAV audio and return raw bytes."""
    if _backend is None:
        raise RuntimeError("init_audio() must be called before generate_audio()")
    return _backend.generate(text, language)
