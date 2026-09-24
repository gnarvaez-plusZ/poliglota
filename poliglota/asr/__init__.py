"""Motores de reconocimiento de voz (ASR) intercambiables."""
from __future__ import annotations

from ..config import settings
from .base import ASREngine, Transcript


def build_asr(engine: str | None = None) -> ASREngine:
    name = (engine or settings.asr_engine).lower()
    if name == "gemini":
        from .gemini_live import GeminiLiveASR

        return GeminiLiveASR()
    if name == "mock":
        from .mock import MockASR

        return MockASR()
    if name == "whisper":
        from .whisper_local import WhisperASR

        return WhisperASR()
    raise ValueError(f"Motor de ASR desconocido: {name!r} (usa 'gemini', 'whisper' o 'mock')")


__all__ = ["ASREngine", "Transcript", "build_asr"]
