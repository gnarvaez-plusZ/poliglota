"""Motores de traduccion intercambiables."""
from __future__ import annotations

from ..config import settings
from .base import NullTranslator, TranslationResult, Translator


def build_translator(engine: str | None = None) -> Translator:
    name = (engine or settings.mt_engine).lower()
    if name == "gemini":
        from .gemini import GeminiTranslator

        return GeminiTranslator()
    if name == "gemma":
        from .gemma_local import GemmaTranslator

        return GemmaTranslator()
    if name in ("none", "null", ""):
        return NullTranslator()
    raise ValueError(f"Motor de traduccion desconocido: {name!r}")


__all__ = ["Translator", "NullTranslator", "TranslationResult", "build_translator"]
