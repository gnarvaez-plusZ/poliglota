"""Traduccion con Gemma servido por Ollama: el camino 100% local."""
from __future__ import annotations

import json
import logging

import httpx

from ..config import settings
from .base import TranslationResult
from .prompt import SYSTEM, build_user_prompt

log = logging.getLogger("poliglota.mt.gemma")


class GemmaTranslator:
    name = "gemma"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.gemma_model
        self._client = httpx.AsyncClient(base_url=settings.ollama_host, timeout=30.0)

    async def translate(
        self,
        text: str,
        *,
        source_lang: str,
        targets: list[str],
        context: str = "",
        glossary: str = "",
    ) -> TranslationResult:
        wanted = [t for t in targets if t != source_lang]
        if not text.strip() or not wanted:
            return TranslationResult()
        prompt = build_user_prompt(
            text, source_lang=source_lang, targets=wanted, context=context, glossary=glossary
        )
        try:
            r = await self._client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 512},
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            r.raise_for_status()
            data = json.loads(r.json()["message"]["content"])
            return TranslationResult(texts={k: str(v) for k, v in data.items() if k in wanted})
        except Exception:
            log.exception("Fallo la traduccion local; se emite el original")
            return TranslationResult()
