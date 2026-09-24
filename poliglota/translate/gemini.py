"""Traduccion con Gemini Flash: una llamada, todos los idiomas."""
from __future__ import annotations

import json
import logging

from google import genai
from google.genai import types

from ..config import settings
from .base import TranslationResult
from .prompt import SYSTEM, build_user_prompt

log = logging.getLogger("poliglota.mt.gemini")


class GeminiTranslator:
    name = "gemini"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.mt_model
        self._client = genai.Client(api_key=settings.require_key())

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
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={code: types.Schema(type=types.Type.STRING) for code in wanted},
            required=wanted,
        )
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.0,
                    # Un subtitulo nunca es largo; el tope evita que el modelo
                    # se vaya de tema y acota la latencia del peor caso.
                    max_output_tokens=512,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            data = json.loads(resp.text)
            usage = getattr(resp, "usage_metadata", None)
            return TranslationResult(
                texts={k: str(v) for k, v in data.items() if k in wanted},
                tokens=getattr(usage, "total_token_count", 0) or 0,
            )
        except Exception:
            log.exception("Fallo la traduccion; se emite el original")
            return TranslationResult()
