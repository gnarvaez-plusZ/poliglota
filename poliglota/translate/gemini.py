"""Traduccion con Gemini: una llamada, todos los idiomas, y tolerante a saturacion.

Durante las pruebas el modelo devolvio `503 UNAVAILABLE - high demand`. En una
conferencia en vivo eso no puede dejar a la audiencia sin subtitulos, asi que el
traductor reintenta con espera creciente y, si el modelo preferido sigue caido,
baja a uno de respaldo. Un fallo total no rompe nada: la sala sigue mostrando el
idioma original, que ya salio sin esperar a la traduccion.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from ..config import settings
from .base import TranslationResult
from .prompt import SYSTEM, build_user_prompt

log = logging.getLogger("poliglota.mt.gemini")

_MAX_ATTEMPTS = 3
# Codigos que indican "volve a intentar", no "tu pedido esta mal".
_RETRYABLE = {429, 500, 502, 503, 504}
_BUSY_COOLDOWN = 20.0   # segundos de descanso para un modelo saturado
_DEAD_COOLDOWN = 600.0  # un modelo que rechaza el pedido no se recupera solo


class GeminiTranslator:
    name = "gemini"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.mt_model
        # Si el preferido esta saturado, se sigue traduciendo con el de respaldo.
        self.fallbacks = [m for m in settings.mt_fallbacks if m != self.model]
        # {modelo: momento en que vuelve a estar disponible}
        self._cooldown: dict[str, float] = {}
        self._client = genai.Client(api_key=settings.require_key())

    def _config(self, wanted: list[str]) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={c: types.Schema(type=types.Type.STRING) for c in wanted},
                required=wanted,
            ),
            temperature=0.0,
            # Un subtitulo nunca es largo; el tope acota la latencia del peor caso.
            max_output_tokens=512,
            # Sin thinking_config a proposito: los modelos "lite" lo rechazan con
            # 400 INVALID_ARGUMENT, y son justo los que mejor sirven aca por
            # rapidos. Traducir una linea no necesita razonamiento previo.
        )

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
        cfg = self._config(wanted)
        retries = 0
        now = time.monotonic()

        for model in [self.model, *self.fallbacks]:
            # Cortacircuitos: un modelo que acaba de devolver 429 o 503 no va a
            # estar sano un segundo despues. Insistir solo gasta cuota y suma
            # latencia a un subtitulo que ya va tarde. Se lo saltea hasta que
            # venza su enfriamiento.
            if self._cooldown.get(model, 0.0) > now:
                continue

            for attempt in range(_MAX_ATTEMPTS):
                try:
                    resp = await self._client.aio.models.generate_content(
                        model=model, contents=prompt, config=cfg
                    )
                    data = json.loads(resp.text)
                    usage = getattr(resp, "usage_metadata", None)
                    self._cooldown.pop(model, None)
                    return TranslationResult(
                        texts={k: str(v) for k, v in data.items() if k in wanted},
                        tokens=getattr(usage, "total_token_count", 0) or 0,
                        retries=retries,
                    )
                except genai_errors.APIError as exc:
                    code = getattr(exc, "code", None)
                    if code not in _RETRYABLE:
                        # 400 o 404: el modelo no sirve para este pedido y no va
                        # a servir nunca. Se lo aparta por largo rato.
                        log.warning("%s descartado: %s", model, str(exc)[:110])
                        self._cooldown[model] = time.monotonic() + _DEAD_COOLDOWN
                        break
                    if attempt == _MAX_ATTEMPTS - 1:
                        log.warning("%s saturado (%s); enfriando", model, code)
                        self._cooldown[model] = time.monotonic() + _BUSY_COOLDOWN
                        break
                    retries += 1
                    # Espera creciente con ruido, para no sincronizar el reintento
                    # de todas las salas contra el mismo instante.
                    delay = 0.4 * (2 ** attempt) + random.uniform(0, 0.25)
                    await asyncio.sleep(delay)
                except Exception:
                    log.exception("Fallo inesperado traduciendo con %s", model)
                    self._cooldown[model] = time.monotonic() + _BUSY_COOLDOWN
                    break

        log.warning("Sin traduccion tras %d reintentos; se emite el original", retries)
        return TranslationResult(retries=retries)
