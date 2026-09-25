#!/usr/bin/env python3
"""Verifica contra la API real: que modelos Live existen y si la config es aceptada.

Se corre antes de confiar en el pipeline. Sin esto, el motor de ASR esta escrito
contra la documentacion y no contra el servidor.
"""
from __future__ import annotations

import asyncio
import sys

from google import genai
from google.genai import types

from poliglota.config import settings

CANDIDATES = [
    settings.live_model,
    "gemini-3.8-live",
    "gemini-3.8-live-extended-thinking",
    "gemini-3.1-flash-live-preview",
    "gemini-live-2.5-flash-native-audio",
    "gemini-live-2.5-flash-preview",
    "gemini-2.0-flash-live-001",
]


async def probe(client: genai.Client, model: str) -> tuple[bool, str]:
    """Abre una sesion con la config exacta que usa el pipeline y manda silencio."""
    cfg = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(
            mode=types.AudioTranscriptionConfigMode.SMART,
            diarization=True,
            adaptation_phrases=["Kubernetes", "gRPC"],
            language_hints=types.LanguageHints(language_codes=["en"]),
        ),
        system_instruction="Silent relay. Always reply with a single period.",
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        ),
        session_resumption=types.SessionResumptionConfig(),
    )
    try:
        async with asyncio.timeout(25):
            async with client.aio.live.connect(model=model, config=cfg) as s:
                await s.send_realtime_input(
                    audio=types.Blob(data=b"\x00\x00" * 8000, mime_type="audio/pcm;rate=16000")
                )
                async for _ in s.receive():
                    return True, "conecta y responde"
                return True, "conecta (sin respuesta en la ventana)"
    except TimeoutError:
        return True, "conecta (sin respuesta en 25s)"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:130]}"


async def main() -> int:
    if not settings.gemini_api_key:
        print("Falta GEMINI_API_KEY en .env")
        return 2
    client = genai.Client(api_key=settings.gemini_api_key)

    print("Modelos que el catalogo declara con soporte Live:")
    listed = []
    try:
        async for m in await client.aio.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if any("bidi" in str(a).lower() or "live" in str(a).lower() for a in actions):
                name = m.name.replace("models/", "")
                listed.append(name)
                print(f"  {name}")
    except Exception as exc:
        print(f"  (no se pudo listar: {exc})")
    if not listed:
        print("  (ninguno declarado; se prueban los candidatos igual)")

    print("\nProbando la config real del pipeline contra cada candidato:")
    ok = []
    seen = set()
    for model in [*dict.fromkeys(CANDIDATES), *listed]:
        if model in seen:
            continue
        seen.add(model)
        works, detail = await probe(client, model)
        print(f"  {'OK  ' if works else 'NO  '} {model:44} {detail}")
        if works:
            ok.append(model)

    print(f"\nUsables: {ok or 'NINGUNO'}")
    if ok and ok[0] != settings.live_model:
        print(f"Ajusta POLIGLOTA_LIVE_MODEL={ok[0]} en .env")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
