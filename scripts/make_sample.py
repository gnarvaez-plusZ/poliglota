#!/usr/bin/env python3
"""Genera audio de charla tecnica para probar el pipeline, usando el TTS de Gemini.

Existe para tener material de prueba reproducible y sin problemas de derechos:
un texto cargado de jerga ("Kubernetes", "gRPC", "eBPF", "p99") que es justo
donde un reconocedor sin cebar se equivoca. Sirve para medir cuanto aporta el
glosario, comparando la misma pista con y sin `adaptation_phrases`.

Para el video de la entrega hay que usar audio real de una charla, no esto.

    scripts/make_sample.py samples/charla-en.wav --lang en
"""
from __future__ import annotations

import argparse
import asyncio
import struct
import sys
import wave

from google import genai
from google.genai import types

from poliglota.config import settings

TTS_MODELS = [
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
    "gemini-flash-latest-tts",
]

SCRIPTS = {
    "en": (
        "Read this as a confident conference speaker, at a natural pace:\n\n"
        "Good morning everyone. Today I want to talk about what actually broke when we scaled "
        "our retrieval augmented generation pipeline to forty thousand requests per minute. "
        "The first myth I want to kill is that your vector database is the bottleneck. It almost never is. "
        "In our case the p99 latency was dominated by the embedding step, which we were running synchronously "
        "inside the request path. We moved it behind a Kafka topic and the p99 dropped from eight hundred "
        "milliseconds to ninety. The second thing is chunking. Everyone argues about chunk size and nobody "
        "measures overlap. We run all of this on Kubernetes, with a horizontal pod autoscaler driven by "
        "queue depth rather than CPU, and we trace every hop with eBPF and export the spans over gRPC to "
        "Grafana Tempo. If you take one thing home: measure before you shard."
    ),
    "es": (
        "Leé esto como un orador de conferencia, con ritmo natural:\n\n"
        "Buenos días a todos. Hoy quiero contarles qué se rompió de verdad cuando llevamos nuestro "
        "pipeline de generación aumentada por recuperación a cuarenta mil pedidos por minuto. "
        "El primer mito que quiero matar es que el cuello de botella está en la base de datos vectorial. "
        "Casi nunca lo está. En nuestro caso la latencia p99 estaba dominada por el paso de embeddings, "
        "que corríamos de forma sincrónica dentro del request. Lo movimos detrás de un tópico de Kafka y "
        "la p99 bajó de ochocientos milisegundos a noventa. Todo esto corre sobre Kubernetes, con un "
        "autoescalador horizontal guiado por profundidad de cola y no por CPU, y trazamos cada salto con "
        "eBPF exportando los spans por gRPC."
    ),
}


def write_wav(path: str, pcm: bytes, rate: int = 24000) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="archivo .wav de salida")
    ap.add_argument("--lang", default="en", choices=sorted(SCRIPTS))
    ap.add_argument("--voice", default="Charon", help="voz prearmada de Gemini TTS")
    args = ap.parse_args()

    if not settings.gemini_api_key:
        print("Falta GEMINI_API_KEY en .env", file=sys.stderr)
        return 2
    client = genai.Client(api_key=settings.gemini_api_key)

    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=args.voice)
            )
        ),
    )

    for model in TTS_MODELS:
        try:
            resp = await client.aio.models.generate_content(
                model=model, contents=SCRIPTS[args.lang], config=cfg
            )
            pcm = resp.candidates[0].content.parts[0].inline_data.data
            write_wav(args.out, pcm)
            secs = len(pcm) / (24000 * 2)
            print(f"{args.out}: {secs:.1f}s generados con {model} (voz {args.voice})")
            print(f"Probalo:  scripts/feed.py auditorio {args.out} --lang {args.lang} \\")
            print(f"            --title 'Scaling RAG in production' --abstract 'Kafka, Kubernetes, eBPF, gRPC, p99'")
            return 0
        except Exception as exc:
            print(f"  {model}: {type(exc).__name__}: {str(exc)[:110]}", file=sys.stderr)

    print("Ningun modelo de TTS respondio.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
