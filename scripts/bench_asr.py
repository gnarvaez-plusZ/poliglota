#!/usr/bin/env python3
"""Compara modelos y configuraciones de transcripcion con audio real.

La eleccion del modelo y del modo no se puede hacer leyendo la documentacion:
hay combinaciones que el servidor rechaza y diferencias de precision que solo
aparecen con jerga tecnica. Este script alimenta la misma pista a cada
combinacion, a ritmo real, y reporta tres cosas que importan para el reto:
latencia hasta el primer texto, aciertos sobre terminos tecnicos, y si la
combinacion siquiera conecta.
"""
from __future__ import annotations

import asyncio
import re
import sys
import time

from google import genai
from google.genai import types

from poliglota.config import settings

RAW = "samples/charla-en.raw"
CHUNK = 3200          # 100 ms
GLOSSARY = [
    "Kubernetes", "Kafka", "eBPF", "gRPC", "Grafana Tempo", "autoscaler",
    "retrieval augmented generation", "embedding", "p99", "sharding", "chunking",
]
# Terminos que el orador dice y que un reconocedor sin cebar suele errar.
EXPECTED = ["kubernetes", "kafka", "ebpf", "grpc", "grafana", "tempo",
            "autoscaler", "embedding", "retrieval", "p99"]

MODELS = [
    "gemini-3.5-transcribe-live",
    "gemini-3.8-live",
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-latest",
]


def configs() -> dict[str, dict]:
    return {
        "SMART+glosario": dict(mode="SMART", diar=False, phrases=True),
        "SMART sin glosario": dict(mode="SMART", diar=False, phrases=False),
        "VERBATIM+diariz+glosario": dict(mode="VERBATIM", diar=True, phrases=True),
    }


def build(cfg: dict) -> types.LiveConnectConfig:
    tc = types.AudioTranscriptionConfig(
        mode=getattr(types.AudioTranscriptionConfigMode, cfg["mode"]),
        language_hints=types.LanguageHints(language_codes=["en"]),
    )
    if cfg["diar"]:
        tc.diarization = True
    if cfg["phrases"]:
        tc.adaptation_phrases = GLOSSARY
    return types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=tc,
        system_instruction="Silent relay. Always reply with a single period.",
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        ),
        session_resumption=types.SessionResumptionConfig(),
    )


async def run_one(client, model: str, name: str, cfg: dict, pcm: bytes) -> dict:
    res = {"model": model, "cfg": name, "ok": False, "ttft": None, "text": "", "err": ""}
    try:
        async with asyncio.timeout(70):
            async with client.aio.live.connect(model=model, config=build(cfg)) as s:
                t0 = time.time()
                parts: list[str] = []

                async def send():
                    # Ritmo real: 100 ms de audio cada 100 ms.
                    for i in range(0, len(pcm), CHUNK):
                        await s.send_realtime_input(
                            audio=types.Blob(data=pcm[i:i + CHUNK], mime_type="audio/pcm;rate=16000")
                        )
                        await asyncio.sleep(0.1)
                    await s.send_realtime_input(audio_stream_end=True)

                task = asyncio.create_task(send())
                try:
                    async for m in s.receive():
                        sc = m.server_content
                        if not sc:
                            continue
                        t = sc.input_transcription
                        if t and t.text:
                            if res["ttft"] is None:
                                res["ttft"] = int((time.time() - t0) * 1000)
                            parts.append(t.text)
                        if task.done() and sc.turn_complete:
                            break
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                res["text"] = " ".join(parts).strip()
                res["ok"] = bool(res["text"])
    except TimeoutError:
        res["err"] = "timeout 70s"
    except Exception as exc:
        res["err"] = f"{type(exc).__name__}: {str(exc)[:80]}"
    return res


def score(text: str) -> tuple[int, list[str]]:
    low = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    hits = [t for t in EXPECTED if t in low]
    return len(hits), [t for t in EXPECTED if t not in low]


async def main() -> int:
    try:
        pcm = open(RAW, "rb").read()
    except FileNotFoundError:
        print(f"Falta {RAW}. Generalo con make_sample.py + ffmpeg.", file=sys.stderr)
        return 2
    client = genai.Client(api_key=settings.require_key())
    print(f"Audio: {len(pcm)/32000:.0f}s | {len(EXPECTED)} terminos tecnicos esperados\n")

    rows = []
    for model in MODELS:
        for name, cfg in configs().items():
            r = await run_one(client, model, name, cfg, pcm)
            if r["ok"]:
                n, missing = score(r["text"])
                r["hits"], r["missing"] = n, missing
                print(f"  OK  {model:38} {name:26} ttft={r['ttft']:>5}ms  jerga {n}/{len(EXPECTED)}")
            else:
                r["hits"], r["missing"] = -1, []
                print(f"  --  {model:38} {name:26} {r['err']}")
            rows.append(r)

    good = [r for r in rows if r["ok"]]
    if not good:
        print("\nNinguna combinacion funciono.")
        return 1

    good.sort(key=lambda r: (-r["hits"], r["ttft"] or 9999))
    best = good[0]
    print(f"\n{'='*78}\nMEJOR: {best['model']}  /  {best['cfg']}")
    print(f"  ttft {best['ttft']} ms | jerga {best['hits']}/{len(EXPECTED)}"
          + (f" | falta: {', '.join(best['missing'])}" if best["missing"] else " | sin fallos"))
    print(f"  transcripcion: {best['text'][:400]}")

    by_cfg: dict[str, list[int]] = {}
    for r in good:
        by_cfg.setdefault(r["cfg"], []).append(r["hits"])
    print("\nAporte del glosario (promedio de aciertos por configuracion):")
    for name, hits in by_cfg.items():
        print(f"  {name:26} {sum(hits)/len(hits):.1f}/{len(EXPECTED)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
