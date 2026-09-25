"""Verifica los idiomas: transcripcion en es/en/pt y traduccion en ambos sentidos.

El reto pide como minimo transcribir y traducir de ingles a espanol, pero una
conferencia latinoamericana necesita lo contrario mucho mas seguido. Ademas
prueba `language_auto` contra audio mezclado: el orador habla espanol y dice la
jerga en ingles sin traducirla, que es donde un reconocedor con un solo idioma
fijado se equivoca.

Necesita las pistas de `samples/`. Generarlas con:

    for L in en es pt mixto; do
      scripts/make_sample.py samples/charla-$L.wav --lang $L
      ffmpeg -i samples/charla-$L.wav -ac 1 -ar 16000 -f s16le samples/charla-$L.raw
    done
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

from poliglota.config import settings
from poliglota.translate import build_translator

ROOT = Path(__file__).resolve().parent.parent
CHUNK = 3200
WINDOW = 30  # segundos de audio por caso

# Pista, idioma declarado, y palabras que el orador dice DENTRO de la ventana.
ASR_CASES = [
    ("charla-en",    "en",   ["retrieval", "augmented", "vector", "embedding"]),
    ("charla-es",    "es",   ["pipeline", "recuperación", "vectorial", "latencia"]),
    ("charla-pt",    "pt",   ["pipeline", "recuperação", "vetorial", "latência"]),
    ("charla-mixto", "es",   ["pipeline", "retrieval", "bottleneck", "vector"]),
    ("charla-mixto", "auto", ["pipeline", "retrieval", "bottleneck", "vector"]),
]

# Linea, idioma origen, destinos, y jerga que NO debe traducirse.
MT_CASES = [
    ("es", ["en", "pt"],
     "Movimos el paso de embeddings atrás de un tópico de Kafka y la latencia p99 bajó a noventa milisegundos.",
     ["Kafka", "embedding", "p99"]),
    ("pt", ["es", "en"],
     "Tudo isso roda em Kubernetes, com um autoscaler horizontal guiado por profundidade de fila.",
     ["Kubernetes", "autoscaler"]),
    ("en", ["es", "pt"],
     "We trace every hop with eBPF and export the spans over gRPC to Grafana Tempo.",
     ["eBPF", "gRPC", "Grafana"]),
]

GLOSSARY = "Kafka, Kubernetes, embeddings, p99, eBPF, gRPC, Grafana Tempo, autoscaler"


def _config(lang: str) -> types.LiveConnectConfig:
    tc = types.AudioTranscriptionConfig()
    if lang == "auto":
        tc.language_auto = types.LanguageAuto()
    else:
        tc.language_hints = types.LanguageHints(language_codes=[lang])
    return types.LiveConnectConfig(input_audio_transcription=tc)


async def transcribe(path: Path, lang: str) -> tuple[str, float]:
    pcm = path.read_bytes()[: 16000 * 2 * WINDOW]
    client = genai.Client(api_key=settings.require_key())
    best, ttft, t0 = "", 0.0, time.time()
    try:
        async with asyncio.timeout(WINDOW + 30):
            async with client.aio.live.connect(model=settings.live_model, config=_config(lang)) as s:
                async def send():
                    for i in range(0, len(pcm), CHUNK):
                        await s.send_realtime_input(
                            audio=types.Blob(data=pcm[i:i + CHUNK], mime_type="audio/pcm;rate=16000")
                        )
                        await asyncio.sleep(0.1)
                    await s.send_realtime_input(audio_stream_end=True)

                pump = asyncio.create_task(send())
                async for m in s.receive():
                    sc = m.server_content
                    if not sc:
                        continue
                    for field in ("interim_input_transcription", "input_transcription"):
                        t = getattr(sc, field, None)
                        if t and t.text:
                            ttft = ttft or time.time() - t0
                            best = t.text if len(t.text) > len(best) else best
                    if sc.generation_complete and pump.done():
                        break
                pump.cancel()
                await asyncio.gather(pump, return_exceptions=True)
    except TimeoutError:
        pass
    return best, ttft


async def main() -> int:
    fails: list[str] = []

    print("== Transcripcion por idioma ==")
    for name, lang, expected in ASR_CASES:
        path = ROOT / "samples" / f"{name}.raw"
        if not path.exists():
            print(f"  --    {name:14} ({lang:4}) falta {path.name}; ver el encabezado")
            continue
        text, ttft = await transcribe(path, lang)
        low = re.sub(r"[^a-z0-9áéíóúñçãõâê ]", " ", text.lower())
        hits = [w for w in expected if w in low]
        # Se exige la mitad: la API varia bastante corrida a corrida y una
        # ventana corta puede cortar la frase antes de que diga el termino.
        ok = len(hits) >= max(2, len(expected) // 2)
        print(f"  {'PASA ' if ok else 'FALLA'} {name:14} ({lang:4}) {len(hits)}/{len(expected)} "
              f"terminos  1er texto {ttft:.1f}s")
        print(f"        {text[:120]}")
        if not ok:
            fails.append(f"asr:{name}/{lang}")

    print("\n== Traduccion en ambos sentidos ==")
    translator = build_translator("gemini")
    for src, targets, line, keep in MT_CASES:
        t0 = time.time()
        res = await translator.translate(line, source_lang=src, targets=targets,
                                         context="Charla tecnica sobre escalar RAG en produccion.",
                                         glossary=GLOSSARY)
        ms = int((time.time() - t0) * 1000)
        if not res.texts:
            print(f"  FALLA {src}->{'+'.join(targets)}: sin traduccion")
            fails.append(f"mt:{src}")
            continue
        for code, out in res.texts.items():
            kept = [w for w in keep if w.lower() in out.lower()]
            ok = len(kept) == len(keep)
            print(f"  {'PASA ' if ok else 'FALLA'} {src}->{code}  jerga {len(kept)}/{len(keep)}  [{ms}ms]")
            print(f"        {out[:120]}")
            if not ok:
                fails.append(f"mt:{src}->{code}")

    print(f"\n{'IDIOMAS OK' if not fails else f'{len(fails)} FALLAS: {fails}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
