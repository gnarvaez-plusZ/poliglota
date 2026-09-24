"""Motor de ASR falso: reproduce un guion escrito, palabra por palabra.

Existe por tres razones practicas. Permite probar el pipeline completo sin
gastar una sola llamada a la API, deja correr los tests en CI sin credenciales,
y salva la demo si el wifi del evento se cae en el peor momento.

Respeta el mismo contrato que los motores reales: emite parciales que crecen y
un final por frase, con la latencia tipica de un ASR en streaming.
"""
from __future__ import annotations

import asyncio
import time

from ..config import settings
from .base import Transcript

DEFAULT_SCRIPT = [
    "Good morning everyone, thanks for joining this talk about scaling retrieval augmented generation.",
    "The first thing we learned is that your vector database is almost never the bottleneck.",
    "In our case the real cost was the embedding step, which we were running synchronously.",
    "So we moved it to a Kafka queue and the p95 latency dropped from eight hundred milliseconds to ninety.",
    "The second lesson is about chunking. Everyone talks about chunk size, nobody talks about overlap.",
    "We run this on Kubernetes with a horizontal pod autoscaler driven by queue depth, not CPU.",
]


class MockASR:
    name = "mock"

    def __init__(self, script: list[str] | None = None, wpm: int = 150) -> None:
        self.script = script or DEFAULT_SCRIPT
        self._word_delay = 60.0 / wpm

    async def stream(
        self,
        audio: asyncio.Queue,
        out: asyncio.Queue,
        *,
        source_lang: str = "en",
        hints: str = "",
    ) -> None:
        drain = asyncio.create_task(self._drain(audio))
        try:
            for line in self.script:
                words: list[str] = []
                for word in line.split():
                    words.append(word)
                    await asyncio.sleep(self._word_delay)
                    await out.put(
                        Transcript(
                            text=" ".join(words),
                            is_final=False,
                            lag_ms=180,
                            lang=source_lang,
                        )
                    )
                await out.put(
                    Transcript(text=line, is_final=True, lag_ms=220, lang=source_lang)
                )
                await asyncio.sleep(0.6)
        except asyncio.CancelledError:
            raise
        finally:
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)
            await out.put(None)

    @staticmethod
    async def _drain(audio: asyncio.Queue) -> None:
        """Consume el audio que llegue para que la cola no se tape."""
        while True:
            chunk = await audio.get()
            if chunk is None:
                return
