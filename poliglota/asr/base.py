"""Contrato comun de los motores de ASR.

Un motor consume un stream de PCM16 16 kHz mono y emite `Transcript`s. Cada
`Transcript` trae SIEMPRE el texto completo de la frase en curso, no un trozo:
acumular es responsabilidad del motor, que es quien conoce la semantica de su
propio protocolo. La sala solo reemplaza lo que muestra y, cuando llega un final,
lo confirma. Asi se puede cambiar de motor sin tocar el pipeline.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Transcript:
    # Texto completo de la frase en curso (acumulado por el motor).
    text: str
    is_final: bool
    # Latencia real de subtitulado: cuanto quedo el texto por detras del audio
    # que ya habia entrado al sistema cuando el motor lo emitio.
    lag_ms: int = 0
    lang: str = ""
    # Etiqueta de hablante cuando el motor hace diarizacion. En un panel o en
    # las preguntas del publico, saber quien habla cambia la lectura.
    speaker: str = ""


@dataclass
class Usage:
    """Consumo real reportado por el motor, para contabilidad de costos."""

    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0


class ASREngine(Protocol):
    name: str

    async def stream(
        self,
        audio: asyncio.Queue,
        out: asyncio.Queue,
        *,
        source_lang: str = "en",
        hints: str = "",
    ) -> None:
        """Lee chunks PCM16 de `audio` hasta recibir None y publica en `out`.

        Publica `Transcript` por cada revision del subtitulo, `Usage` cuando el
        motor informa consumo, y `None` al terminar.
        """
        ...
