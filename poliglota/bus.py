"""Pub/sub en memoria: un productor (el pipeline de la sala) y N consumidores (viewers).

Cada suscriptor tiene su propia cola acotada. Si un viewer se cuelga, se le
descartan los mensajes viejos en vez de frenar la sala entera: la latencia de
los demas nunca depende del cliente mas lento.
"""
from __future__ import annotations

import asyncio
from typing import Any


class Bus:
    def __init__(self, maxsize: int = 64) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._maxsize = maxsize

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    def publish(self, message: Any) -> None:
        for q in self._subs:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # Backpressure: tiramos el mas viejo, conservamos el mas nuevo.
                try:
                    q.get_nowait()
                    q.put_nowait(message)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
