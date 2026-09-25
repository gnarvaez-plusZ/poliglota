"""Compuerta de voz: deja pasar el audio con habla y retiene el silencio.

La API de transcripcion se paga por audio enviado. En una conferencia una sala
esta abierta muchas mas horas de las que alguien habla: el armado, el cambio de
orador, el coffee break, la espera antes de empezar. Todo eso son minutos
facturados por transcribir nada.

Es una compuerta por energia, no un VAD neuronal. Silero seria mas preciso con
ruido de fondo, pero arrastra PyTorch a una imagen que hoy pesa poco y decide en
milisegundos algo que en una sala de conferencia con microfono de solapa es
bastante evidente. Si hace falta mas precision, este es el punto de reemplazo.

Dos cuidados para no cortar palabras, que es lo unico inaceptable aca:

* **Preroll**: se guardan los ultimos milisegundos de silencio y se envian junto
  con el primer fragmento de voz. Sin esto se come el ataque de la primera
  silaba y "Kubernetes" llega como "ubernetes".
* **Hangover**: tras la ultima energia alta se sigue enviando un rato. Una pausa
  breve dentro de una frase no debe cortar el envio, porque el reconocedor usa
  ese contexto para cerrar la oracion.

Ante la duda la compuerta se abre: perder audio cuesta precision, y la precision
vale mas que los tokens que ahorra.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class SpeechGate:
    def __init__(
        self,
        sample_rate: int = 16000,
        sample_width: int = 2,
        threshold: float = 0.006,
        preroll_ms: int = 400,
        hangover_ms: int = 1200,
    ) -> None:
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.threshold = threshold
        self.hangover_s = hangover_ms / 1000
        self._preroll: deque[bytes] = deque()
        self._preroll_bytes = 0
        self._preroll_cap = int(sample_rate * sample_width * preroll_ms / 1000)
        self._quiet_s = 0.0
        self.bytes_in = 0
        self.bytes_out = 0

    @property
    def saved_ratio(self) -> float:
        """Fraccion de audio que no se envio. 0 si todavia no paso nada."""
        if not self.bytes_in:
            return 0.0
        return max(0.0, 1.0 - self.bytes_out / self.bytes_in)

    def feed(self, pcm: bytes) -> list[bytes]:
        """Devuelve los fragmentos a enviar: vacio si es silencio que se retiene."""
        self.bytes_in += len(pcm)
        if not pcm:
            return []

        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return []
        rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
        duration = len(pcm) / (self.sample_rate * self.sample_width)

        if rms >= self.threshold:
            self._quiet_s = 0.0
            out = [*self._preroll, pcm]
            self._preroll.clear()
            self._preroll_bytes = 0
        else:
            self._quiet_s += duration
            if self._quiet_s <= self.hangover_s:
                # Pausa corta dentro de una frase: se sigue enviando.
                out = [pcm]
            else:
                self._remember(pcm)
                out = []

        self.bytes_out += sum(len(c) for c in out)
        return out

    def _remember(self, pcm: bytes) -> None:
        """Guarda el silencio reciente por si enseguida empieza a hablar."""
        self._preroll.append(pcm)
        self._preroll_bytes += len(pcm)
        while self._preroll_bytes > self._preroll_cap and len(self._preroll) > 1:
            self._preroll_bytes -= len(self._preroll.popleft())
