"""Tipos de datos que viajan entre el pipeline y los clientes.

Un `Segment` es la unidad de subtitulado: nace como parcial (el ASR todavia esta
escuchando) y se reemite como final cuando el motor cierra la frase. Los clientes
reemplazan el parcial en pantalla usando `seq` como clave.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Segment:
    room_id: str
    seq: int
    text: str
    lang: str
    is_final: bool = False
    # Momento en que entro al servidor el audio que origino este texto. Es el
    # ancla para medir latencia punta a punta de forma honesta.
    t_audio: float = field(default_factory=time.time)
    t_asr: float = 0.0
    translations: dict[str, str] = field(default_factory=dict)

    @property
    def asr_latency_ms(self) -> int:
        if not self.t_asr:
            return 0
        return int((self.t_asr - self.t_audio) * 1000)

    def wire(self, extra: dict | None = None) -> dict:
        payload = {
            "type": "segment",
            "room": self.room_id,
            "seq": self.seq,
            "final": self.is_final,
            "lang": self.lang,
            "text": self.text,
            "tr": self.translations,
            "asr_ms": self.asr_latency_ms,
            "t": self.t_audio,
        }
        if extra:
            payload.update(extra)
        return payload
