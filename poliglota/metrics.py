"""Telemetria por sala.

Nota sobre que se mide y que no. El modelo de transcripcion acepta
`word_timestamp` pero nunca devuelve los tiempos por palabra, asi que no hay
forma de alinear una palabra del subtitulo con el instante del audio que la
origino. Sin esa alineacion, cualquier "retraso punta a punta" mostrado en vivo
seria una invencion.

Por eso el panel reporta solo cantidades que se miden de verdad:

* `refresh_ms`  : cada cuanto se actualiza el subtitulo en pantalla. Es lo que
                  la audiencia percibe como fluidez.
* `ttft_ms`     : cuanto tardo en aparecer el primer texto desde que entro audio.
* `mt_ms`       : cuanto tarda la traduccion, medido sobre la llamada real.
* tokens        : consumo informado por el motor, no estimado.

Los numeros de referencia con una sola sala (2,0 s hasta el primer texto,
refresco cada ~490 ms) estan medidos y documentados en el README.
"""
from __future__ import annotations

import time
from collections import deque


class RoomMetrics:
    def __init__(self, window: int = 120) -> None:
        self.started = time.time()
        self.audio_seconds = 0.0
        self.segments_final = 0
        self.translation_calls = 0
        self.translation_failures = 0
        self.translation_retries = 0
        # Audio descartado por atraso del motor. Si sube, el subtitulo
        # se esta quedando atras y el operador tiene que enterarse.
        self.dropped_chunks = 0
        # Tokens informados por el motor, no estimados: es lo que se factura.
        self.asr_tokens = 0
        self.mt_tokens = 0

        self._refresh: deque[int] = deque(maxlen=window)
        self._mt_ms: deque[int] = deque(maxlen=window)
        self._first_audio: float | None = None
        self.ttft_ms: int = 0
        self._last_update: float | None = None

    def note_audio(self, n_bytes: int, sample_rate: int, sample_width: int) -> None:
        if self._first_audio is None:
            self._first_audio = time.time()
        self.audio_seconds += n_bytes / (sample_rate * sample_width)

    def note_subtitle(self) -> None:
        """Una actualizacion de subtitulo llego a los espectadores."""
        now = time.time()
        if not self.ttft_ms and self._first_audio:
            self.ttft_ms = int((now - self._first_audio) * 1000)
        if self._last_update is not None:
            gap = int((now - self._last_update) * 1000)
            # Descartamos los huecos largos: son pausas del orador, no lentitud
            # del sistema, y ensuciarian la mediana.
            if gap < 5000:
                self._refresh.append(gap)
        self._last_update = now

    def note_asr_usage(self, total_tokens: int) -> None:
        self.asr_tokens += total_tokens

    def note_mt_usage(self, total_tokens: int) -> None:
        self.mt_tokens += total_tokens

    def note_mt(self, ms: int, ok: bool, retries: int = 0) -> None:
        self.translation_calls += 1
        self.translation_retries += retries
        if ok:
            self._mt_ms.append(ms)
        else:
            self.translation_failures += 1

    @staticmethod
    def _pct(values: deque[int], q: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        return ordered[min(int(q * len(ordered)), len(ordered) - 1)]

    def snapshot(self) -> dict:
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "audio_s": round(self.audio_seconds, 1),
            "segments": self.segments_final,
            "refresh_p50": self._pct(self._refresh, 0.50),
            "refresh_p95": self._pct(self._refresh, 0.95),
            "ttft_ms": self.ttft_ms,
            "mt_p50": self._pct(self._mt_ms, 0.50),
            "mt_p95": self._pct(self._mt_ms, 0.95),
            "mt_calls": self.translation_calls,
            "mt_failures": self.translation_failures,
            "mt_retries": self.translation_retries,
            "dropped_chunks": self.dropped_chunks,
            "asr_tokens": self.asr_tokens,
            "mt_tokens": self.mt_tokens,
            "tokens_per_audio_min": (
                round((self.asr_tokens + self.mt_tokens) / max(self.audio_seconds / 60, 1e-9))
                if self.audio_seconds > 5 else 0
            ),
        }
