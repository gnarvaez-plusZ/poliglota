"""Telemetria por sala.

La hackathon evalua latencia y escalabilidad, asi que el sistema mide ambas y las
muestra en vivo en vez de pedir que se confie en la demo. Los percentiles se
calculan sobre una ventana movil para que reflejen el estado actual de la sala.
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
        # Tokens informados por el motor, no estimados: es lo que se factura.
        self.asr_tokens = 0
        self.mt_tokens = 0
        self._asr_lag: deque[int] = deque(maxlen=window)
        self._mt_ms: deque[int] = deque(maxlen=window)

    def note_audio(self, n_bytes: int, sample_rate: int, sample_width: int) -> None:
        self.audio_seconds += n_bytes / (sample_rate * sample_width)

    def note_asr(self, lag_ms: int) -> None:
        if lag_ms > 0:
            self._asr_lag.append(lag_ms)

    def note_asr_usage(self, total_tokens: int) -> None:
        self.asr_tokens += total_tokens

    def note_mt_usage(self, total_tokens: int) -> None:
        self.mt_tokens += total_tokens

    def note_mt(self, ms: int, ok: bool) -> None:
        self.translation_calls += 1
        if ok:
            self._mt_ms.append(ms)
        else:
            self.translation_failures += 1

    @staticmethod
    def _pct(values: deque[int], q: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        idx = min(int(q * len(ordered)), len(ordered) - 1)
        return ordered[idx]

    def snapshot(self) -> dict:
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "audio_s": round(self.audio_seconds, 1),
            "segments": self.segments_final,
            "asr_lag_p50": self._pct(self._asr_lag, 0.50),
            "asr_lag_p95": self._pct(self._asr_lag, 0.95),
            "mt_p50": self._pct(self._mt_ms, 0.50),
            "mt_p95": self._pct(self._mt_ms, 0.95),
            "mt_calls": self.translation_calls,
            "mt_failures": self.translation_failures,
            "asr_tokens": self.asr_tokens,
            "mt_tokens": self.mt_tokens,
            "tokens_per_audio_min": round(
                (self.asr_tokens + self.mt_tokens) / max(self.audio_seconds / 60, 1e-9)
            ) if self.audio_seconds > 5 else 0,
        }
