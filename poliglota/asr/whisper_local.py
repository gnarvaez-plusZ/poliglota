"""ASR local con faster-whisper: despliegue sin nube, sin API key y con costo cero.

Sirve para dos escenarios reales de conferencia: sedes con internet inestable y
organizadores que no quieren gasto variable por sala. La contra es que la latencia
pasa a depender del hardware.

Estrategia de streaming: acumulamos la frase en curso, re-transcribimos la ventana
completa cada `refresh` segundos para emitir parciales, y cerramos el segmento
cuando el nivel de senal cae (silencio) o cuando la frase se vuelve demasiado larga.
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from ..config import settings
from .base import Transcript

log = logging.getLogger("poliglota.asr.whisper")

_REFRESH_S = 0.9        # cada cuanto re-transcribimos para refrescar el parcial
_SILENCE_RMS = 0.006    # umbral de energia por debajo del cual consideramos silencio
_SILENCE_HOLD_S = 0.7   # cuanto silencio cierra la frase
_MAX_UTTERANCE_S = 14.0 # corte duro para que un orador continuo no crezca sin limite


def _pcm_to_float(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0


class WhisperASR:
    name = "faster-whisper"

    def __init__(self, model: str | None = None) -> None:
        from faster_whisper import WhisperModel

        name = model or settings.whisper_model
        device = settings.whisper_device
        if device == "auto":
            try:
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        compute = "float16" if device == "cuda" else "int8"
        log.info("Cargando faster-whisper %s en %s (%s)", name, device, compute)
        self._model = WhisperModel(name, device=device, compute_type=compute)

    def _transcribe(self, audio: np.ndarray, lang: str, hints: str) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language=lang or None,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=hints or None,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def stream(
        self,
        audio: asyncio.Queue,
        out: asyncio.Queue,
        *,
        source_lang: str = "en",
        hints: str = "",
    ) -> None:
        loop = asyncio.get_running_loop()
        utterance = bytearray()
        last_refresh = 0.0
        silent_since: float | None = None
        last_audio_ts = time.time()
        bytes_per_s = settings.sample_rate * settings.sample_width

        async def flush(final: bool) -> None:
            nonlocal last_refresh
            if len(utterance) < bytes_per_s // 3:
                return
            samples = _pcm_to_float(bytes(utterance))
            text = await loop.run_in_executor(
                None, self._transcribe, samples, source_lang, hints
            )
            last_refresh = time.time()
            if not text:
                return
            await out.put(
                Transcript(
                    text=text,
                    is_final=final,
                    lag_ms=max(int((time.time() - last_audio_ts) * 1000), 0),
                    lang=source_lang,
                )
            )

        while True:
            chunk = await audio.get()
            if chunk is None:
                await flush(final=True)
                break

            last_audio_ts = time.time()
            utterance.extend(chunk)

            rms = float(np.sqrt(np.mean(np.square(_pcm_to_float(chunk)))))
            now = time.time()

            if rms < _SILENCE_RMS:
                silent_since = silent_since or now
            else:
                silent_since = None

            too_long = len(utterance) > _MAX_UTTERANCE_S * bytes_per_s
            ended = silent_since is not None and (now - silent_since) >= _SILENCE_HOLD_S

            if (ended and len(utterance) > bytes_per_s // 2) or too_long:
                await flush(final=True)
                utterance.clear()
                silent_since = None
            elif now - last_refresh >= _REFRESH_S:
                await flush(final=False)

        await out.put(None)
