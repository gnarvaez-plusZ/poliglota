"""ASR en streaming contra la Live API de Gemini.

Lo que hace la diferencia en una conferencia real, y como se resuelve aca:

* **Jerga tecnica.** La Live API acepta `adaptation_phrases`: el glosario de la
  charla se pasa como sesgo de vocabulario del reconocedor, no como una sugerencia
  en el prompt. Es la diferencia entre "cuber netes" y "Kubernetes".
* **Charlas mas largas que la sesion.** El limite de audio de una sesion es de 15
  minutos; una charla dura 45. `context_window_compression` con ventana deslizante
  la estira, y `session_resumption` permite reconectar con el hilo intacto cuando
  el servidor manda `GoAway`.
* **Latencia.** Se consumen dos streams distintos: `interim_input_transcription`
  es la hipotesis en curso, que se pinta ya; `input_transcription` es el texto
  confirmado. El publico ve las palabras casi en vivo y despues se corrigen solas.
* **Quien habla.** Con `diarization` activada, las preguntas del publico y los
  paneles quedan atribuidos en vez de fundirse en un unico bloque de texto.
"""
from __future__ import annotations

import asyncio
import logging
import time

from google import genai
from google.genai import types

from ..config import settings
from .base import Transcript, Usage

log = logging.getLogger("poliglota.asr.gemini")

# El modelo no tiene que conversar: solo prestamos su oido. La instruccion lo
# mantiene callado para no pagar tokens de una respuesta que descartamos.
_SYSTEM = (
    "You are a silent audio relay inside a live conference captioning system. "
    "Never speak, never answer, never comment on the audio. Always reply with a single period."
)

_MAX_PHRASES = 250  # tope defensivo para el sesgo de vocabulario


def _merge(acc: str, new: str) -> str:
    """Une texto confirmado sin depender de si el servidor manda trozos o acumulado."""
    new = new.strip()
    if not new:
        return acc
    if not acc:
        return new
    if new.startswith(acc):
        return new
    if acc.endswith(new):
        return acc
    return f"{acc} {new}".strip()


class GeminiLiveASR:
    name = "gemini-live"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.live_model
        self._client = genai.Client(api_key=settings.require_key())

    def _config(self, source_lang: str, hints: str, handle: str | None) -> types.LiveConnectConfig:
        phrases = [p.strip() for p in hints.split(",") if p.strip()][:_MAX_PHRASES]

        transcription = types.AudioTranscriptionConfig(
            # SMART agrega puntuacion y limpia muletillas: un subtitulo se lee,
            # no se audita. VERBATIM serviria para una transcripcion legal.
            mode=types.AudioTranscriptionConfigMode.SMART,
            diarization=True,
            word_timestamp=False,
        )
        if phrases:
            transcription.adaptation_phrases = phrases
        if source_lang == "auto":
            # Una conferencia latinoamericana alterna espanol e ingles, a veces
            # dentro de la misma charla. Dejamos que el modelo lo resuelva.
            transcription.language_auto = types.LanguageAuto()
        else:
            transcription.language_hints = types.LanguageHints(language_codes=[source_lang])

        return types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=transcription,
            system_instruction=_SYSTEM,
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            session_resumption=types.SessionResumptionConfig(handle=handle),
        )

    async def stream(
        self,
        audio: asyncio.Queue,
        out: asyncio.Queue,
        *,
        source_lang: str = "en",
        hints: str = "",
    ) -> None:
        handle: str | None = None
        stop = False
        # Marca del ultimo audio entregado al modelo: el ancla honesta para medir
        # cuanto viene atras el subtitulo respecto del orador.
        clock = {"last_audio": time.time()}

        while not stop:
            confirmed = ""
            interim = ""
            speaker = ""
            try:
                cfg = self._config(source_lang, hints, handle)
                async with self._client.aio.live.connect(model=self.model, config=cfg) as session:
                    log.info(
                        "Live API conectada (modelo=%s, reanudada=%s, frases=%s)",
                        self.model,
                        bool(handle),
                        len(cfg.input_audio_transcription.adaptation_phrases or []),
                    )

                    async def pump() -> None:
                        nonlocal stop
                        while True:
                            chunk = await audio.get()
                            if chunk is None:
                                stop = True
                                await session.send_realtime_input(audio_stream_end=True)
                                return
                            clock["last_audio"] = time.time()
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=chunk,
                                    mime_type=f"audio/pcm;rate={settings.sample_rate}",
                                )
                            )

                    pump_task = asyncio.create_task(pump())

                    async def emit(final: bool) -> None:
                        text = _merge(confirmed, interim) if not final else confirmed
                        if not text.strip():
                            return
                        await out.put(
                            Transcript(
                                text=text.strip(),
                                is_final=final,
                                lag_ms=max(int((time.time() - clock["last_audio"]) * 1000), 0),
                                lang=source_lang,
                                speaker=speaker,
                            )
                        )

                    try:
                        async for message in session.receive():
                            if (upd := message.session_resumption_update) and upd.new_handle:
                                handle = upd.new_handle

                            if (usage := message.usage_metadata) is not None:
                                await out.put(
                                    Usage(
                                        prompt_tokens=usage.prompt_token_count or 0,
                                        response_tokens=usage.response_token_count or 0,
                                        total_tokens=usage.total_token_count or 0,
                                    )
                                )

                            if (away := message.go_away) is not None:
                                log.info("GoAway: quedan %s; se reconectara", away.time_left)

                            sc = message.server_content
                            if sc is None:
                                continue

                            if (t := sc.interim_input_transcription) and t.text:
                                interim = t.text
                                speaker = t.speaker_label or speaker
                                await emit(final=False)

                            if (t := sc.input_transcription) and t.text:
                                confirmed = _merge(confirmed, t.text)
                                interim = ""
                                speaker = t.speaker_label or speaker
                                await emit(final=False)
                                if t.finished:
                                    await emit(final=True)
                                    confirmed = speaker = ""

                            if sc.turn_complete and confirmed.strip():
                                await emit(final=True)
                                confirmed = interim = speaker = ""
                    finally:
                        pump_task.cancel()
                        await asyncio.gather(pump_task, return_exceptions=True)

                if stop:
                    break
                log.warning("Sesion Live cerrada por el servidor; reconectando")

            except asyncio.CancelledError:
                raise
            except Exception:
                if stop:
                    break
                log.exception("Error en la sesion Live; reintento en 1s")
                await asyncio.sleep(1.0)

        await out.put(None)
