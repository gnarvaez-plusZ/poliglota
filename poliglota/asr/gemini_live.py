"""ASR en streaming contra gemini-3.5-transcribe-live.

Todo lo que sigue sale de medir contra la API, no de la documentacion:

* El modelo dedicado a transcripcion es el unico que acepta una sesion sin
  modalidad de audio de salida. Los modelos conversacionales rechazan
  `response_modalities=["TEXT"]`, asi que usarlos obligaria a pagar por un audio
  de respuesta que descartariamos.
* `interim_input_transcription` es el stream en vivo y es ACUMULATIVO: cada
  mensaje trae la transcripcion completa de la sesion, no un trozo. Se refresca
  cada ~490 ms. `input_transcription` llega mas espaciado y es el texto confirmado.
* La configuracion minima gana. Medido sobre una charla tecnica, la config por
  defecto acerto 10 de 10 terminos de jerga (Kubernetes, Kafka, eBPF, gRPC, P99)
  con 1.7 s hasta el primer texto. Agregar `adaptation_phrases` no mejoro la
  precision y triplico la latencia inicial; activar `diarization` multiplico por
  ocho el intervalo de refresco. Por eso ambas quedan desactivadas por defecto y
  se ofrecen como opcion para nombres propios que el modelo no conozca.
* El modelo no marca fin de frase: no emite `turn_complete` por oracion. La
  segmentacion en subtitulos la hacemos nosotros sobre el texto acumulado.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time

from google import genai
from google.genai import types

from ..config import settings
from .base import Transcript, Usage

log = logging.getLogger("poliglota.asr.gemini")

# Cierre de oracion seguido de espacio o fin. Suficiente para subtitular; no
# pretende resolver abreviaturas.
_SENTENCE_END = re.compile(r'(?<=[.!?])[\s"\')\]]*(?=\s|$)')
# Un orador que no hace pausas no puede producir una linea infinita: por encima
# de este largo cortamos en el ultimo limite natural que haya.
_MAX_LINE = 180
_SOFT_BREAK = re.compile(r"(?<=[,;:])\s")


def split_sentences(text: str) -> tuple[list[str], str, int]:
    """Separa el acumulado en oraciones cerradas, el resto en curso, y cuantos
    caracteres del original consumieron las cerradas.

    Devolver el desplazamiento evita tener que buscar cada oracion por contenido
    para saber por donde seguir: un orador que repite una frase haria que la
    busqueda casara la aparicion anterior y el subtitulo retrocederia.
    """
    if not text:
        return [], "", 0
    done: list[str] = []
    start = 0
    consumed = 0
    for m in _SENTENCE_END.finditer(text):
        piece = text[start:m.end()].strip()
        if piece:
            done.append(piece)
            start = m.end()
            consumed = m.end()
    tail = text[start:].strip()

    # La cola crecio demasiado sin punto final: la cortamos en la ultima coma.
    while len(tail) > _MAX_LINE:
        breaks = [b.end() for b in _SOFT_BREAK.finditer(tail[:_MAX_LINE])]
        cut = breaks[-1] if breaks else tail.rfind(" ", 0, _MAX_LINE)
        if cut <= 0:
            break
        done.append(tail[:cut].strip())
        tail = tail[cut:].strip()
        consumed = len(text) - len(tail)
    return done, tail, consumed


class GeminiLiveASR:
    name = "gemini-live"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.live_model
        self._client = genai.Client(api_key=settings.require_key())

    def _config(self, source_lang: str, hints: str) -> types.LiveConnectConfig:
        tc = types.AudioTranscriptionConfig()
        if source_lang and source_lang != "auto":
            tc.language_hints = types.LanguageHints(language_codes=[source_lang])
        elif source_lang == "auto":
            tc.language_auto = types.LanguageAuto()

        # Desactivado por defecto: medido, no ayuda y cuesta latencia. Se habilita
        # para charlas con nombres propios raros que el modelo no pueda conocer.
        if settings.asr_adaptation and hints:
            phrases = [p.strip() for p in hints.split(",") if p.strip()][:200]
            if phrases:
                tc.adaptation_phrases = phrases
        # Incompatible con adaptation_phrases, y multiplica por ocho el refresco.
        if settings.asr_diarization and not tc.adaptation_phrases:
            tc.diarization = True

        return types.LiveConnectConfig(
            input_audio_transcription=tc,
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            session_resumption=types.SessionResumptionConfig(),
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
        clock = {"last_audio": time.time()}

        # Indice de la primera linea de esta sesion. Al reconectar, el modelo
        # vuelve a numerar desde cero su acumulado, pero la charla sigue: las
        # lineas nuevas tienen que continuar la numeracion, no pisarla.
        base = 0

        # Primer fragmento de audio de cada sesion. Se lo espera ANTES de
        # conectar: el servidor aborta con `1008 policy violation` una sesion que
        # no recibe audio, y medido tarda unos 60 segundos en hacerlo. Una sala
        # creada antes de que empiece la charla se pasaba conectando y muriendo
        # en vacio, y si el orador arrancaba justo en una ventana muerta el
        # primer subtitulo llegaba 17 segundos tarde en vez de 2.
        first: bytes | None = None

        while not stop:
            # Ultimo texto que mando el modelo y ultima version entregada de cada
            # linea. El modelo REVISA lo que ya dijo, asi que una linea puede
            # cambiar despues de haberse emitido.
            sent: dict[int, str] = {}
            speaker = ""

            if first is None:
                first = await audio.get()
                if first is None:
                    break

            try:
                cfg = self._config(source_lang, hints)
                if handle:
                    cfg.session_resumption = types.SessionResumptionConfig(handle=handle)

                async with self._client.aio.live.connect(model=self.model, config=cfg) as session:
                    log.info("transcribe-live conectado (%s, reanudado=%s)", self.model, bool(handle))

                    async def pump() -> None:
                        nonlocal stop, first
                        chunk = first
                        first = None
                        while True:
                            if chunk is None:
                                chunk = await audio.get()
                            if chunk is None:
                                stop = True
                                await session.send_realtime_input(audio_stream_end=True)
                                return
                            clock["last_audio"] = time.time()
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=chunk, mime_type=f"audio/pcm;rate={settings.sample_rate}"
                                )
                            )
                            chunk = None

                    pump_task = asyncio.create_task(pump())

                    async def push(full_text: str) -> None:
                        """Convierte el acumulado del modelo en lineas de subtitulo.

                        El modelo no solo agrega texto: reescribe lo que ya dijo
                        cuando se corrige. Por eso cada oracion se identifica por
                        su POSICION en la charla y no por su contenido: si cambia,
                        se reemite con el mismo indice y el espectador ve que la
                        linea se corrige sola, en vez de verla aparecer dos veces.
                        """
                        sentences, tail, _ = split_sentences(full_text)
                        lag = max(int((time.time() - clock["last_audio"]) * 1000), 0)

                        for i, sentence in enumerate(sentences):
                            if sent.get(i) == sentence:
                                continue  # sin cambios: no molestamos al espectador
                            sent[i] = sentence
                            await out.put(
                                Transcript(text=sentence, is_final=True, lag_ms=lag,
                                           lang=source_lang, index=base + i, speaker=speaker)
                            )
                        if tail:
                            await out.put(
                                Transcript(text=tail, is_final=False, lag_ms=lag,
                                           lang=source_lang, index=base + len(sentences),
                                           speaker=speaker)
                            )

                    try:
                        async for message in session.receive():
                            if (upd := message.session_resumption_update) and upd.new_handle:
                                handle = upd.new_handle
                            if (u := message.usage_metadata) is not None:
                                await out.put(Usage(total_tokens=u.total_token_count or 0))
                            if (away := message.go_away) is not None:
                                log.info("GoAway: quedan %s; se reconectara", away.time_left)

                            sc = message.server_content
                            if sc is None:
                                continue
                            if (t := sc.interim_input_transcription) and t.text:
                                speaker = t.speaker_label or speaker
                                await push(t.text)
                            if (t := sc.input_transcription) and t.text:
                                speaker = t.speaker_label or speaker
                                await push(t.text)
                    finally:
                        pump_task.cancel()
                        await asyncio.gather(pump_task, return_exceptions=True)
                        # La proxima sesion empieza a numerar donde termino esta.
                        base += len(sent)

                if stop:
                    break
                log.warning("Sesion cerrada por el servidor; reconectando desde la linea %d", base)
            except asyncio.CancelledError:
                raise
            except Exception:
                if stop:
                    break
                # `base` ya se avanzo en el `finally` de la sesion; si el fallo
                # fue al conectar, no hubo lineas y no hay nada que avanzar.
                log.exception("Error en la sesion; reintento en 1s")
                await asyncio.sleep(1.0)

        await out.put(None)
