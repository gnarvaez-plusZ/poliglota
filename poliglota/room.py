"""Una `Room` es una charla en vivo: una fuente de audio, un pipeline, N espectadores.

Decisiones de diseno que sostienen la latencia y el costo:

* El texto en el idioma original se publica APENAS sale del ASR, sin esperar la
  traduccion. La traduccion llega despues como un parche sobre el mismo `seq`.
  Asi el subtitulo aparece a la velocidad del ASR y no a la del traductor.
* Solo se traduce a los idiomas que tienen al menos un espectador conectado.
  Una sala sin publico frances no paga tokens de frances.
* Los parciales se traducen con estrangulador: cambian varias veces por segundo y
  no vale la pena pagar cada revision. Los finales siempre se traducen.
* Las salas son independientes entre si. Correr dos o diez es crear mas objetos
  `Room`: no hay estado compartido que las serialice.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import Counter, OrderedDict

from . import glossary
from .asr import build_asr
from .asr.base import Usage
from .bus import Bus
from .config import LANGUAGES, settings
from .gate import SpeechGate
from .speakers import SpeakerTracker, get_registry
from .metrics import RoomMetrics
from .translate import build_translator

log = logging.getLogger("poliglota.room")

_CONTEXT_SEGMENTS = 3        # cuantos finales previos viajan como contexto
_FINALS_MEMORY = 400         # lineas recordadas para detectar correcciones


class Room:
    def __init__(
        self,
        room_id: str,
        *,
        title: str = "",
        speakers: str = "",
        abstract: str = "",
        source_lang: str = "en",
        asr_engine: str | None = None,
        mt_engine: str | None = None,
    ) -> None:
        self.id = room_id
        self.title = title or room_id
        self.speakers = speakers
        self.abstract = abstract
        self.source_lang = source_lang
        self.created = time.time()

        self.bus = Bus()
        self.metrics = RoomMetrics()
        # Retiene el silencio para no pagar por transcribir nada. Ante la duda
        # abre: perder audio cuesta precision, y la precision vale mas.
        self.gate = SpeechGate(
            sample_rate=settings.sample_rate,
            sample_width=settings.sample_width,
            threshold=settings.vad_threshold,
        ) if settings.vad_gate else None
        # Quien habla: se decide localmente sobre el audio que ya pasa por la
        # sala, contra las huellas registradas. Sin huellas no hace nada.
        self.tracker = SpeakerTracker(get_registry())
        self.glossary = glossary.build(title=title, abstract=abstract, speakers=speakers)

        self._asr_name = asr_engine or settings.asr_engine
        self._mt_name = mt_engine or settings.mt_engine
        self._asr = None
        self._mt = None

        # 50 chunks = 5 segundos de audio. El tamano es una decision, no un
        # numero al azar: si el motor tarda en conectar o se atrasa, esta cola
        # es todo el pasado que el sistema puede llegar a mostrar. Con 256
        # chunks (25 s) una sesion lenta hacia que el subtitulo arrancara
        # transcribiendo medio minuto viejo. En subtitulado en vivo es preferible
        # perder una frase a quedar media charla atras del orador.
        self._audio: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._transcripts: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._tasks: list[asyncio.Task] = []
        # Traducciones en vuelo. Se rastrean para poder cancelarlas: una con
        # reintentos en curso dejaba el apagado colgado indefinidamente.
        self._jobs: set[asyncio.Task] = set()

        self._seq = 0
        self._partial = ""
        self._speaker = ""
        # Ultima version emitida de cada linea final. Permite detectar si el
        # motor corrigio una linea ya mostrada y no reemitir lo identico.
        self._finals: OrderedDict[int, str] = OrderedDict()
        self._speaker_by_seq: OrderedDict[int, str] = OrderedDict()
        self._history: list[str] = []
        self._last_partial_mt = 0.0
        self._partial_mt_inflight = False
        self._viewer_langs: Counter[str] = Counter()
        self._has_source = False
        self.running = False

    # ---------- ciclo de vida ----------

    async def start(self) -> None:
        if self.running:
            return
        self._asr = build_asr(self._asr_name)
        self._mt = build_translator(self._mt_name)
        self.running = True
        self._tasks = [
            asyncio.create_task(self._run_asr(), name=f"asr:{self.id}"),
            asyncio.create_task(self._consume(), name=f"consume:{self.id}"),
        ]
        log.info("Sala %s arriba (asr=%s, mt=%s)", self.id, self._asr.name, self._mt.name)
        self._publish_state()

    async def stop(self, timeout: float = 5.0) -> None:
        if not self.running:
            return
        self.running = False
        with contextlib.suppress(asyncio.QueueFull):
            self._audio.put_nowait(None)

        pending = [*self._tasks, *self._jobs]
        for task in pending:
            task.cancel()
        if pending:
            # Un motor remoto puede demorar en soltar su socket. Se le da un
            # margen y despues se sigue: apagar el evento no puede quedar a
            # merced de que un proveedor conteste.
            _, stuck = await asyncio.wait(pending, timeout=timeout)
            if stuck:
                log.warning("Sala %s: %d tarea(s) no cerraron en %.0fs", self.id, len(stuck), timeout)
        self._tasks.clear()
        self._jobs.clear()
        log.info("Sala %s detenida", self.id)

    async def _run_asr(self) -> None:
        try:
            await self._asr.stream(
                self._audio,
                self._transcripts,
                source_lang=self.source_lang,
                hints=self.glossary,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("El motor de ASR de la sala %s murio", self.id)
            self.bus.publish({"type": "error", "room": self.id, "detail": "asr_failed"})

    # ---------- entrada de audio ----------

    def feed(self, pcm: bytes) -> None:
        """Entrega audio PCM16 16 kHz mono al pipeline. No bloquea nunca."""
        if not self.running:
            return
        self._has_source = True
        self.metrics.note_audio(len(pcm), settings.sample_rate, settings.sample_width)
        self.tracker.feed(pcm)

        for piece in (self.gate.feed(pcm) if self.gate else [pcm]):
            self.metrics.note_sent(len(piece), settings.sample_rate, settings.sample_width)
            try:
                self._audio.put_nowait(piece)
            except asyncio.QueueFull:
                # Si el ASR se atrasa, preferimos perder audio viejo antes que
                # acumular un retraso que crece sin techo frente al orador.
                self.metrics.dropped_chunks += 1
                try:
                    self._audio.get_nowait()
                    self._audio.put_nowait(piece)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    # ---------- espectadores ----------

    def add_viewer(self, lang: str) -> asyncio.Queue:
        self._viewer_langs[lang] += 1
        q = self.bus.subscribe()
        self._publish_state()
        return q

    def remove_viewer(self, lang: str, q: asyncio.Queue) -> None:
        self.bus.unsubscribe(q)
        self._viewer_langs[lang] -= 1
        if self._viewer_langs[lang] <= 0:
            del self._viewer_langs[lang]
        self._publish_state()

    def switch_viewer_lang(self, old: str, new: str) -> None:
        """Cambia el idioma de un espectador ya conectado, sin reabrir el socket."""
        if old == new:
            return
        self._viewer_langs[old] -= 1
        if self._viewer_langs[old] <= 0:
            del self._viewer_langs[old]
        self._viewer_langs[new] += 1
        self._publish_state()

    def active_targets(self) -> list[str]:
        return [l for l in self._viewer_langs if l != self.source_lang and l in LANGUAGES]

    # ---------- pipeline ----------

    async def _consume(self) -> None:
        while True:
            item = await self._transcripts.get()
            if item is None:
                break

            if isinstance(item, Usage):
                self.metrics.note_asr_usage(item.total_tokens)
                continue

            if item.speaker:
                self._speaker = item.speaker
            # La huella de voz manda sobre cualquier etiqueta del motor: es la
            # que tiene nombre. Se vota en cada refresco y se decide por frase.
            if self.tracker.active:
                # La huella se calcula en un hilo: con el embedder neuronal son
                # ~90 ms de CPU, y bloquear el loop frena la bomba de audio hacia
                # el ASR de todas las salas. Se hace en el hilo aparte y se sigue.
                self._speaker = await asyncio.to_thread(self.tracker.observe)

            self._has_source = True
            self.metrics.note_subtitle()
            if item.is_final:
                if self.tracker.active:
                    self._speaker = self.tracker.commit() or self._speaker
                await self._finalize(item.text.strip(), item.index)
                continue

            # El motor manda la frase completa en curso: solo reemplazamos.
            self._partial = item.text.strip()
            if self._partial:
                seq = item.index if item.index >= 0 else self._seq
                self.bus.publish(self._segment_event(self._partial, False, seq=seq))
                self._maybe_translate_partial(seq)

    async def _finalize(self, text: str, index: int = -1) -> None:
        text = text.strip()
        self._partial = ""
        if not text:
            return

        # Si el motor numera las lineas, mandamos eso: una correccion suya llega
        # con el mismo seq y el espectador ve la linea corregirse, no duplicarse.
        if index >= 0:
            seq = index
            revision = self._finals.get(seq) is not None
            self._seq = max(self._seq, seq + 1)
        else:
            seq = self._seq
            self._seq += 1
            revision = False

        if self._finals.get(seq) == text:
            return  # identica a lo ya emitido: no hay nada que decir
        self._finals[seq] = text
        self._speaker_by_seq[seq] = self._speaker
        while len(self._speaker_by_seq) > _FINALS_MEMORY:
            self._speaker_by_seq.popitem(last=False)
        while len(self._finals) > _FINALS_MEMORY:
            self._finals.popitem(last=False)
        if not revision:
            self.metrics.segments_final += 1

        # El original sale ya. La traduccion parchea este mismo seq cuando llegue.
        self.bus.publish(self._segment_event(text, True, seq=seq))

        self._history.append(text)
        del self._history[:-_CONTEXT_SEGMENTS]
        self._speaker = ""

        targets = self.active_targets()
        if targets:
            self._spawn(self._translate(seq, text, targets, final=True))

    def _maybe_translate_partial(self, seq: int) -> None:
        now = time.time()
        if self._partial_mt_inflight or now - self._last_partial_mt < settings.partial_mt_ms / 1000:
            return
        targets = self.active_targets()
        if not targets:
            return
        self._last_partial_mt = now
        self._partial_mt_inflight = True
        self._spawn(self._translate(seq, self._partial, targets, final=False))

    def _spawn(self, coro) -> None:
        """Lanza una traduccion en paralelo sin perderle el rastro."""
        task = asyncio.create_task(coro, name=f"mt:{self.id}")
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)

    async def _translate(self, seq: int, text: str, targets: list[str], *, final: bool) -> None:
        t0 = time.time()
        try:
            context = " ".join(self._history[-_CONTEXT_SEGMENTS:])
            result = await self._mt.translate(
                text,
                source_lang=self.source_lang,
                targets=targets,
                context=context,
                glossary=self.glossary,
            )
            ms = int((time.time() - t0) * 1000)
            self.metrics.note_mt(ms, ok=bool(result), retries=result.retries)
            if result.tokens:
                self.metrics.note_mt_usage(result.tokens)
            if result.texts:
                self.bus.publish(
                    {
                        "type": "translation",
                        "room": self.id,
                        "seq": seq,
                        "final": final,
                        "tr": result.texts,
                        "mt_ms": ms,
                        "speaker": self._speaker_by_seq.get(seq, self._speaker),
                    }
                )
        except Exception:
            self.metrics.note_mt(int((time.time() - t0) * 1000), ok=False)
            log.exception("Traduccion fallida en la sala %s", self.id)
        finally:
            if not final:
                self._partial_mt_inflight = False

    # ---------- salida ----------

    def _segment_event(self, text: str, final: bool, seq: int | None = None) -> dict:
        return {
            "type": "segment",
            "room": self.id,
            "seq": self._seq if seq is None else seq,
            "final": final,
            "lang": self.source_lang,
            "text": text,
            "speaker": self._speaker,
            "t": time.time(),
        }

    def _publish_state(self) -> None:
        self.bus.publish({"type": "state", **self.state()})

    def state(self) -> dict:
        return {
            "room": self.id,
            "title": self.title,
            "speakers": self.speakers,
            "source_lang": self.source_lang,
            "running": self.running,
            "live": self._has_source,
            "viewers": sum(self._viewer_langs.values()),
            "viewer_langs": dict(self._viewer_langs),
            "asr": self._asr.name if self._asr else self._asr_name,
            "mt": self._mt.name if self._mt else self._mt_name,
            "glossary_terms": len(self.glossary.split(", ")) if self.glossary else 0,
            "vad": bool(self.gate),
            "speaker_id": self.tracker.active,
            "speakers": self.tracker.registry.names(),
            "current_speaker": self.tracker.current,
            "speaker_last": (
                {"name": m.name, "z": round(m.z, 2), "score": round(m.score, 3),
                 "runner_up": m.runner_up, "runner_score": round(m.runner_score, 3)}
                if (m := self.tracker.last) else None
            ),
            "metrics": self.metrics.snapshot(),
        }


class RoomRegistry:
    """Todas las salas del evento. Crear salas es barato: una charla, un objeto."""

    def __init__(self) -> None:
        self._rooms: dict[str, Room] = {}

    def __contains__(self, room_id: str) -> bool:
        return room_id in self._rooms

    def get(self, room_id: str) -> Room | None:
        return self._rooms.get(room_id)

    def all(self) -> list[Room]:
        return list(self._rooms.values())

    async def create(self, room_id: str, **kwargs) -> Room:
        existing = self._rooms.get(room_id)
        if existing:
            return existing
        room = Room(room_id, **kwargs)
        # Solo se registra si arranco de verdad: una sala a medio construir
        # ensucia el panel y acepta audio que nadie va a transcribir.
        await room.start()
        self._rooms[room_id] = room
        return room

    async def remove(self, room_id: str) -> bool:
        room = self._rooms.pop(room_id, None)
        if not room:
            return False
        await room.stop()
        return True

    async def shutdown(self) -> None:
        await asyncio.gather(*(r.stop() for r in self._rooms.values()), return_exceptions=True)
        self._rooms.clear()
