"""API HTTP + WebSocket de Poliglota.

Dos WebSockets por sala, con roles separados a proposito:

  /ws/ingest/{room}  <- audio PCM16 16 kHz mono, binario. Una unica fuente.
  /ws/view/{room}    -> subtitulos en JSON. N espectadores, cada uno con su idioma.

Separarlos permite que la fuente de audio (cabina de sonido, OBS, un script con
ffmpeg) viva en un proceso distinto del publico, y que caerse un espectador no
toque la captura.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import LANGUAGES, settings
from .room import RoomRegistry
from .speakers import get_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("poliglota")

WEB = Path(__file__).resolve().parent.parent / "web"
rooms = RoomRegistry()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await rooms.shutdown()


app = FastAPI(title="Poliglota", version="0.1.0", lifespan=lifespan)


class RoomSpec(BaseModel):
    id: str
    title: str = ""
    speakers: str = ""
    abstract: str = ""
    source_lang: str = "en"
    asr_engine: str | None = None
    mt_engine: str | None = None


# ---------- API ----------


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "asr": settings.asr_engine,
        "mt": settings.mt_engine,
        "key_configured": bool(settings.gemini_api_key),
        "rooms": len(rooms.all()),
        "languages": LANGUAGES,
    }


@app.get("/api/rooms")
async def list_rooms() -> dict:
    return {"rooms": [r.state() for r in rooms.all()]}


@app.post("/api/rooms")
async def create_room(spec: RoomSpec) -> dict:
    room_id = spec.id.strip().lower().replace(" ", "-")
    if not room_id:
        raise HTTPException(400, "El id de sala no puede estar vacio")
    try:
        room = await rooms.create(
            room_id,
            title=spec.title,
            speakers=spec.speakers,
            abstract=spec.abstract,
            source_lang=spec.source_lang,
            asr_engine=spec.asr_engine,
            mt_engine=spec.mt_engine,
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return room.state()


@app.delete("/api/rooms/{room_id}")
async def delete_room(room_id: str) -> dict:
    if not await rooms.remove(room_id):
        raise HTTPException(404, "Sala inexistente")
    return {"deleted": room_id}


@app.get("/api/rooms/{room_id}")
async def get_room(room_id: str) -> dict:
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, "Sala inexistente")
    return room.state()


# ---------- WebSockets ----------


@app.websocket("/ws/ingest/{room_id}")
async def ws_ingest(ws: WebSocket, room_id: str) -> None:
    """Recibe audio crudo. Crea la sala al vuelo si no existia."""
    await ws.accept()
    room = rooms.get(room_id)
    if room is None:
        room = await rooms.create(room_id, title=room_id)
    await ws.send_text(json.dumps({"type": "ready", "room": room.id}))
    # Se probo precalentar la sesion con el modelo empujando 100 ms de
    # silencio al abrir el socket, para sacar el apreton de manos del camino
    # del primer subtitulo. La medicion salio peor (12 s contra 2,8 s) y no se
    # pudo separar el efecto del cambio de la degradacion de la API en ese
    # momento. Ante la duda, no se hace: el motor conecta con la primera palabra.

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                room.feed(data)
            elif (text := msg.get("text")) is not None:
                # Canal de control en la misma conexion: metadatos de la charla
                # que llegan despues de abierto el stream.
                with contextlib.suppress(Exception):
                    payload = json.loads(text)
                    if payload.get("type") == "meta":
                        _apply_meta(room, payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Error en la ingesta de %s", room_id)
    finally:
        log.info("Fuente de audio desconectada de %s", room_id)


def _apply_meta(room, payload: dict) -> None:
    from . import glossary

    room.title = payload.get("title") or room.title
    room.speakers = payload.get("speakers") or room.speakers
    room.abstract = payload.get("abstract") or room.abstract
    room.source_lang = payload.get("source_lang") or room.source_lang
    room.glossary = glossary.build(
        title=room.title, abstract=room.abstract, speakers=room.speakers
    )
    log.info("Sala %s: glosario regenerado (%d terminos)", room.id, len(room.glossary.split(", ")))


@app.websocket("/ws/view/{room_id}")
async def ws_view(ws: WebSocket, room_id: str) -> None:
    """Emite subtitulos a un espectador. `lang` define que traduccion quiere."""
    await ws.accept()
    room = rooms.get(room_id)
    if room is None:
        await ws.send_text(json.dumps({"type": "error", "detail": "room_not_found"}))
        await ws.close()
        return

    lang = (ws.query_params.get("lang") or "es").lower()
    if lang not in LANGUAGES:
        lang = "es"

    queue = room.add_viewer(lang)
    await ws.send_text(json.dumps({"type": "state", **room.state()}))

    async def watch_client() -> None:
        """Permite cambiar de idioma sin reconectar."""
        nonlocal lang
        while True:
            raw = await ws.receive_text()
            with contextlib.suppress(Exception):
                payload = json.loads(raw)
                if payload.get("type") == "lang" and payload.get("lang") in LANGUAGES:
                    room.switch_viewer_lang(lang, payload["lang"])
                    lang = payload["lang"]

    watcher = asyncio.create_task(watch_client())
    try:
        while True:
            event = await queue.get()
            await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    except Exception:
        log.debug("Espectador de %s cerrado", room_id, exc_info=True)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        room.remove_viewer(lang, queue)


# ---------- UI ----------


# ---------- hablantes ----------


@app.get("/api/speakers")
async def list_speakers() -> dict:
    reg = get_registry()
    return {"speakers": reg.describe(), "margin": reg.margin, "embedder": reg.embedder.name}


@app.post("/api/speakers/identify")
async def identify_speaker(request: Request) -> dict:
    """Quien habla en el audio del cuerpo (PCM16 16 kHz mono). Para probar el registro."""
    pcm = await request.body()
    m = get_registry().identify(pcm)
    if m is None:
        return {"name": None, "score": 0.0, "detail": "sin voz suficiente o sin huellas registradas"}
    return {"name": m.name, "score": round(m.score, 3), "z": round(m.z, 2), "runner_up": m.runner_up,
            "runner_score": round(m.runner_score, 3), "known": m.known}


@app.post("/api/speakers/{name}")
async def enroll_speaker(name: str, request: Request) -> dict:
    """Registra (o refuerza) la voz de `name` con el audio del cuerpo (PCM16 16 kHz mono)."""
    pcm = await request.body()
    if len(pcm) < settings.sample_rate * settings.sample_width * 3:
        raise HTTPException(400, "hacen falta al menos 3 segundos de audio")
    try:
        vp = get_registry().enroll(name, pcm)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"name": vp.name, "seconds": round(vp.seconds, 1), "clips": vp.clips}


@app.delete("/api/speakers/{name}")
async def forget_speaker(name: str) -> dict:
    if not get_registry().forget(name):
        raise HTTPException(404, "no hay una huella con ese nombre")
    return {"deleted": name}


@app.get("/enroll")
async def enroll_page() -> FileResponse:
    return _page("enroll.html")


@app.get("/qr/{room_id}")
async def qr(room_id: str, request: Request) -> Response:
    """QR que lleva a la vista de subtitulos de la sala.

    Se proyecta en la pantalla de la sala o se pone en el overlay del stream.
    El asistente lo escanea y se lleva los subtitulos al telefono, en el idioma
    que quiera, sin instalar nada. Es la diferencia entre una pantalla comun
    para todos y que cada uno lea en su idioma.

    La URL se arma con el host por el que entro el pedido, para que el codigo
    apunte a la IP de la red del evento y no a `localhost`.
    """
    import io
    import qrcode
    import qrcode.image.svg

    base = str(request.base_url).rstrip("/")
    lang = request.query_params.get("lang")
    target = f"{base}/room/{room_id}" + (f"?lang={lang}" if lang else "")

    img = qrcode.make(target, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return Response(
        content=buf.getvalue(),
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600", "X-Poliglota-Target": target},
    )


@app.get("/favicon.ico")
@app.get("/favicon.svg")
async def favicon() -> FileResponse:
    return FileResponse(WEB / "favicon.svg", media_type="image/svg+xml")


# Las paginas se sirven con `no-cache`: el navegador las revalida en cada
# carga. Sin esto, un arreglo publicado en el servidor podia no llegar a la
# cabina de sonido porque su navegador seguia usando la copia vieja, y la
# unica pista era "sigue lento".
_PAGE_HEADERS = {"Cache-Control": "no-cache"}


def _page(name: str) -> FileResponse:
    return FileResponse(WEB / name, headers=_PAGE_HEADERS)


@app.get("/")
async def index() -> FileResponse:
    return _page("index.html")


@app.get("/room/{room_id}")
async def room_page(room_id: str) -> FileResponse:
    return _page("room.html")


@app.get("/overlay/{room_id}")
async def overlay_page(room_id: str) -> FileResponse:
    return _page("overlay.html")


@app.get("/capture/{room_id}")
async def capture_page(room_id: str) -> FileResponse:
    return _page("capture.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")


def cli() -> None:
    uvicorn.run(
        "poliglota.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        ws_max_size=16 * 1024 * 1024,
    )


if __name__ == "__main__":
    cli()
