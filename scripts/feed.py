#!/usr/bin/env python3
"""Alimenta una sala de Poliglota desde cualquier fuente que ffmpeg sepa abrir.

    scripts/feed.py auditorio charla.mp4 --lang en --title "Scaling RAG"
    scripts/feed.py track-2 https://ejemplo.com/stream.m3u8 --lang en
    scripts/feed.py auditorio default --device pulse       # entrada de audio del sistema

El audio se envia a ritmo real (`-re`). No es un detalle cosmetico: si se
empujara un archivo a toda velocidad, el reconocedor recibiria una hora de charla
en segundos y las metricas de latencia dejarian de significar algo.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import shutil
import sys
import time

try:
    import websockets
except ImportError:
    sys.exit("Falta 'websockets'. Instala el proyecto:  uv pip install -e .")

SAMPLE_RATE = 16000
CHUNK = 3200  # 100 ms de PCM16 a 16 kHz


def default_monitor() -> str | None:
    """Nombre del monitor del dispositivo de salida por defecto.

    El "monitor" es lo que suena por los parlantes. Capturarlo permite
    alimentar una sala con cualquier cosa que reproduzca la maquina —una charla
    en el navegador, un archivo, una videollamada— sin depender de que el
    navegador sepa compartir audio de pestana, que en Firefox no se puede.
    """
    import shutil
    import subprocess

    if not shutil.which("pactl"):
        return None
    try:
        sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
        return f"{sink}.monitor" if sink else None
    except Exception:
        return None


def build_ffmpeg(source: str, device: str | None, realtime: bool) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if device:
        cmd += ["-f", device]
    elif realtime:
        cmd += ["-re"]
    cmd += ["-i", source]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
    return cmd


async def run(args: argparse.Namespace) -> int:
    if not shutil.which("ffmpeg"):
        print("ffmpeg no esta instalado o no esta en el PATH.", file=sys.stderr)
        return 1

    scheme = "wss" if args.server.startswith("https") else "ws"
    host = args.server.split("://", 1)[-1].rstrip("/")
    url = f"{scheme}://{host}/ws/ingest/{args.room}"

    source = args.source
    if args.device == "pulse" and source in ("default", "monitor", "sistema"):
        source = default_monitor()
        if not source:
            print("No se pudo detectar la salida de audio por defecto. "
                  "Listala con:  pactl list short sources | grep monitor", file=sys.stderr)
            return 1
        print(f"-> capturando el audio del sistema: {source}")

    cmd = build_ffmpeg(source, args.device, not args.fast)
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    sent = 0
    last_report = -1
    started = time.time()
    try:
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(json.dumps({
                "type": "meta",
                "title": args.title,
                "speakers": args.speakers,
                "abstract": args.abstract,
                "source_lang": args.lang,
            }))
            print(f"-> {args.room}: alimentando desde {source}")

            while True:
                chunk = await proc.stdout.read(CHUNK)
                if not chunk:
                    break
                await ws.send(chunk)
                sent += len(chunk)
                secs = int(sent / (SAMPLE_RATE * 2))
                if secs >= 10 and secs % 10 == 0 and secs != last_report:
                    last_report = secs
                    print(f"   {args.room}: {secs}s de audio enviados", end="\r", flush=True)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"\n-> {args.room}: interrumpido")
    except Exception as exc:
        print(f"\n-> {args.room}: error {exc}", file=sys.stderr)
        stderr = await proc.stderr.read()
        if stderr:
            print(stderr.decode(errors="replace")[:600], file=sys.stderr)
        return 1
    finally:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        await proc.wait()

    total = sent / (SAMPLE_RATE * 2)
    print(f"\n-> {args.room}: fin. {total:.0f}s de audio en {time.time() - started:.0f}s reales")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("room", help="identificador de la sala")
    p.add_argument("source", help="archivo, URL, stream, o 'default' con --device pulse "
                                  "para capturar lo que suene en la maquina")
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--lang", default="en", help="idioma del orador (o 'auto')")
    p.add_argument("--title", default="", help="titulo de la charla: alimenta el glosario")
    p.add_argument("--speakers", default="")
    p.add_argument("--abstract", default="", help="resumen: de aca salen los terminos tecnicos")
    p.add_argument("--device", default=None, metavar="FMT",
                   help="captura de dispositivo en vez de archivo (pulse, alsa, avfoundation, dshow)")
    p.add_argument("--fast", action="store_true",
                   help="enviar a maxima velocidad en vez de tiempo real (invalida las metricas de latencia)")
    args = p.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
