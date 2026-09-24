"""Prueba de extremo a extremo del pipeline, sin credenciales ni red externa.

Levanta dos salas simultaneas con el motor mock, conecta espectadores en dos
idiomas y verifica que el contrato de eventos se cumpla: parciales que crecen,
finales numerados, y aislamiento entre salas.
"""
from __future__ import annotations

import asyncio
import json

import uvicorn
import websockets

from poliglota.main import app

PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"
WS = f"ws://127.0.0.1:{PORT}"


async def create_room(session, room_id: str, title: str) -> None:
    import httpx

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE}/api/rooms",
            json={
                "id": room_id,
                "title": title,
                "source_lang": "en",
                "asr_engine": "mock",
                "mt_engine": "none",
            },
        )
        r.raise_for_status()


async def collect(room_id: str, lang: str, seconds: float) -> list[dict]:
    events: list[dict] = []
    async with websockets.connect(f"{WS}/ws/view/{room_id}?lang={lang}") as ws:
        try:
            async with asyncio.timeout(seconds):
                async for raw in ws:
                    events.append(json.loads(raw))
        except TimeoutError:
            pass
    return events


async def main() -> int:
    config = uvicorn.Config(app, port=PORT, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    failures: list[str] = []
    try:
        await create_room(None, "auditorio", "Scaling RAG with Kubernetes")
        await create_room(None, "track-2", "Observability with eBPF")

        a, b = await asyncio.gather(
            collect("auditorio", "es", 6.0),
            collect("track-2", "pt", 6.0),
        )

        def check(name: str, cond: bool, detail: str = "") -> None:
            print(f"  {'PASA' if cond else 'FALLA'}  {name}{'  <- ' + detail if detail and not cond else ''}")
            if not cond:
                failures.append(name)

        print("\n== Sala 'auditorio' (espectador es) ==")
        segs_a = [e for e in a if e.get("type") == "segment"]
        finals_a = [e for e in segs_a if e["final"]]
        partials_a = [e for e in segs_a if not e["final"]]
        check("recibe parciales", len(partials_a) > 5, f"{len(partials_a)}")
        check("recibe finales", len(finals_a) >= 1, f"{len(finals_a)}")
        check("los parciales crecen",
              len(partials_a) > 1 and len(partials_a[-1]["text"]) > len(partials_a[0]["text"]))
        check("los finales numeran secuencial",
              [e["seq"] for e in finals_a] == list(range(len(finals_a))),
              str([e["seq"] for e in finals_a]))
        check("reporta latencia de asr", all(e["asr_ms"] > 0 for e in segs_a))
        check("el estado inicial llega primero", a[0].get("type") == "state", a[0].get("type", "?"))

        print("\n== Aislamiento entre salas ==")
        segs_b = [e for e in b if e.get("type") == "segment"]
        check("la segunda sala transcribe en paralelo", len(segs_b) > 5, f"{len(segs_b)}")
        check("cada sala recibe solo lo suyo",
              {e["room"] for e in segs_a} == {"auditorio"} and {e["room"] for e in segs_b} == {"track-2"})

        if finals_a:
            print(f"\n  ejemplo de final: seq={finals_a[0]['seq']} \"{finals_a[0]['text'][:70]}...\"")
    finally:
        server.should_exit = True
        await task

    print(f"\n{'TODO VERDE' if not failures else f'{len(failures)} FALLAS: {failures}'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
