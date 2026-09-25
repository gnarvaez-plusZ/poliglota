"""Pruebas de la compuerta de voz, sin red ni credenciales.

Lo que hay que demostrar es exactamente esto: que retiene el silencio largo y
que **no toca una sola muestra del habla**. Ahorrar tokens comiendose el ataque
de una palabra seria un mal negocio: la precision es el criterio que mas pesa.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from poliglota.gate import SpeechGate

RATE, WIDTH = 16000, 2
CHUNK = 3200  # 100 ms

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASA ' if ok else 'FALLA'} {name}" + (f"   <- {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def silence(ms: int) -> bytes:
    return b"\x00\x00" * int(RATE * ms / 1000)


def speech(ms: int, amp: float = 0.25) -> bytes:
    n = int(RATE * ms / 1000)
    t = np.arange(n) / RATE
    wave = (np.sin(2 * math.pi * 220 * t) * amp * 32767).astype(np.int16)
    return wave.tobytes()


def chunks(pcm: bytes) -> list[bytes]:
    return [pcm[i:i + CHUNK] for i in range(0, len(pcm), CHUNK)]


# --- una sala abierta sin nadie hablando no debe costar nada ---
g = SpeechGate()
out = b"".join(b"".join(g.feed(c)) for c in chunks(silence(30_000)))
check("30s de sala vacia casi no se envian", g.saved_ratio > 0.90,
      f"solo ahorro {100 * g.saved_ratio:.0f}%")
check("el silencio largo no pasa", len(out) < RATE * WIDTH * 3, f"{len(out) / (RATE * WIDTH):.1f}s")

# --- el habla tiene que pasar entera, muestra por muestra ---
g = SpeechGate()
voice = speech(4_000)
got = b"".join(b"".join(g.feed(c)) for c in chunks(voice))
check("el habla pasa completa", voice in got, f"entraron {len(voice)}B, salieron {len(got)}B")

# --- el preroll evita comerse el ataque de la primera palabra ---
g = SpeechGate()
emitted = []
for c in chunks(silence(5_000)):
    emitted += g.feed(c)
before = sum(len(x) for x in emitted)
for c in chunks(speech(1_000)):
    emitted += g.feed(c)
after = sum(len(x) for x in emitted) - before
# Sale el habla MAS el preroll guardado: por eso supera 1 s.
check("manda preroll antes de la primera silaba", after > RATE * WIDTH * 1.0,
      f"{after / (RATE * WIDTH):.2f}s para 1.00s de voz")

# --- una pausa corta dentro de una frase no corta el envio ---
g = SpeechGate()
sent = []
for part in (speech(800), silence(500), speech(800)):
    for c in chunks(part):
        sent += g.feed(c)
total = sum(len(x) for x in sent) / (RATE * WIDTH)
check("una pausa breve no interrumpe la frase", total >= 2.0,
      f"envio {total:.2f}s de 2.10s")

# --- contabilidad coherente ---
g = SpeechGate()
for c in chunks(speech(1_000) + silence(10_000)):
    g.feed(c)
check("nunca reporta ahorro negativo", 0.0 <= g.saved_ratio <= 1.0, f"{g.saved_ratio}")
check("no inventa bytes", g.bytes_out <= g.bytes_in, f"{g.bytes_out} > {g.bytes_in}")

print(f"\n{'COMPUERTA OK' if not fails else f'{len(fails)} FALLAS: {fails}'}")
sys.exit(1 if fails else 0)
