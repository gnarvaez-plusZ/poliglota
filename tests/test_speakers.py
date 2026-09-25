"""Pruebas de identificacion de hablante.

Dos niveles:

1. Voces sinteticas (sin red): dos "hablantes" con tono y timbre distintos,
   generados con armonicos. Prueban la plomeria de forma determinista: registro,
   reconocimiento, umbral de Desconocido, persistencia y votacion por frase.
2. Voces reales de TTS (si existen en samples/): tres voces de Gemini leyendo
   el MISMO texto. Registra dos y comprueba que reconoce a cada una en tramos
   que no vio, y que a la tercera la llama Desconocido. Al usar el mismo texto,
   lo unico que difiere entre pistas es la voz.

    .venv/bin/python tests/test_speakers.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from poliglota.speakers import UNKNOWN, SR, SpeakerRegistry, SpeakerTracker

ROOT = Path(__file__).resolve().parent.parent
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASA ' if ok else 'FALLA'} {name}" + (f"   <- {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def synth_voice(seconds: float, f0: float, tilt: float, seed: int, vibrato: float = 0.03) -> bytes:
    """Voz sintetica: tren de armonicos con envolvente espectral propia.

    `f0` es el tono base y `tilt` cuanto caen los armonicos altos: entre las dos
    cosas definen un "timbre". Se modula el tono y la amplitud con un ruido
    lento para que los frames no sean identicos y el desvio no sea cero.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    drift = np.interp(t, np.linspace(0, seconds, 40), rng.normal(0, 1, 40))
    inst_f0 = f0 * (1 + vibrato * drift)
    phase = 2 * np.pi * np.cumsum(inst_f0) / SR
    x = np.zeros_like(t)
    for h in range(1, 25):
        x += (h ** -tilt) * np.sin(h * phase + rng.uniform(0, np.pi))
    env = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 3)))   # silabas
    x = x / np.max(np.abs(x)) * 0.5 * env
    return (x * 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------- sinteticas ----
print("== Voces sinteticas (sin red) ==")
with tempfile.TemporaryDirectory() as tmp:
    reg = SpeakerRegistry(Path(tmp) / "huellas.json")
    juan = lambda s, seed: synth_voice(s, f0=110.0, tilt=1.6, seed=seed)   # grave, oscuro
    pedro = lambda s, seed: synth_voice(s, f0=185.0, tilt=0.9, seed=seed)  # agudo, brillante
    otro = lambda s, seed: synth_voice(s, f0=300.0, tilt=0.4, seed=seed)   # muy distinto

    # Registro con varias semillas: una voz real varia de una frase a otra, y un
    # registro artificialmente uniforme subestima esa dispersion.
    reg.enroll("Juan", b"".join(juan(4, sd) for sd in (1, 4, 7)))
    reg.enroll("Pedro", b"".join(pedro(4, sd) for sd in (2, 5, 8)))
    check("registra dos hablantes", reg.names() == ["Juan", "Pedro"], str(reg.names()))

    mj = reg.identify(juan(2, 11))
    mp = reg.identify(pedro(2, 12))
    check("reconoce a Juan en un tramo nuevo", mj is not None and mj.name == "Juan",
          f"{mj.name if mj else None} ({mj.score:.2f} vs {mj.runner_score:.2f})" if mj else "None")
    check("reconoce a Pedro en un tramo nuevo", mp is not None and mp.name == "Pedro",
          f"{mp.name if mp else None} ({mp.score:.2f} vs {mp.runner_score:.2f})" if mp else "None")

    mo = reg.identify(otro(2, 13))
    check("a una voz ajena la llama Desconocido", mo is not None and mo.name == UNKNOWN,
          f"{mo.name if mo else None} ({mo.score:.2f})" if mo else "None")

    check("silencio no decide nada", reg.identify(b"\x00\x00" * SR * 2) is None)

    # Persistencia: otro registro sobre el mismo archivo tiene que ver lo mismo.
    reg2 = SpeakerRegistry(Path(tmp) / "huellas.json")
    check("las huellas sobreviven al reinicio", reg2.names() == ["Juan", "Pedro"])

    check("olvidar borra", reg.forget("Pedro") and reg.names() == ["Juan"])

    # Votacion por frase: la ventana suelta puede equivocarse, la frase no.
    reg.enroll("Pedro", b"".join(pedro(4, sd) for sd in (3, 6, 9)))
    tr = SpeakerTracker(reg)
    tr.min_interval_s = 0

    def en_rebanadas(pcm: bytes, ms: int = 500):
        """Como llega el audio real: un habla continua, entregada por trozos."""
        paso = SR * 2 * ms // 1000
        for i in range(0, len(pcm), paso):
            tr.feed(pcm[i:i + paso])
            tr.observe()

    en_rebanadas(juan(3, 21))
    check("atribuye la frase por mayoria", tr.commit() == "Juan", tr.current)
    tr._buf.clear()
    en_rebanadas(pedro(3, 22))
    check("cambia de hablante al cambiar la voz", tr.commit() == "Pedro", tr.current)

# ---------------------------------------------------------------- TTS reales ----
from poliglota.speakers import build_embedder

voces = {v: ROOT / "samples" / f"voz-{v}.raw" for v in ("Charon", "Kore")}
tercera = ROOT / "samples" / "voz-3.raw"
real3 = tercera.exists()
voces["Tercera"] = tercera if real3 else ROOT / "samples" / "voz-Puck.raw"

try:
    import resemblyzer  # noqa: F401
    embedders = ["mfcc", "resemblyzer"]
except ImportError:
    embedders = ["mfcc"]

if all(p.exists() for p in voces.values()):
    pcm = {v: p.read_bytes() for v, p in voces.items()}
    S = SR * 2
    print(f"\n== Voces reales de TTS, mismo texto: tercera voz "
          f"{'REAL (' + tercera.resolve().name + ')' if real3 else 'DERIVADA de Kore, no es otra persona'} ==")

    def tramos(b: bytes, largo: int = 3):
        for i in range(14, len(b) // S - largo, largo):
            yield b[i * S:(i + largo) * S]

    for kind in embedders:
        with tempfile.TemporaryDirectory() as tmp:
            reg = SpeakerRegistry(Path(tmp) / "h.json", embedder=build_embedder(kind))
            reg.enroll("Juan", pcm["Charon"][: 12 * S])
            reg.enroll("Pedro", pcm["Kore"][: 12 * S])
            print(f"  -- {reg.embedder.name} (margen {reg.margin}) --")
            for pista, esperado in (("Charon", "Juan"), ("Kore", "Pedro"), ("Tercera", UNKNOWN)):
                ok = tot = 0
                zs = []
                for clip in tramos(pcm[pista]):
                    m = reg.identify(clip)
                    if m is None:
                        continue
                    tot += 1
                    ok += m.name == esperado
                    zs.append(m.z)
                rate = ok / tot if tot else 0
                etiqueta = f"{pista:7} -> {esperado:11} {ok}/{tot} tramos"
                if pista == "Tercera" and (kind == "mfcc" or not real3):
                    # El clasico separa hombre/mujer, no dos voces del mismo registro:
                    # limitacion medida y documentada, no se le exige.
                    print(f"  ---   {etiqueta}   (informativo: {'embedder clasico' if kind == 'mfcc' else 'voz derivada'})")
                else:
                    check(f"{reg.embedder.name}: {etiqueta}", rate >= 0.8, f"z {np.round(zs, 2).tolist()}")
else:
    print("  (faltan samples/voz-*.raw; generalos con scripts/make_sample.py --voice Charon|Kore|Aoede)")

print(f"\n{'HABLANTES OK' if not fails else f'{len(fails)} FALLAS: {fails}'}")
sys.exit(1 if fails else 0)
