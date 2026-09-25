"""Identificacion de hablante por huella de voz: "Juan: ...", "Pedro: ...", "Desconocido: ...".

No es diarizacion sino reconocimiento: cada persona registra su voz antes de la
charla y en vivo cada frase se atribuye a quien mas se le parece. Si nadie se
parece lo suficiente, es "Desconocido". Sirve para un panel, para las preguntas
del publico o para una charla a dos voces.

Por que no la diarizacion de Gemini: medida hoy, multiplica por ocho el
intervalo de refresco del subtitulo (490 ms -> 2728 ms) y devuelve etiquetas
anonimas ("1", "2") que igual habria que mapear a nombres a mano. Esto corre
local, en numpy puro, en pocos milisegundos, y no toca la latencia del ASR.

La huella es clasica: estadisticas de MFCC mas estadisticas de tono (F0). No es
un embedding neuronal y hay que decirlo claro: distingue bien voces distintas
sobre el mismo microfono (dos personas en un panel), y confunde voces parecidas.
El punto de reemplazo por un modelo neuronal es `Embedder`; el resto no cambia.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger("poliglota.speakers")

UNKNOWN = "Desconocido"

# ---------------------------------------------------------------- señal ----

SR = 16000
_WIN = 400          # 25 ms
_HOP = 160          # 10 ms
_NFFT = 512
_NMEL = 26
_NCEP = 13
_FMIN, _FMAX = 80, 7600
_F0_MIN, _F0_MAX = 60.0, 400.0
# Un frame cuenta como voz si su energia supera este piso absoluto (escala
# int16 normalizada) y ademas una fraccion del pico del clip.
_RMS_FLOOR = 0.008
_RMS_REL = 0.06
# Piso de dispersion propia (por dimension estandarizada), por si el registro fue muy uniforme.
_MIN_SPREAD = 0.15


def _mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def _mel_filterbank() -> np.ndarray:
    pts = _hz(np.linspace(_mel(_FMIN), _mel(_FMAX), _NMEL + 2))
    bins = np.floor((_NFFT + 1) * pts / SR).astype(int)
    fb = np.zeros((_NMEL, _NFFT // 2 + 1))
    for i in range(_NMEL):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        mid = max(mid, lo + 1)
        hi = max(hi, mid + 1)
        fb[i, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        fb[i, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fb


def _dct_matrix() -> np.ndarray:
    n = np.arange(_NMEL)
    k = np.arange(_NCEP)[:, None]
    m = np.cos(np.pi * k * (2 * n + 1) / (2 * _NMEL)) * np.sqrt(2.0 / _NMEL)
    m[0] /= np.sqrt(2.0)
    return m


_FB = _mel_filterbank()
_DCT = _dct_matrix()
_HAMMING = np.hamming(_WIN)


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < _WIN:
        return np.empty((0, _WIN), dtype=np.float32)
    n = 1 + (len(x) - _WIN) // _HOP
    idx = np.arange(_WIN)[None, :] + _HOP * np.arange(n)[:, None]
    return x[idx] * _HAMMING


def _pitch(frames: np.ndarray) -> np.ndarray:
    """F0 por frame via autocorrelacion normalizada; NaN donde no hay voz."""
    if frames.shape[0] == 0:
        return np.empty(0)
    lag_lo, lag_hi = int(SR / _F0_MAX), int(SR / _F0_MIN)
    f = frames - frames.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(f, n=2 * _WIN, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :_WIN]
    e0 = ac[:, 0:1] + 1e-9
    acn = ac / e0
    seg = acn[:, lag_lo:lag_hi + 1]
    best = np.argmax(seg, axis=1) + lag_lo
    peak = seg[np.arange(len(best)), best - lag_lo]
    f0 = SR / best.astype(np.float64)
    f0[peak < 0.45] = np.nan          # periodicidad debil: no es voz sonora
    return f0


class Embedder(Protocol):
    dim: int

    def embed(self, pcm: bytes) -> np.ndarray | None:
        """Huella unitaria del clip, o None si no hay voz suficiente."""
        ...


class MfccPitchEmbedder:
    """Huella clasica: media y desvio de MFCC + estadisticas de tono.

    Cada bloque se escala a un rango comparable para que el coseno no quede
    dominado por los coeficientes bajos, que son los que mas dependen del
    microfono y menos del hablante.
    """

    name = "mfcc+pitch"
    dim = (_NCEP - 1) * 2 + 3
    standardize = True
    min_voiced_s = 0.8
    # Piso de dispersion propia, en la escala del espacio estandarizado.
    min_spread = _MIN_SPREAD * np.sqrt(dim)
    # Medido: las voces propias quedan por debajo de z 1,7; el margen deja aire.
    default_margin = 2.5

    def features(self, pcm: bytes) -> tuple[np.ndarray, float] | None:
        x = pcm_to_float(pcm)
        if len(x) < _WIN * 4:
            return None
        x = np.append(x[0], x[1:] - 0.97 * x[:-1])          # pre-enfasis
        fr = _frames(x)
        rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
        voiced = rms > max(_RMS_FLOOR, _RMS_REL * float(rms.max()))
        if voiced.sum() * _HOP / SR < self.min_voiced_s:
            return None

        spec = np.abs(np.fft.rfft(fr[voiced], n=_NFFT, axis=1)) ** 2
        mel = np.log(spec @ _FB.T + 1e-8)
        mfcc = (mel @ _DCT.T)[:, 1:]                          # sin c0 (energia)
        # La MEDIA de los MFCC es el timbre promedio: sobre el mismo microfono es
        # casi puro hablante y es la senal mas fuerte en clips cortos. Una
        # version anterior la restaba (CMN) creyendo que era el canal, y con eso
        # tiraba lo que mas distingue a dos voces. Se conserva junto con la
        # dispersion, que aporta la "forma" de como habla cada uno.
        v_mean = mfcc.mean(axis=0)
        v_std = mfcc.std(axis=0)

        f0 = _pitch(fr[voiced])
        f0 = f0[~np.isnan(f0)]
        if len(f0) >= 5:
            lf0 = np.log(f0)
            p = np.array([lf0.mean(), lf0.std(), len(f0) / max(voiced.sum(), 1)])
        else:
            p = np.array([np.log(150.0), 0.0, 0.0])

        vec = np.concatenate([v_mean / 6.0, v_std / 3.0, (p - np.array([5.0, 0, 0.5])) * np.array([4.0, 2.0, 1.0])])
        return vec.astype(np.float64), float(voiced.sum() * _HOP / SR)

    def embed(self, pcm: bytes) -> np.ndarray | None:
        out = self.features(pcm)
        if out is None:
            return None
        vec, _ = out
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else None


class ResemblyzerEmbedder:
    """Embedding neuronal de voz (GE2E, 256 dimensiones) via Resemblyzer.

    Es lo que separa dos voces del mismo registro: el embedder clasico distingue
    un hombre de una mujer, pero medido con tres voces de TTS confundio dos
    femeninas 26 veces de 26. Este las separa. Cuesta una dependencia (PyTorch en
    CPU) y unos segundos de carga la primera vez; corre en ~50 ms por clip.
    """

    name = "resemblyzer"
    dim = 256
    standardize = False       # el espacio ya viene bien condicionado
    min_voiced_s = 0.8
    # Los embeddings son unitarios: las distancias entre voces rondan 0,3-0,7.
    min_spread = 0.06
    # Medido con tres voces de TTS y clips de 3 s: con 1,7 se acepta el 100% de
    # los registrados y se rechaza el 92% de una voz ajena; con 5 s, 100/100.
    default_margin = 1.7

    def __init__(self) -> None:
        self._enc = None
        self._lock = threading.Lock()

    def _encoder(self):
        if self._enc is None:
            with self._lock:
                if self._enc is None:
                    import torch
                    from resemblyzer import VoiceEncoder  # import perezoso: es pesado

                    # Un clip de 3 s no justifica 16 hilos; con muchos, torch pelea
                    # la CPU con ffmpeg, el servidor y el resto de las salas.
                    torch.set_num_threads(2)
                    self._enc = VoiceEncoder("cpu", verbose=False)
        return self._enc

    def features(self, pcm: bytes) -> tuple[np.ndarray, float] | None:
        x = pcm_to_float(pcm)
        if len(x) < _WIN * 4:
            return None
        fr = _frames(x)
        rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
        voiced = rms > max(_RMS_FLOOR, _RMS_REL * float(rms.max()))
        secs = float(voiced.sum() * _HOP / SR)
        if secs < self.min_voiced_s:
            return None
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(x.astype(np.float32), source_sr=SR)
        if len(wav) < SR * 0.6:
            return None
        emb = self._encoder().embed_utterance(wav)
        return np.asarray(emb, dtype=np.float64), secs

    def embed(self, pcm: bytes) -> np.ndarray | None:
        out = self.features(pcm)
        return None if out is None else out[0]


def build_embedder(kind: str = "auto") -> Embedder:
    """`auto` usa el neuronal si esta instalado y cae al clasico si no."""
    kind = (kind or "auto").lower()
    if kind in ("auto", "resemblyzer"):
        try:
            import resemblyzer  # noqa: F401

            return ResemblyzerEmbedder()
        except ImportError:
            if kind == "resemblyzer":
                raise RuntimeError("resemblyzer no esta instalado: uv pip install poliglota[voces]")
            log.info("resemblyzer no disponible; identificacion de voz con el embedder clasico")
    return MfccPitchEmbedder()


# -------------------------------------------------------------- registro ----

_SUBCLIP_S = 3.0      # la voz registrada se parte en tramos de este largo
_TIE_RATIO = 0.08     # empate: el actual queda si esta a <8% de distancia del mejor


@dataclass
class VoicePrint:
    """Una voz registrada: sus muestras crudas y lo que se deriva de ellas."""

    name: str
    samples: list[np.ndarray]           # rasgos crudos de cada sub-clip
    seconds: float = 0.0
    created: float = field(default_factory=time.time)

    @property
    def clips(self) -> int:
        return len(self.samples)

    def to_json(self) -> dict:
        return {"name": self.name, "samples": [v.tolist() for v in self.samples],
                "seconds": self.seconds, "created": self.created}

    @classmethod
    def from_json(cls, d: dict) -> "VoicePrint":
        return cls(d["name"], [np.asarray(v, dtype=np.float64) for v in d["samples"]],
                   d.get("seconds", 0.0), d.get("created", 0.0))


@dataclass
class Match:
    name: str
    score: float               # similitud legible (coseno), solo informativa
    z: float                   # distancia en "dispersiones propias" del mejor
    runner_up: str | None = None
    runner_score: float = 0.0

    @property
    def known(self) -> bool:
        return self.name != UNKNOWN


class SpeakerRegistry:
    """Huellas registradas. Global al evento: Juan se registra una vez y se lo
    reconoce en cualquier sala. Se persiste en disco para sobrevivir reinicios.

    Decision: cada dimension se estandariza con la dispersion observada en todas
    las muestras registradas (asi ninguna domina), y un clip se atribuye a la
    voz mas cercana SOLO si esta a menos de `margin` dispersiones propias de esa
    voz, donde la dispersion propia es cuanto se alejan de su centro las
    muestras de su propio registro. Es auto-calibrado: cada persona define con
    su registro que tan "distinta de si misma" puede sonar antes de ser otra.
    """

    def __init__(self, path: Path | None, embedder: Embedder | None = None,
                 margin: float | None = None) -> None:
        self.path = path
        self.embedder = embedder or MfccPitchEmbedder()
        # Sin margen explicito, cada embedder trae el suyo, calibrado con datos.
        self.margin = float(margin) if margin is not None else float(getattr(self.embedder, "default_margin", 2.5))
        self._prints: dict[str, VoicePrint] = {}
        self._lock = threading.Lock()
        self._scale = np.ones(self.embedder.dim)
        self._centroids: dict[str, np.ndarray] = {}
        self._spreads: dict[str, float] = {}
        self._load()

    # ---- persistencia ----

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            if data.get("embedder") not in (None, self.embedder.name):
                log.warning("Las huellas de %s se guardaron con el embedder %s y ahora corre %s: "
                            "hay que registrar las voces de nuevo", self.path, data.get("embedder"), self.embedder.name)
                return
            for d in data.get("speakers", []):
                vp = VoicePrint.from_json(d)
                if vp.samples and vp.samples[0].shape[0] == self.embedder.dim:
                    self._prints[vp.name] = vp
            self._refit()
            log.info("Huellas cargadas: %s", ", ".join(self._prints) or "ninguna")
        except Exception:
            log.exception("No se pudieron cargar las huellas de %s", self.path)

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"embedder": self.embedder.name,
                                   "speakers": [p.to_json() for p in self._prints.values()]}))
        tmp.replace(self.path)

    # ---- geometria ----

    def _refit(self) -> None:
        """Recalcula escala por dimension, centros y dispersiones propias."""
        allv = [v for p in self._prints.values() for v in p.samples]
        if len(allv) >= 2 and getattr(self.embedder, "standardize", True):
            M = np.vstack(allv)
            # Piso relativo a la magnitud de cada dimension: una dimension casi
            # constante en el registro no puede volverse un gatillo que dispare
            # distancias enormes ante la minima variacion.
            self._scale = np.maximum(M.std(axis=0), 0.08 * np.abs(M).mean(axis=0) + 0.02)
        else:
            self._scale = np.ones(self.embedder.dim)
        self._centroids, self._spreads = {}, {}
        for p in self._prints.values():
            z = np.vstack(p.samples) / self._scale
            c = z.mean(axis=0)
            d = np.linalg.norm(z - c, axis=1)
            self._centroids[p.name] = c
            self._spreads[p.name] = float(d.mean()) if len(d) > 1 else 0.0
        # Tope: ninguna voz puede ser mucho mas "ancha" que las demas. Medido con
        # voces reales, la huella mas dispersa absorbia a las otras: cualquier
        # clip a distancia media caia dentro de su radio. Con el tope, un
        # registro desprolijo no se convierte en un iman.
        if len(self._spreads) > 1:
            cap = 1.25 * float(np.median(list(self._spreads.values())))
            self._spreads = {n: min(v, cap) for n, v in self._spreads.items()}
        # Piso de dispersion: nadie puede ser "mas estrecho" que una fraccion de
        # la distancia a su vecino mas cercano. Un registro muy uniforme (leer
        # monotono, o poco audio) subestima cuanto varia una voz real, y sin
        # este piso esa persona saldria como Desconocido demasiado seguido.
        floor_abs = float(getattr(self.embedder, "min_spread", _MIN_SPREAD * np.sqrt(self.embedder.dim)))
        names = list(self._centroids)
        for n in names:
            nearest = min((np.linalg.norm(self._centroids[n] - self._centroids[m]) for m in names if m != n), default=0.0)
            self._spreads[n] = max(self._spreads[n], floor_abs, 0.30 * nearest)

    def _subclips(self, pcm: bytes) -> list[np.ndarray]:
        step = int(SR * 2 * _SUBCLIP_S)
        out = []
        for i in range(0, max(len(pcm) - step // 2, 1), step):
            f = self.embedder.features(pcm[i:i + step])
            if f is not None:
                out.append(f[0])
        return out

    # ---- api ----

    def names(self) -> list[str]:
        return list(self._prints)

    def __len__(self) -> int:
        return len(self._prints)

    def describe(self) -> list[dict]:
        return [{"name": p.name, "seconds": round(p.seconds, 1), "clips": p.clips, "created": p.created}
                for p in self._prints.values()]

    def enroll(self, name: str, pcm: bytes) -> VoicePrint:
        """Registra o refuerza una voz. Los sub-clips se acumulan."""
        name = name.strip()
        if not name or name.lower() == UNKNOWN.lower():
            raise ValueError("nombre invalido")
        samples = self._subclips(pcm)
        if len(samples) < 2:
            raise ValueError("no hay voz suficiente en el audio: hablá al menos 6 segundos seguidos")
        secs = len(pcm) / (SR * 2)
        with self._lock:
            prev = self._prints.get(name)
            if prev:
                vp = VoicePrint(name, prev.samples + samples, prev.seconds + secs, prev.created)
            else:
                vp = VoicePrint(name, samples, secs)
            self._prints[name] = vp
            self._refit()
            self._save()
        log.info("Huella %s: %d sub-clips, %.1fs", name, vp.clips, vp.seconds)
        return vp

    def forget(self, name: str) -> bool:
        with self._lock:
            ok = self._prints.pop(name, None) is not None
            if ok:
                self._refit()
                self._save()
        return ok

    def identify(self, pcm: bytes, prefer: str | None = None) -> Match | None:
        """Quien habla en este clip, o None si no hay voz para decidir."""
        if not self._prints:
            return None
        f = self.embedder.features(pcm)
        if f is None:
            return None
        return self.identify_vector(f[0], prefer=prefer)

    def identify_vector(self, raw: np.ndarray, prefer: str | None = None) -> Match:
        """Quien esta mas cerca, y si esta lo bastante cerca para tener nombre.

        Se ordena por DISTANCIA cruda al centro de cada voz, no por z. Una
        version anterior ordenaba por z = distancia / dispersion propia, y eso
        hacia ganar los empates a la voz mas dispersa —divide por mas—: medido
        con voces reales, 4 de 12 clips se atribuian al otro hablante. La
        dispersion propia se usa solo para decidir si el mas cercano esta lo
        bastante cerca (z <= margen), que es para lo que sirve.

        `prefer`: hablante actual. Si el mejor y el actual estan casi a la misma
        distancia, se conserva el actual: un empate no justifica cambiar de
        nombre en pantalla, y la votacion por frase ya decide lo demas.
        """
        with self._lock:
            z = raw / self._scale
            ranked = []
            for name, c in self._centroids.items():
                d = float(np.linalg.norm(z - c))
                # Similitud legible para la interfaz: coseno de los vectores unitarios.
                cs = float(np.dot(z, c) / ((np.linalg.norm(z) * np.linalg.norm(c)) or 1.0))
                ranked.append((d, name, cs, d / self._spreads[name]))
        ranked.sort()
        if prefer and len(ranked) > 1 and ranked[0][1] != prefer:
            cur = next((r for r in ranked if r[1] == prefer), None)
            if cur is not None and cur[0] <= ranked[0][0] * (1.0 + _TIE_RATIO):
                ranked.remove(cur)
                ranked.insert(0, cur)
        d0, best, cs0, z0 = ranked[0]
        run, cs1 = (ranked[1][1], ranked[1][2]) if len(ranked) > 1 else (None, 0.0)
        name = best if z0 <= self.margin else UNKNOWN
        return Match(name, cs0, z0, run, cs1)


# -------------------------------------------------------- en la sala ----


class SpeakerTracker:
    """Sigue quien habla en una sala a partir del audio que ya pasa por ella.

    Guarda los ultimos segundos de audio; cada vez que el ASR emite texto,
    identifica el ultimo tramo y acumula votos. Al cerrarse una frase, se le
    atribuye el hablante mas votado desde la frase anterior. Es robusto a que
    una ventana suelta se equivoque, y no agrega ninguna llamada a la red.
    """

    # Medido: con 3 s la voz ajena se separa mejor que con 2 s, y la votacion por
    # frase absorbe lo que la ventana mas larga tarda en cambiar de hablante.
    window_s = 3.0
    # 15 s guardados: 3 para identificar y el resto para poder registrar a quien
    # esta hablando desde el audio de la sala misma (mismo microfono, mismo
    # canal que despues se va a reconocer).
    keep_s = 15.0
    min_interval_s = 0.4
    # Con menos audio que esto las estadisticas son ruido: no se decide.
    min_audio_s = 2.0

    def __init__(self, registry: SpeakerRegistry) -> None:
        self.registry = registry
        self._buf = bytearray()
        self._cap = int(SR * 2 * self.keep_s)
        self._votes: dict[str, float] = {}
        self._last = 0.0
        self.current: str = ""
        self.last_score: float = 0.0
        # Ultima decision completa, para diagnostico: sin esto, un "Desconocido"
        # en pantalla no dice si fue por poco o por mucho, ni contra quien.
        self.last: Match | None = None

    @property
    def active(self) -> bool:
        return len(self.registry) > 0

    def feed(self, pcm: bytes) -> None:
        self._buf.extend(pcm)
        if len(self._buf) > self._cap:
            del self._buf[: len(self._buf) - self._cap]

    def observe(self) -> str:
        """Identifica el ultimo tramo (con estrangulador) y devuelve el mejor candidato actual."""
        if not self.active:
            return ""
        now = time.monotonic()
        if now - self._last < self.min_interval_s:
            return self.current
        self._last = now
        tail = bytes(self._buf[-int(SR * 2 * self.window_s):])
        if len(tail) < SR * 2 * self.min_audio_s:
            return self.current
        m = self.registry.identify(tail, prefer=self.current or None)
        if m is None:
            return self.current
        self.last = m
        # Voto ponderado por confianza: cuanto mas cerca del centro (z chico),
        # mas pesa; Desconocido pesa por cuanto se paso del margen.
        w = max(0.1, self.registry.margin - m.z) if m.known else min(2.0, m.z - self.registry.margin + 0.1)
        self._votes[m.name] = self._votes.get(m.name, 0.0) + w
        self.current = max(self._votes.items(), key=lambda kv: kv[1])[0]
        self.last_score = m.score
        return self.current

    def recent_audio(self, seconds: float = 12.0) -> bytes:
        """Los ultimos segundos de audio de la sala, para registrar a quien habla."""
        return bytes(self._buf[-int(SR * 2 * seconds):])

    def commit(self) -> str:
        """Hablante de la frase que acaba de cerrarse; reinicia los votos."""
        if not self.active:
            return ""
        name = self.current
        self._votes.clear()
        return name


# --------------------------------------------------------- singleton ----

_registry: SpeakerRegistry | None = None


def get_registry() -> SpeakerRegistry:
    """Registro unico del proceso, creado con la configuracion del entorno."""
    global _registry
    if _registry is None:
        from .config import settings

        _registry = SpeakerRegistry(Path(settings.speakers_file), embedder=build_embedder(settings.speaker_embedder),
                                    margin=settings.speaker_margin)
    return _registry
