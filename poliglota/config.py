"""Configuracion global, leida del entorno (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))

    asr_engine: str = field(default_factory=lambda: _env("POLIGLOTA_ASR", "gemini"))
    mt_engine: str = field(default_factory=lambda: _env("POLIGLOTA_MT", "gemini"))

    live_model: str = field(default_factory=lambda: _env("POLIGLOTA_LIVE_MODEL", "gemini-3.5-transcribe-live"))
    # Ambas medidas como contraproducentes en charlas tecnicas: el sesgo de
    # vocabulario no mejoro la precision y triplico la latencia inicial, y la
    # diarizacion multiplico por ocho el intervalo de refresco. Ver README.
    asr_adaptation: bool = field(default_factory=lambda: _env("POLIGLOTA_ASR_ADAPTATION", "0") == "1")
    asr_diarization: bool = field(default_factory=lambda: _env("POLIGLOTA_ASR_DIARIZATION", "0") == "1")
    mt_model: str = field(default_factory=lambda: _env("POLIGLOTA_MT_MODEL", "gemini-3.5-flash-lite"))
    # Respaldos ante saturacion del modelo preferido, en orden de preferencia.
    mt_fallbacks: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            m.strip() for m in _env(
                "POLIGLOTA_MT_FALLBACKS", "gemini-flash-lite-latest,gemini-flash-latest"
            ).split(",") if m.strip()
        )
    )

    whisper_model: str = field(default_factory=lambda: _env("POLIGLOTA_WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: _env("POLIGLOTA_WHISPER_DEVICE", "auto"))

    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    gemma_model: str = field(default_factory=lambda: _env("POLIGLOTA_GEMMA_MODEL", "gemma3:4b"))

    # Compuerta de voz: retiene el silencio para no pagar por transcribirlo.
    # APAGADA por defecto. Sobre habla continua no ahorra nada (medido: 0% en
    # una charla sin pausas) y toca el camino del audio, que es de donde sale la
    # precision. Su valor esta en la sala abierta sin nadie hablando: armado,
    # cambio de orador, coffee break. Encenderla para eventos de jornada larga.
    # Cada cuanto se traduce la frase en curso, en milisegundos. Es lo que
    # determina si quien lee traducido percibe el subtitulo tan vivo como quien
    # lee el original. El modelo refresca el original cada ~490 ms; con el
    # estrangulador en 1500 ms la version traducida iba tres veces mas lenta y
    # se sentia pesada. Bajarlo cuesta llamadas: cada revision del parcial es
    # una llamada mas. Subirlo si la cuota da 429.
    partial_mt_ms: int = field(default_factory=lambda: int(_env("POLIGLOTA_PARTIAL_MT_MS", "600")))

    vad_gate: bool = field(default_factory=lambda: _env("POLIGLOTA_VAD", "0") == "1")
    vad_threshold: float = field(default_factory=lambda: float(_env("POLIGLOTA_VAD_THRESHOLD", "0.006")))
    # Precio por millon de tokens, en USD. Queda en cero a proposito: el precio
    # de lista cambia y depende del contrato, asi que lo carga quien opera el
    # evento. Con cero, el panel muestra tokens y no inventa dinero.
    cost_per_mtok: float = field(default_factory=lambda: float(_env("POLIGLOTA_COST_PER_MTOK", "0")))

    # Identificacion de hablante por huella de voz. Se activa sola en cuanto hay
    # al menos una voz registrada. El margen es cuantas "dispersiones propias"
    # puede alejarse un clip de una voz registrada y seguir siendo ella; mas
    # alla, la frase sale como "Desconocido". Bajarlo hace mas "Desconocido";
    # subirlo hace mas confusiones entre registrados. Vacio: cada embedder usa
    # el suyo, calibrado con datos (2,5 el clasico, 1,7 el neuronal).
    speaker_margin: float | None = field(
        default_factory=lambda: float(_env("POLIGLOTA_SPEAKER_MARGIN", "0")) or None
    )
    speakers_file: str = field(default_factory=lambda: _env("POLIGLOTA_SPEAKERS_FILE", "data/speakers.json"))
    # auto | resemblyzer | mfcc. `auto` usa el neuronal si esta instalado.
    speaker_embedder: str = field(default_factory=lambda: _env("POLIGLOTA_SPEAKER_EMBEDDER", "auto"))

    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))

    # Audio: el pipeline entero trabaja en PCM16 mono 16 kHz little-endian,
    # que es el formato nativo de entrada de la Live API.
    sample_rate: int = 16000
    sample_width: int = 2
    channels: int = 1

    def require_key(self) -> str:
        if not self.gemini_api_key:
            raise RuntimeError(
                "Falta GEMINI_API_KEY. Copia .env.example a .env y pega tu key de "
                "https://aistudio.google.com/apikey  (o usa POLIGLOTA_ASR=whisper para modo local)."
            )
        return self.gemini_api_key


settings = Settings()

# Idiomas ofrecidos a la audiencia. El MVP exige en->es; el resto es fan-out gratis
# porque una sola transcripcion alimenta N traducciones.
LANGUAGES: dict[str, str] = {
    "es": "Espanol",
    "en": "English",
    "pt": "Portugues",
    "fr": "Francais",
    "it": "Italiano",
    "de": "Deutsch",
}
