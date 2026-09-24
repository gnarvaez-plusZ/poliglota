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

    live_model: str = field(default_factory=lambda: _env("POLIGLOTA_LIVE_MODEL", "gemini-3.8-live"))
    mt_model: str = field(default_factory=lambda: _env("POLIGLOTA_MT_MODEL", "gemini-flash-latest"))

    whisper_model: str = field(default_factory=lambda: _env("POLIGLOTA_WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: _env("POLIGLOTA_WHISPER_DEVICE", "auto"))

    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    gemma_model: str = field(default_factory=lambda: _env("POLIGLOTA_GEMMA_MODEL", "gemma3:4b"))

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
