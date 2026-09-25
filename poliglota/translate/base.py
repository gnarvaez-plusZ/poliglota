"""Contrato comun de los motores de traduccion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TranslationResult:
    # {codigo_idioma: traduccion}
    texts: dict[str, str] = field(default_factory=dict)
    # Tokens informados por el motor. Cero en los motores locales, que no facturan.
    tokens: int = 0
    # Reintentos que costo obtener este resultado. Se vigila en el panel: si sube,
    # el proveedor esta saturado y conviene bajar de modelo.
    retries: int = 0

    def __bool__(self) -> bool:
        return bool(self.texts)


class Translator(Protocol):
    name: str

    async def translate(
        self,
        text: str,
        *,
        source_lang: str,
        targets: list[str],
        context: str = "",
        glossary: str = "",
    ) -> TranslationResult:
        """Traduce una linea a todos los `targets` en una sola pasada."""
        ...


class NullTranslator:
    """Pasa el texto sin tocar. Util para medir la latencia pura del ASR."""

    name = "none"

    async def translate(
        self, text: str, *, source_lang: str, targets: list[str], **_
    ) -> TranslationResult:
        return TranslationResult({t: text for t in targets if t != source_lang})
