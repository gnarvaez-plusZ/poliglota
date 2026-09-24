"""Prompt compartido por los motores de traduccion.

Tres decisiones que mueven la calidad mas que el modelo elegido:

1. Todos los idiomas destino salen en UNA sola llamada. Traducir a 5 idiomas
   cuesta practicamente lo mismo que a 1, porque el audio y el contexto se
   tokenizan una vez sola. Es la diferencia entre escalar y no escalar.
2. Se pasa el contexto reciente de la charla. Sin el, cada frase se traduce
   aislada y la terminologia baila entre oraciones.
3. Se pasa el glosario de la charla. Los nombres propios y la jerga tecnica no
   se traducen: "garbage collector" no es "recolector de basura" en una charla
   de JVM, y un ASR sin cebar escribe "cuber netes".
"""
from __future__ import annotations

SYSTEM = (
    "You translate live conference subtitles. You are fast, literal and consistent.\n"
    "Rules:\n"
    "- Translate the LINE only. Never explain, never add or remove information.\n"
    "- Keep technical terms, product names, acronyms and code identifiers in their "
    "original form. Do not localise jargon that the audience uses in English.\n"
    "- Preserve the register of a spoken talk. Subtitles are short: no flourish.\n"
    "- If the line is already in the target language, return it unchanged.\n"
    "- Return strict JSON: one key per requested language code, value = translation. "
    "No markdown, no extra keys."
)


def build_user_prompt(
    text: str,
    *,
    source_lang: str,
    targets: list[str],
    context: str = "",
    glossary: str = "",
) -> str:
    parts: list[str] = []
    if glossary:
        parts.append(f"TALK GLOSSARY (keep these spellings verbatim):\n{glossary}")
    if context:
        parts.append(f"RECENT CONTEXT (for terminology consistency, do not translate):\n{context}")
    parts.append(f"SOURCE LANGUAGE: {source_lang}")
    parts.append(f"TARGET LANGUAGES: {', '.join(targets)}")
    parts.append(f"LINE TO TRANSLATE:\n{text}")
    return "\n\n".join(parts)
