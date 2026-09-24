"""Glosario de charla: lo que separa un subtitulo util de uno gracioso.

Un ASR generico transcribe "cuber netes" y "arreglar" donde el orador dijo
"Kubernetes" y "RAG". Con el titulo y el abstract de la charla ya alcanza para
cebar al modelo con la jerga correcta, y el mismo glosario viaja al traductor
para que no localice terminos que la audiencia usa en ingles.
"""
from __future__ import annotations

import re

# Terminos que en una charla tecnica en Latinoamerica se dicen en ingles y no
# deben traducirse aunque el diccionario tenga equivalente.
BASE_TERMS = [
    "deploy", "deployment", "build", "commit", "merge", "pull request", "rollback",
    "endpoint", "backend", "frontend", "framework", "container", "cluster", "pod",
    "Kubernetes", "Docker", "Terraform", "Prometheus", "Grafana", "Kafka", "Redis",
    "Postgres", "serverless", "edge", "latency", "throughput", "observability",
    "embedding", "embeddings", "fine-tuning", "prompt", "token", "tokens",
    "inference", "benchmark", "dataset", "pipeline", "streaming", "batch",
    "LLM", "RAG", "MCP", "API", "SDK", "CLI", "GPU", "CPU", "TPU", "SLA", "SLO",
]

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+._-]*")
# Palabra que abre oracion: va en mayuscula por gramatica, no por ser un termino.
_SENTENCE_START = re.compile(r"(?:^|[.!?:;\n]\s*)([A-Za-z][A-Za-z0-9+._-]*)")


def _clean(word: str) -> str:
    return word.strip(".,;:!?()[]\"'")


def extract(title: str = "", abstract: str = "", speakers: str = "") -> list[str]:
    """Saca terminos candidatos de los metadatos de la charla.

    Heuristica deliberadamente simple: siglas, nombres propios y palabras con
    puntuacion interna (gRPC, Node.js, scikit-learn) son justo los tokens que un
    ASR escribe mal y que un traductor no deberia tocar. Se descartan las
    palabras que solo estan en mayuscula por abrir oracion.
    """
    text = f"{title}. {abstract}"
    openers = {_clean(m.group(1)) for m in _SENTENCE_START.finditer(text)}

    found: dict[str, None] = {}
    for name in (s.strip() for s in speakers.split(",")):
        if name:
            found[name] = None

    for raw in _WORD.findall(text):
        word = _clean(raw)
        if len(word) < 3:
            continue
        internal_punct = any(c in word[1:-1] for c in ".-+")
        acronym = word.isupper() and len(word) <= 6
        # Mayuscula interna: gRPC, eBPF, iOS. Casi nunca ocurre en prosa normal.
        camel = any(c.isupper() for c in word[1:])
        if acronym or internal_punct or camel:
            found[word] = None
        elif word[0].isupper() and word not in openers:
            found[word] = None
    return list(found)


def build(title: str = "", abstract: str = "", speakers: str = "", extra: list[str] | None = None) -> str:
    """Devuelve el glosario listo para inyectar en el prompt."""
    terms: dict[str, None] = {}
    for t in extract(title, abstract, speakers):
        terms[t] = None
    for t in (extra or []):
        terms[t.strip()] = None
    for t in BASE_TERMS:
        terms.setdefault(t, None)
    return ", ".join(t for t in terms if t)
