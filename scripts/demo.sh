#!/usr/bin/env bash
# Levanta dos salas en paralelo, el requisito de sesiones simultaneas del reto.
# Sin argumentos usa el motor mock, asi corre sin credenciales ni audio.
#
#   scripts/demo.sh                          # dos salas simuladas
#   scripts/demo.sh charla-a.mp4 charla-b.mp4  # dos salas con audio real
set -euo pipefail

SERVER="${SERVER:-http://localhost:8000}"
cd "$(dirname "$0")/.."

if ! curl -sf "$SERVER/api/health" >/dev/null; then
  echo "El servidor no responde en $SERVER. Arrancalo con:  .venv/bin/poliglota" >&2
  exit 1
fi

# Con audio real dejamos que mande el motor configurado en el servidor;
# sin audio caemos al mock para que la demo corra sin credenciales.
if [ $# -ge 2 ]; then
  ASR_JSON="null"; MT_JSON="null"
else
  # Sin audio ni credenciales: mock de punta a punta.
  ASR_JSON='"mock"'; MT_JSON='"none"'
fi

room() {
  curl -sf -X POST "$SERVER/api/rooms" -H 'content-type: application/json' \
    -d "{\"id\":\"$1\",\"title\":\"$2\",\"speakers\":\"$3\",\"abstract\":\"$4\",\"source_lang\":\"en\",\"asr_engine\":$ASR_JSON,\"mt_engine\":$MT_JSON}" \
    >/dev/null && echo "  sala '$1' creada"
}

echo "Creando dos salas simultaneas..."
room auditorio "Scaling RAG in production" "Ada Lovelace" \
  "We moved embeddings to Kafka and run on Kubernetes with an HPA driven by queue depth. SLO 200ms."
room track-2 "Observability with eBPF" "Alan Turing" \
  "Tracing syscalls with eBPF and Prometheus, shipping spans to Grafana Tempo over gRPC."

if [ $# -ge 2 ]; then
  echo "Alimentando con audio real (ritmo tiempo real)..."
  scripts/feed.py auditorio "$1" --lang en --title "Scaling RAG in production" &
  scripts/feed.py track-2  "$2" --lang en --title "Observability with eBPF" &
fi

echo
echo "  Panel:       $SERVER/"
echo "  Subtitulos:  $SERVER/room/auditorio   |  $SERVER/room/track-2"
echo "  Overlay OBS: $SERVER/overlay/auditorio?lang=es"
command -v xdg-open >/dev/null && xdg-open "$SERVER/" >/dev/null 2>&1 || true
[ $# -ge 2 ] && wait
