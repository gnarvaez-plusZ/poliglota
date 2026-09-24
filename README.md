# Políglota

**Subtitulado y traducción simultánea en tiempo real para conferencias multi-sala.**

Una charla entra como audio en vivo. Sale como subtítulos en el idioma original y
traducidos a los idiomas que la audiencia esté pidiendo en ese momento, con un
retraso de menos de un segundo respecto del orador. Varias salas corren en
paralelo en el mismo proceso.

Construido para la [Nerdearla Vibeathon 2026](https://nerdearla26.devpost.com/).

---

## Por qué existe

En una conferencia con sesiones en paralelo, la accesibilidad lingüística es un
problema de logística, no de modelos. Contratar intérpretes simultáneos para cada
sala no escala, y los subtituladores automáticos genéricos fallan justo donde
más importa: la jerga técnica. Un ASR sin contexto escribe *"cuber netes"*, y un
traductor sin contexto convierte *"garbage collector"* en *"recolector de basura"*
en medio de una charla de JVM.

Políglota ataca las dos cosas: **ceba el reconocimiento con el vocabulario de la
charla** y **traduce con el contexto de las frases anteriores**, manteniendo un
glosario de términos que no se traducen.

## Cómo funciona

```
  fuente de audio          Políglota                        audiencia
 ┌────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐
 │ micrófono      │   │  Sala "auditorio"    │   │ 🇪🇸 espectador  → es      │
 │ pestaña / OBS  ├──▶│   ASR ──▶ traducción ├──▶│ 🇧🇷 espectador  → pt      │
 │ ffmpeg / RTMP  │ws │    │         │       │ws │ 🖥️  overlay OBS → es      │
 └────────────────┘   │    └── métricas ─────┤   └──────────────────────────┘
                      │  Sala "track-2" ...  │
                      └──────────────────────┘
```

1. **Ingesta.** Cualquier fuente abre un WebSocket y empuja PCM16 mono de 16 kHz.
   Hay tres clientes listos: captura desde el navegador (micrófono o pestaña),
   un alimentador por `ffmpeg` para archivos y streams, y cualquier cosa que
   hable WebSocket.
2. **Reconocimiento.** El audio va a un motor de ASR en streaming, cebado con el
   glosario de la charla. Los parciales salen a pantalla mientras el orador habla.
3. **Traducción.** Cada frase cerrada se traduce **a todos los idiomas activos en
   una sola llamada**, con las frases anteriores como contexto.
4. **Distribución.** Un bus pub/sub por sala reparte a N espectadores. Un
   espectador lento nunca frena a los demás.

### Las tres decisiones que definen el resultado

**El original no espera a la traducción.** El texto en el idioma de origen se
publica apenas sale del ASR; la traducción llega después como un parche sobre el
mismo número de secuencia. El subtítulo aparece a la velocidad del reconocimiento,
no a la del traductor.

**Se traduce solo a los idiomas con público.** Una sala sin espectadores en
francés no gasta un token en francés. El costo sigue a la demanda real, y todos
los idiomas de una frase salen en una única llamada: traducir a cinco cuesta casi
lo mismo que a uno.

**Las charlas duran más que las sesiones.** El límite de sesión de audio de la
Live API es de 15 minutos; una charla dura 45. Se resuelve con compresión de
ventana de contexto y reanudación de sesión: el servidor avisa antes de cortar y
el pipeline reconecta con el handle vigente sin perder el hilo.

## Motores intercambiables

Se eligen por variable de entorno, sin tocar código:

| | Nube (por defecto) | Local |
|---|---|---|
| **ASR** | Gemini Live API | `faster-whisper` |
| **Traducción** | Gemini Flash | Gemma vía Ollama |
| **Necesita** | `GEMINI_API_KEY` | GPU recomendada |
| **Costo** | por minuto de audio | cero |
| **Sirve para** | máxima calidad y latencia | sedes sin internet estable |

## Arranque rápido

Requisitos: Python 3.11+ y una API key de [Google AI Studio](https://aistudio.google.com/apikey).

```bash
git clone <este-repo> && cd poliglota
cp .env.example .env         # pegá tu GEMINI_API_KEY
uv venv && uv pip install -e .
.venv/bin/poliglota
```

Abrí <http://localhost:8000>.

Con Docker:

```bash
cp .env.example .env         # pegá tu GEMINI_API_KEY
docker compose up
```

### Credenciales

Una sola: `GEMINI_API_KEY`, de Google AI Studio. Es gratuita para el tier de
desarrollo. En modo local (`POLIGLOTA_ASR=whisper`, `POLIGLOTA_MT=gemma`) no hace
falta ninguna.

## Uso

| Página | Para quién |
|---|---|
| `/` | Operador del evento: todas las salas, métricas en vivo |
| `/capture/{sala}` | Cabina de sonido: captura micrófono o pestaña |
| `/room/{sala}` | Asistente: subtítulos con selector de idioma |
| `/overlay/{sala}` | OBS: overlay transparente para el stream |

### Demo de dos salas simultáneas

```bash
scripts/demo.sh
```

Levanta dos salas con audio real en paralelo y abre el panel.

### Alimentar una sala desde un archivo o stream

```bash
scripts/feed.py auditorio charla.mp4 --lang en --title "Scaling RAG in production"
scripts/feed.py track-2 https://ejemplo.com/stream.m3u8 --lang en
```

## API

```
GET    /api/health              estado y motores activos
GET    /api/rooms               todas las salas + métricas
POST   /api/rooms               crear sala
DELETE /api/rooms/{id}          cerrar sala
WS     /ws/ingest/{id}          entrada de audio (PCM16 16 kHz mono)
WS     /ws/view/{id}?lang=es    salida de subtítulos (JSON)
```

## Licencia

Apache-2.0. Ver [LICENSE](LICENSE).
