# Políglota

**Subtitulado y traducción simultánea en tiempo real para conferencias multi-sala.**

Una charla entra como audio en vivo. Sale como subtítulos en el idioma original y
traducidos a los idiomas que la audiencia esté pidiendo en ese momento. Varias
salas corren en paralelo en el mismo proceso.

Construido para la [Nerdearla Vibeathon 2026](https://nerdearla26.devpost.com/).

---

## Por qué existe

En una conferencia con sesiones en paralelo, la accesibilidad lingüística es un
problema de logística, no de modelos. Contratar intérpretes simultáneos para
cada sala no escala, y un subtitulador genérico falla justo donde más importa:
la jerga técnica y la consistencia de terminología a lo largo de una charla.

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

1. **Ingesta.** Cualquier fuente abre un WebSocket y empuja PCM16 mono de 16 kHz:
   captura desde el navegador (micrófono o pestaña), un alimentador por `ffmpeg`
   para archivos y streams, o cualquier cosa que hable WebSocket.
2. **Reconocimiento.** El audio va a `gemini-3.5-transcribe-live` en streaming.
   Las palabras aparecen mientras el orador habla y se corrigen solas.
3. **Traducción.** Cada frase cerrada se traduce **a todos los idiomas activos en
   una sola llamada**, con las frases anteriores como contexto y un glosario que
   impide localizar la jerga.
4. **Distribución.** Un bus pub/sub por sala reparte a N espectadores. Un
   espectador lento nunca frena a los demás.

## Números medidos

Sobre 50 s de charla técnica real, contra la API de producción:

| | Una sala | Dos salas simultáneas |
|---|---|---|
| Primer subtítulo | **2,0 s** | 2,1 – 5,2 s |
| Refresco del subtítulo | **~490 ms** | 443 – 466 ms |
| Traducción (es + pt, una llamada) | **~780 ms** | 0,8 – 1,5 s |
| Jerga técnica reconocida | **10/10** | 10/10 |

Reproducibles con `.venv/bin/python tests/test_live.py`.

## Idiomas

**Transcripción verificada en español, inglés y portugués**, más detección
automática. Se elige por sala; la audiencia elige aparte a qué idioma quiere leer.

| Pista | Idioma declarado | Términos acertados | Primer texto |
|---|---|---|---|
| Inglés | `en` | 3/4 | 1,9 s |
| Español | `es` | 4/4 | 2,5 s |
| Portugués | `pt` | 4/4 | 1,7 s |
| Español con jerga en inglés | `es` | 4/4 | 14,0 s |
| Español con jerga en inglés | `auto` | 4/4 | **1,9 s** |

La última fila es el caso real de una conferencia latinoamericana: el orador
habla español y dice *retrieval augmented generation*, *bottleneck* y *queue
depth* en inglés, sin traducirlos. Fijar el idioma funciona, pero **declarar
`auto` reduce el arranque de 14 s a 1,9 s** en ese audio mezclado. Para una
charla en Nerdearla, `auto` es la opción sensata.

El único defecto observado con `auto` es que el modelo a veces repite el
fragmento inicial (*"Buenas a todos. Hoy lesBuenas a todos. Hoy les quiero…"*).
Se corrige solo en la siguiente revisión de la línea.

La traducción va en **las seis direcciones** entre los tres idiomas, con la jerga
técnica intacta en todas:

```
ES → EN   Movimos el paso de embeddings atrás de un tópico de Kafka…
          We moved the embeddings step behind a Kafka topic and the p99 latency…
PT → ES   Tudo isso roda em Kubernetes, com um autoscaler horizontal…
          Todo esto corre en Kubernetes, con un autoscaler horizontal…
EN → PT   We trace every hop with eBPF and export the spans over gRPC…
          Rastreamos cada salto com eBPF e exportamos os spans via gRPC…
```

Reproducible con `.venv/bin/python tests/test_languages.py`. Hay además otros
tres idiomas de salida configurados (francés, italiano, alemán) que no están
medidos.

### Lo que medir cambió respecto del diseño inicial

Tres decisiones del diseño original resultaron equivocadas al contrastarlas
contra la API real. Quedan anotadas porque el camino importa tanto como el
resultado:

**El glosario no mejora el reconocimiento.** La idea era inyectar la jerga de la
charla como `adaptation_phrases` para que el modelo no escribiera *"cuber netes"*.
Medido: la configuración mínima acertó **10 de 10** términos (Kubernetes, Kafka,
eBPF, gRPC, P99…) con 1,7 s hasta el primer texto; agregar el sesgo de
vocabulario no mejoró la precisión y **triplicó la latencia inicial** (5,9 s).
El modelo ya conoce ese vocabulario. El glosario quedó donde sí aporta: en el
traductor, para que *"garbage collector"* no se vuelva *"recolector de basura"*
en una charla de JVM. El sesgo de ASR sigue disponible con
`POLIGLOTA_ASR_ADAPTATION=1` para nombres propios que el modelo no pueda conocer.

**La diarización cuesta demasiado.** Etiquetar quién habla multiplicó por ocho el
intervalo de refresco (490 ms → 2728 ms). Para un panel puede valer la pena;
para una charla no. Queda apagada, tras `POLIGLOTA_ASR_DIARIZATION=1`.

**El buffer de audio era el verdadero problema de latencia.** Con dos salas, el
primer subtítulo tardaba 27 s. Parecía un límite de cuota del proveedor. Era una
cola de audio de 256 chunks: si la sesión tardaba en establecerse, se acumulaban
**25 segundos de audio viejo** y el sistema arrancaba transcribiendo el pasado.
Acotarla a 5 s bajó el arranque a **2,1 s**, trece veces mejor. En subtitulado en
vivo es preferible perder una frase a quedar medio minuto atrás del orador; el
audio descartado se reporta en el panel para que el operador se entere.

## Decisiones de diseño

**Los subtítulos se leen como subtítulos.** El overlay usa el patrón *roll-up*
de la televisión en vivo: dos líneas firmes arriba y la frase en curso abajo,
cortadas a 42 caracteres. La regla que casi nunca se implementa es el **tiempo
mínimo en pantalla** —15 caracteres por segundo, con un piso de 1,5 s— sin el
cual tres frases seguidas del orador borran la primera antes de que nadie la
haya leído. Cuando el orador va más rápido de lo que se puede leer, la cola se
llena y el sistema abandona ese mínimo para no quedar atrás: ir al día importa
más, y lo salteado sigue en el panel completo. Tras unos segundos de silencio el
historial se oculta, para que texto viejo no parezca actual.

**El original no espera a la traducción.** El texto en el idioma de origen se
publica apenas sale del reconocedor; la traducción llega después como un parche
sobre el mismo número de línea. El subtítulo aparece a la velocidad del
reconocimiento, no a la del traductor.

**Cada línea tiene identidad estable.** El modelo no solo agrega texto: reescribe
lo que ya dijo cuando se corrige. Si cada revisión fuera una línea nueva, la
misma frase aparecería cinco veces en pantalla. Cada oración se identifica por su
posición en la charla, así una corrección **actualiza** la línea en vez de
duplicarla.

**Se traduce solo a los idiomas con público.** Una sala sin espectadores en
francés no gasta un token en francés, y todos los idiomas de una frase salen en
una única llamada: traducir a cinco cuesta casi lo mismo que a uno.

**Las charlas duran más que las sesiones.** El límite de sesión de audio es de 15
minutos; una charla dura 45. Se resuelve con compresión de ventana de contexto y
reanudación de sesión, renumerando las líneas para que la charla continúe sin
pisar lo ya emitido.

**Un proveedor saturado no puede callar la sala.** Bajo carga la API devolvió
`429` y `503` de forma habitual. El traductor reintenta con espera creciente,
baja a modelos de respaldo y aplica un cortacircuitos por modelo. Si todo falla,
la audiencia sigue viendo el idioma original.

## Motores intercambiables

Se eligen por variable de entorno, sin tocar código:

| | Nube (por defecto) | Local |
|---|---|---|
| **ASR** | `gemini-3.5-transcribe-live` | `faster-whisper` |
| **Traducción** | `gemini-3.5-flash-lite` | Gemma vía Ollama |
| **Necesita** | `GEMINI_API_KEY` | GPU recomendada |
| **Costo** | por minuto de audio | cero |

Hay además un motor `mock` que reproduce una charla escrita: permite correr el
sistema entero, y los tests, sin credenciales ni red.

## Arranque rápido

Requisitos: Python 3.11+ y una API key de [Google AI Studio](https://aistudio.google.com/apikey).

```bash
git clone <este-repo> && cd poliglota
cp .env.example .env         # pegá tu GEMINI_API_KEY
uv venv && uv pip install -e .
.venv/bin/poliglota
```

Abrí <http://localhost:8000>. Con Docker: `cp .env.example .env && docker compose up`.

### Credenciales

Una sola: `GEMINI_API_KEY`, de Google AI Studio. En modo local
(`POLIGLOTA_ASR=whisper`, `POLIGLOTA_MT=gemma`) no hace falta ninguna.

## Uso

| Página | Para quién |
|---|---|
| `/` | Operador del evento: todas las salas, métricas en vivo |
| `/capture/{sala}` | Cabina de sonido: captura micrófono o pestaña |
| `/room/{sala}` | Asistente: subtítulos con selector de idioma |
| `/overlay/{sala}` | OBS: overlay transparente para el stream |

```bash
scripts/demo.sh                               # dos salas simuladas, sin credenciales
scripts/feed.py auditorio charla.mp4 --lang en --title "Scaling RAG"
scripts/feed.py track-2 https://ejemplo.com/stream.m3u8 --lang en
```

## Pruebas

```bash
.venv/bin/python tests/test_pipeline.py    # pipeline completo, motor mock, sin credenciales
.venv/bin/python tests/test_live.py        # contra la API real, dos salas, audio real
.venv/bin/python tests/test_languages.py   # es/en/pt: transcripción y traducción en ambos sentidos
node tests/test_captions.mjs               # reglas de subtitulado roll-up, con reloj falso
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

## Sobre la métrica de latencia

El panel muestra **refresco del subtítulo** y **tiempo hasta el primer texto**,
no un "retraso punta a punta". El modelo acepta `word_timestamp` pero nunca
devuelve los tiempos por palabra, así que no hay forma de alinear una palabra del
subtítulo con el instante del audio que la originó. Cualquier retraso punta a
punta mostrado en vivo sería inventado, y una versión anterior de este panel lo
mostraba. Se prefiere reportar menos y que sea cierto.

## Licencia

Apache-2.0. Ver [LICENSE](LICENSE).

## Audio de prueba

Los archivos de `samples/` no se versionan. Para generar una pista de charla
técnica con la que probar el pipeline:

```bash
scripts/make_sample.py samples/charla-en.wav --lang en
ffmpeg -i samples/charla-en.wav -ac 1 -ar 16000 -f s16le samples/charla-full.raw
```
