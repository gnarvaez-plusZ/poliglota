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

## Quién habla

Cada hablante registra su voz una vez (`/enroll`: nombre, 12 segundos hablando,
probar) y desde ahí los subtítulos salen como **`Juan: …`**, **`Pedro: …`** y, para
cualquier voz no registrada, **`Desconocido: …`**. Sirve para paneles, entrevistas
y las preguntas del público.

No es la diarización de Gemini, a propósito: medida, multiplica por ocho el
refresco del subtítulo y devuelve etiquetas anónimas. Esto corre local sobre el
audio que ya pasa por la sala, no agrega llamadas a la API, y decide por frase
con una votación sobre ventanas de 3 s. La huella se compara con la
**dispersión propia** de cada voz registrada: un clip es "Juan" si está a menos
de N dispersiones de Juan, con N calibrado con datos.

Hay dos embedders:

| | `mfcc` (sin dependencias) | `resemblyzer` (`uv pip install -e ".[voces]"`) |
|---|---|---|
| Distingue | hombre / mujer, voces muy distintas | dos voces del mismo registro |
| Voz registrada reconocida | 11/11 y 11/11 | 11/11 y 11/11 |
| Voz ajena rechazada (3 s) | **0/26** | **24/26** (100 % con clips de 5 s) |
| Costo por clip | 5 ms | ~85 ms en CPU |

Medido con tres voces de TTS de Gemini leyendo el mismo texto (Charon, Kore y
Aoede), registrando dos y probando con tramos que el registro no vio. El
embedder clásico **no sabe decir "Desconocido"** cuando la voz ajena se parece a
una registrada: confundió a Aoede con Kore en 26 de 26 tramos. Por eso `auto`
usa el neuronal si está instalado. En un cambio de hablante, la primera frase
puede atribuirse al anterior: la ventana de 3 s todavía lo contiene.

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

**Cada asistente lee en su idioma, sin instalar nada.** El overlay proyectado en
la sala lleva un QR (`/overlay/{sala}?qr=1`). Quien lo escanea abre los
subtítulos en su teléfono y elige su idioma ahí. Es la diferencia entre una
pantalla única que obliga a todos al mismo idioma y que cada persona lea en el
suyo, que es justamente el problema que el sistema vino a resolver. El código se
arma con el host por el que entró el pedido, así apunta a la IP de la red del
evento y no a `localhost`.

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

**El costo se mide, no se estima.** El panel calcula el gasto por hora de sala
sobre el consumo real que informa la API, y proyecta la jornada completa del
evento. La tarifa la carga quien opera (`POLIGLOTA_COST_PER_MTOK`): sin ella el
panel muestra tokens y no inventa dinero, porque el precio de lista cambia y
depende del contrato.

**Se puede no pagar el silencio, pero está apagado por defecto.** Hay una
compuerta de voz (`POLIGLOTA_VAD=1`) que retiene el silencio para no facturar
transcripciones de nada. Medida sobre habla continua **no ahorra nada**: su
valor está en la sala abierta sin nadie hablando —armado, cambio de orador,
coffee break— donde retiene más del 90%. Viene apagada porque toca el camino del
audio, que es de donde sale la precisión, y ahorrar tokens comiéndose el ataque
de una palabra sería un mal negocio. Las pruebas verifican que el habla pasa
byte por byte intacta, con preroll para no perder la primera sílaba y tolerancia
a las pausas cortas dentro de una frase.

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

Para correrlo con `uvicorn` a mano, agregá `--timeout-graceful-shutdown 5`: sin
eso, un espectador con la pestaña abierta impide que el proceso muera al
apagarlo (`.venv/bin/poliglota` ya lo trae puesto).

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
| `/overlay/{sala}?qr=1` | OBS o proyector de sala: overlay transparente con QR |
| `/qr/{sala}` | QR suelto, para proyectar o imprimir |
| `/enroll` | Registrar las voces de los oradores para que los subtítulos lleven nombre |

```bash
scripts/demo.sh                               # dos salas simuladas, sin credenciales
scripts/feed.py auditorio charla.mp4 --lang en --title "Scaling RAG"
scripts/feed.py track-2 https://ejemplo.com/stream.m3u8 --lang en

# Capturar lo que suene en la máquina: sirve para alimentar una sala desde el
# navegador, una videollamada o la consola de sonido, sin depender de que el
# navegador sepa compartir audio de pestaña (Firefox no puede).
scripts/feed.py auditorio default --device pulse --lang auto
```

## Pruebas

```bash
.venv/bin/python tests/test_pipeline.py    # pipeline completo, motor mock, sin credenciales
.venv/bin/python tests/test_live.py        # contra la API real, dos salas, audio real
.venv/bin/python tests/test_languages.py   # es/en/pt: transcripción y traducción en ambos sentidos
node tests/test_captions.mjs               # reglas de subtitulado roll-up, con reloj falso
.venv/bin/python tests/test_gate.py         # compuerta de voz: retiene silencio sin tocar el habla
.venv/bin/python tests/test_speakers.py     # quién habla: voces sintéticas y de TTS, ambos embedders
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
