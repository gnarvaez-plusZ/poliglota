/**
 * Pruebas del buffer de subtítulos roll-up, con reloj falso.
 *
 * El tiempo mínimo de lectura sólo se puede verificar controlando el reloj: con
 * el real habría que dormir segundos por caso y la prueba sería lenta y frágil.
 *
 *   node tests/test_captions.mjs
 */
import { CaptionBuffer } from '../web/captions.js';

let clock = 0;
const now = () => clock;
const fails = [];

function check(name, cond, detail = '') {
  console.log(`  ${cond ? 'PASA ' : 'FALLA'} ${name}${cond || !detail ? '' : `   <- ${detail}`}`);
  if (!cond) fails.push(name);
}

function make(opts = {}) {
  clock = 0;
  return new CaptionBuffer({ now, ...opts });
}

// --- corte de línea ---
{
  const b = make();
  b.commit('Movimos el paso de embeddings atrás de un tópico de Kafka y la latencia bajó mucho');
  const lines = b.visible.history.map((l) => l.text);
  check('corta a 42 caracteres', lines.every((l) => l.length <= 42), JSON.stringify(lines));
  check('no parte palabras', lines.every((l) => !/\S-$/.test(l) || l.length === 42));
  check('conserva el texto', lines.join(' ').includes('embeddings'));
}

// --- palabra más larga que la línea ---
{
  const b = make();
  b.commit('Ver https://ejemplo.com/una/url/absurdamente/larga/que/no/entra/en/una/linea ahora');
  const lines = b.visible.history.map((l) => l.text);
  check('parte una palabra imposible', lines.every((l) => l.length <= 42), JSON.stringify(lines));
}

// --- tiempo mínimo de lectura ---
{
  const b = make({ maxHistory: 2 });
  b.commit('Primera linea corta.');
  b.commit('Segunda linea corta.');
  const before = b.visible.history.map((l) => l.text);
  b.commit('Tercera linea que deberia esperar.');
  const during = b.visible.history.map((l) => l.text);
  check('respeta el tiempo mínimo de lectura',
        during[0] === before[0],
        `desplazó a "${during[0]}" sin esperar`);

  clock += 2000;            // supera el piso de 1,5 s
  b.tick();
  const after = b.visible.history.map((l) => l.text);
  check('avanza cuando venció el tiempo', after[0] !== before[0], JSON.stringify(after));
}

// --- tope de cola: ir al día gana sobre el tiempo mínimo ---
{
  const b = make({ maxHistory: 2 });
  for (let i = 0; i < 8; i++) b.commit(`Linea numero ${i}.`);
  // Durante la ráfaga la pantalla nunca se desborda ni se queda en el principio:
  // al superarse el tope de cola se saltean líneas intermedias a propósito.
  const during = b.visible.history.map((l) => l.text);
  check('nunca muestra más de maxHistory', b.visible.history.length <= 2, JSON.stringify(during));
  check('durante la ráfaga ya salteó el principio',
        !during.some((t) => t.includes('numero 0')), JSON.stringify(during));

  // Las últimas líneas sí esperan su tiempo de lectura, porque la cola volvió a
  // estar bajo el tope. Al vencer, tienen que llegar a lo más reciente.
  clock += 4000;
  b.tick();
  const after = b.visible.history.map((l) => l.text);
  check('termina alcanzando la línea más reciente',
        after.some((t) => t.includes('numero 7')), JSON.stringify(after));
}

// --- cambio de hablante ---
{
  const b = make();
  b.commit('Una pregunta corta.', 'A');
  b.commit('Otra respuesta.', 'B');
  const joined = b.visible.history.map((l) => l.text).join(' | ');
  check('etiqueta el cambio de hablante', joined.includes('— B:'), joined);
}

// --- el render se dispara en cada cambio ---
{
  const painted = [];
  const b = make({ onRender: (v) => painted.push(v) });
  const n = painted.length;
  b.setPartial('texto en curso');
  check('un parcial repinta la pantalla', painted.length > n,
        'setPartial cambió el estado pero no avisó al render');
  check('el parcial llega al render',
        painted[painted.length - 1].partial.includes('curso'));
  const m = painted.length;
  b.commit('Una linea firme.');
  check('un final repinta la pantalla', painted.length > m);
}

// --- el parcial no toca el historial ---
{
  const b = make();
  b.commit('Linea firme.');
  const before = JSON.stringify(b.visible.history.map((l) => l.text));
  b.setPartial('texto en curso que cambia');
  b.setPartial('texto en curso que cambia todavia mas');
  const after = JSON.stringify(b.visible.history.map((l) => l.text));
  check('el parcial no reescribe el historial', before === after);
  check('el parcial se muestra aparte', b.visible.partial.includes('curso'));
}

// --- corrección en el lugar ---
{
  const b = make();
  b.commit('Texto con un herror.');
  const idx = b.visible.history[0].index;
  const ok = b.revise(idx, 'Texto con un error.');
  check('corrige la línea en su lugar',
        ok && b.visible.history[0].text === 'Texto con un error.',
        b.visible.history[0].text);
  check('la corrección no agrega líneas', b.visible.history.length === 1);
}

// --- inactividad ---
{
  const b = make({ idleMs: 5000 });
  b.commit('Algo que se dijo hace rato.');
  clock += 6000;
  b.tick();
  check('oculta el historial tras el silencio', b.visible.history.length === 0);
  b.setPartial('vuelve a hablar');
  check('reaparece al volver a hablar', b.visible.partial.includes('vuelve'));
}

console.log(fails.length ? `\n${fails.length} FALLAS: ${fails}` : '\nSUBTITULOS OK');
process.exit(fails.length ? 1 : 0);
