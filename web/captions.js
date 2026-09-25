/**
 * Buffer de subtítulos roll-up: historial de líneas firmes + línea en curso.
 *
 * Es el patrón de la televisión en vivo (CEA-608): dos o tres líneas visibles,
 * la de abajo escribiéndose y las de arriba subiendo. Sirve para que quien no
 * llegó a terminar de leer una línea la siga viendo un momento más.
 *
 * Tres reglas sostienen la legibilidad, y la tercera es la que casi nunca se
 * implementa:
 *
 *  - Corte a 42 caracteres, en espacios, nunca a mitad de palabra.
 *  - Una línea nueva por cambio de hablante, aunque la anterior esté a medias.
 *  - Tiempo mínimo en pantalla proporcional al largo (15 caracteres por segundo,
 *    con un piso de 1,5 s). Sin esto, tres frases seguidas del orador borran la
 *    primera antes de que nadie la haya leído.
 *
 * El tiempo mínimo genera una tensión: si el orador habla más rápido de lo que
 * se puede leer, la cola crece y el subtítulo se atrasa respecto del audio. Por
 * eso hay un tope de cola: superado, se abandona el tiempo mínimo y se prefiere
 * ir al día con el orador. Quien quiera releer tiene el panel completo.
 */

const CPS = 15;              // caracteres por segundo de lectura
const MIN_HOLD_MS = 1500;    // piso de permanencia de una línea
const MAX_QUEUE = 3;         // líneas en espera antes de soltar el tiempo mínimo

export class CaptionBuffer {
  constructor({
    maxHistory = 2,
    maxChars = 42,
    idleMs = 6000,
    now = () => performance.now(),
    onRender,
  } = {}) {
    Object.assign(this, { maxHistory, maxChars, idleMs, now, onRender });
    this.history = [];      // [{ text, shownAt, speaker }]
    this.pending = [];      // líneas firmes esperando lugar
    this.partial = '';
    this.lastSpeaker = null;
    this.lastActivity = this.now();
    this.hidden = false;
  }

  /** Texto en curso: reemplaza siempre la línea de abajo, nunca toca el historial. */
  setPartial(text, speaker = null) {
    const lines = this.#wrap(this.#prefix(speaker, false) + (text || '').trim());
    // Del parcial solo se muestra la última línea: las anteriores ya son legibles
    // arriba y repetirlas haría parpadear la pantalla.
    this.partial = lines.length ? lines[lines.length - 1] : '';
    this.#activity();
    this.#render();
  }

  /** Texto firme: entra a la cola y sube al historial cuando hay lugar. */
  commit(text, speaker = null) {
    const clean = (text || '').trim();
    if (!clean) return;
    const forceBreak = speaker != null && speaker !== this.lastSpeaker;
    for (const line of this.#wrap(this.#prefix(speaker, forceBreak) + clean)) {
      this.pending.push({ text: line, speaker });
    }
    if (speaker != null) this.lastSpeaker = speaker;
    this.partial = '';
    this.#activity();
    this.#drain();
  }

  /**
   * Corrección: el motor reescribió una línea que ya estaba en pantalla.
   * Se actualiza en su lugar, sin animación ni reordenamiento, para que la
   * corrección no llame la atención más que el contenido.
   */
  revise(index, text) {
    const target = this.history.find((l) => l.index === index);
    if (!target) return false;
    const [first] = this.#wrap(text.trim());
    if (first && first !== target.text) {
      target.text = first;
      this.#render();
    }
    return true;
  }

  /** Avanza la cola respetando el tiempo mínimo de lectura. Idempotente. */
  tick() {
    this.#drain();
    if (!this.hidden && this.now() - this.lastActivity > this.idleMs) {
      // Silencio largo: dejar texto viejo en pantalla lo hace parecer actual.
      this.hidden = true;
      this.#render();
    }
  }

  get visible() {
    return { history: this.hidden ? [] : this.history, partial: this.hidden ? '' : this.partial };
  }

  // ---------- interno ----------

  #drain() {
    // Si la cola pasó el tope, el subtítulo ya quedó demasiado atrás del orador.
    // Se vacía entera de una vez y se muestra lo más reciente, aunque eso saltee
    // líneas: ir al día importa más que no perderse nada, y lo salteado sigue
    // estando en el panel completo. Se decide antes del bucle a propósito, para
    // no volver a frenar en cuanto la cola baje del tope.
    const catchUp = this.pending.length > MAX_QUEUE;

    while (this.pending.length) {
      const oldest = this.history[0];
      if (!catchUp && this.history.length >= this.maxHistory && oldest) {
        const held = this.now() - oldest.shownAt;
        if (held < this.#holdFor(oldest.text)) break;  // todavía se está leyendo
      }
      const next = this.pending.shift();
      this.history.push({ ...next, shownAt: this.now(), index: this.#nextIndex++ });
      if (this.history.length > this.maxHistory) this.history.shift();
    }
    this.#render();
  }

  #nextIndex = 0;

  #holdFor(text) {
    return Math.max(MIN_HOLD_MS, (text.length / CPS) * 1000);
  }

  #prefix(speaker, force) {
    return force && speaker != null ? `— ${speaker}: ` : '';
  }

  #wrap(text) {
    const out = [];
    let current = '';
    for (let word of text.split(/\s+/).filter(Boolean)) {
      // Una URL o un identificador más largo que la línea entera se parte por
      // la fuerza: es feo, pero menos que desbordar la pantalla.
      while (word.length > this.maxChars) {
        if (current) { out.push(current); current = ''; }
        out.push(word.slice(0, this.maxChars - 1) + '-');
        word = word.slice(this.maxChars - 1);
      }
      const candidate = current ? `${current} ${word}` : word;
      if (candidate.length > this.maxChars && current) {
        out.push(current);
        current = word;
      } else {
        current = candidate;
      }
    }
    if (current) out.push(current);
    return out;
  }

  #activity() {
    this.lastActivity = this.now();
    if (this.hidden) {
      this.hidden = false;
      this.history = [];
      this.pending = [];
    }
  }

  #render() {
    this.onRender?.(this.visible);
  }
}
