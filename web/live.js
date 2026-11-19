// Live emulator + policy session for the web page.
//
// Everything here was validated in scripts/rt_play.html: the locally built
// Genesis Plus GX core (which exports retro_get_memory_data), the pure-JS policy
// forward, the 84x84 Observer, and the 27 four-frame action macros.
//
// A session is: boot core -> load ROM from bytes -> unserialize a fixture ->
// then once per 4 emulator frames: observe, decide, act.

import { Policy } from "./policy.js?v=29";
import { Observer, decodeFrame } from "./observer.js?v=9";
import { expandAction } from "./actions.js?v=29";
import { download, withDeadline } from "./assets.js?v=29";
import { t } from "./i18n.js?v=29";

const RETRO_MEMORY_SYSTEM_RAM = 2;

// Work-RAM offsets, from chunppo/common.py.  The RAM as exposed is little-endian
// pairs (their byte() helper does ``raw[off ^ 1]``).
const P1_X = 0x8006, P2_X = 0x8286;
const P1_HP = 0x8042, P2_HP = 0x82C2;
const LOCK = 0x96B2;

/**
 * Ordered input delay, matching chunppo/common.py FIFO.
 * Each physical frame the packet may be delayed 1..5 frames with probability p;
 * later packets never overtake earlier ones.  ``base`` adds a fixed delay.
 */
export class InputFifo {
  constructor(jitter = 0) {
    this.p = jitter;
    this.pending = [];
    this.current = 0;
    this.tick = 0;
    this.release = 0;
    this.lastDelay = 0;
  }

  reset() {
    this.pending = [];
    this.current = 0;
    this.tick = 0;
    this.release = 0;
    this.lastDelay = 0;
  }

  /** Push this frame's request and return the mask to apply now. */
  next(request, base = 0) {
    const delayed = this.p > 0 && Math.random() < this.p;
    const d = delayed ? 1 + Math.floor(Math.random() * 5) : 0;
    this.release = Math.max(this.tick + base + d, this.release);
    this.lastDelay = this.release - this.tick;
    this.pending.push([this.release, request]);
    while (this.pending.length && this.pending[0][0] <= this.tick) {
      this.current = this.pending.shift()[1];
    }
    this.tick++;
    return this.current;
  }
}

// Bump this whenever a ladder fixture is regenerated.
export const FIXTURE_VERSION = 4;
const compiledCores = new Map();
const policies = new Map();
const preparations = new Map();

/**
 * Run one download/parse once and share it between the page-load warm-up and
 * the real start, so picking a ROM never repeats work already in flight.
 * Progress listeners are broadcast to, and a failure clears the slot so the
 * next attempt really retries instead of awaiting a rejected promise.
 */
function prepare(key, start, onProgress) {
  let slot = preparations.get(key);
  if (!slot) {
    const listeners = new Set();
    const promise = start((bytes, total) => {
      for (const listener of listeners) listener(bytes, total);
    }).catch((error) => {
      preparations.delete(key);
      throw error;
    });
    slot = { promise, listeners };
    preparations.set(key, slot);
  }
  if (onProgress) {
    slot.listeners.add(onProgress);
    slot.promise.finally(() => slot.listeners.delete(onProgress)).catch(() => {});
  }
  return slot.promise;
}

/** Fetch, compile and initialise nothing yet: just have the core ready to boot. */
export function prepareCore(corePath = "cores/", { onProgress } = {}) {
  const wasmUrl = `${corePath}genesis_plus_gx.wasm`;
  return prepare(`core:${wasmUrl}`, report => (async () => {
    const [module] = await Promise.all([
      compiledCores.get(wasmUrl) ?? download(wasmUrl, { onProgress: report })
        .then(bytes => withDeadline(WebAssembly.compile(bytes), 15_000, t("core.compileTimeout"))),
      loadCoreScript(`${corePath}genesis_plus_gx.js`),
    ]);
    compiledCores.set(wasmUrl, module);
  })(), onProgress);
}

/** Load the policy weights once; later sessions reuse the parsed network. */
export function preparePolicy(modelPath = "models/", { onProgress } = {}) {
  return prepare(`policy:${modelPath}`, report => (async () => {
    const policy = policies.get(modelPath) ?? await Policy.load(modelPath, { onProgress: report });
    policies.set(modelPath, policy);
    return policy;
  })(), onProgress);
}

/** Download one stage save in the background; loadFixture reuses the bytes. */
export function prepareFixture(url, { onProgress } = {}) {
  return prepare(`fixture:${url}`, report => download(url, { onProgress: report }), onProgress);
}

async function loadCoreScript(url, signal) {
  if (window.createGenesisPlusGX) return;
  if (signal?.aborted) throw signal.reason;
  const script = document.createElement("script");
  let cancel;
  const ready = new Promise((resolve, reject) => {
    cancel = () => reject(signal.reason);
    script.onload = resolve;
    script.onerror = () => reject(new Error(t("core.scriptFailed")));
    signal?.addEventListener("abort", cancel, { once: true });
    script.src = url;
    document.head.appendChild(script);
  });
  try {
    await withDeadline(ready, 25_000, t("core.scriptTimeout"));
    if (typeof window.createGenesisPlusGX !== "function") throw new Error(t("core.scriptInvalid"));
  } finally {
    signal?.removeEventListener("abort", cancel);
    script.onload = script.onerror = null;
    script.remove();
  }
}

export class LiveSession {
  constructor(opts = {}) {
    this.corePath = opts.corePath ?? "cores/";
    this.modelPath = opts.modelPath ?? "models/";
    this.fixtureUrl = opts.fixtureUrl ?? `cores/fixtures/ladder/stage01.state.gz?v=${FIXTURE_VERSION}`;
    this.onFrame = opts.onFrame ?? null;

    this.M = null;
    this.disposed = false;
    this.pendingAudio = null;
    this.cb = null;
    this.frameCopy = null;
    this.ramPtr = 0;
    this.joy = 0;
    this.inputPaused = false;
    this.interruptedMacro = false;
    this.fifo = new InputFifo(0.05);
    this.baseDelay = 0;             // fixed capture->execution delay, frames
    this.policy = null;
    this.observer = null;
    this.macroFrame = 0;
    this.pixelCanvas = document.createElement("canvas");
    this.stats = { decisions: 0, frames: 0, p1KO: 0, p2KO: 0 };
  }

  async loadCore({ signal, onProgress } = {}) {
    await prepareCore(this.corePath, { onProgress });
    if (signal?.aborted) throw signal.reason;
    const module = compiledCores.get(`${this.corePath}genesis_plus_gx.wasm`);
    const M = await withDeadline(window.createGenesisPlusGX({
      print: () => {},
      printErr: () => {},
      instantiateWasm(imports, receive) {
        const instance = new WebAssembly.Instance(module, imports);
        receive(instance, module);
        return instance.exports;
      },
    }), 15_000, t("core.initTimeout"));
    if (signal?.aborted) throw signal.reason;
    this.M = M;

    const envCb = M.addFunction((cmd, data) => {
      switch (cmd) {
        case 9: case 30: {
          const p = M._malloc(64); M.stringToUTF8("/tmp", p, 64);
          M.HEAP32[data >> 2] = p; return 1;
        }
        case 10: return 1;                       // SET_PIXEL_FORMAT (RGB565)
        case 1: case 11: case 16: case 18: case 34: return 1;
        case 2: M.HEAP32[data >> 2] = 1; return 1;   // GET_OVERSCAN
        case 3: M.HEAP32[data >> 2] = 0; return 1;   // GET_CAN_DUPE
        case 36: return 1;
        default: return 0;
      }
    }, "iii");

    const videoCb = M.addFunction((data, w, h, pitch) => {
      if (!data) return;
      const need = pitch * h;
      if (!this.frameCopy || this.frameCopy.length < need) this.frameCopy = new Uint8Array(need);
      this.frameCopy.set(M.HEAPU8.subarray(data, data + need));
      this.cb = { w, h, pitch };
      if (this.onFrame) this.onFrame(this);
    }, "viiii");

    M._retro_set_environment(envCb);
    M._retro_set_video_refresh(videoCb);
    M._retro_set_audio_sample_batch(M.addFunction((data, frames) => {
      this._pushAudio(data, frames);
      return frames;
    }, "iii"));
    M._retro_set_input_poll(M.addFunction(() => {}, "v"));
    M._retro_set_input_state(M.addFunction(
      (port, dev, idx, id) => (port === 0 && ((this.joy >> id) & 1) ? 1 : 0), "iiiii"));

    const si = M._malloc(32);
    M._retro_get_system_info(si);
    M._retro_init();
    return M;
  }

  async loadRom(romBytes) {
    const M = this.M;
    M.FS.writeFile("/tmp/game.md", new Uint8Array(romBytes));
    const pp = M._malloc(64);
    M.stringToUTF8("/tmp/game.md", pp, 64);
    const gi = M._malloc(16);
    M.HEAP32[(gi >> 2) + 0] = pp;
    if (M._retro_load_game(gi) !== 1) throw new Error(t("core.romFailed"));
    M._retro_reset();
    this.ramPtr = M._retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM);
    return this.ramPtr;
  }

  async loadFixture(url = this.fixtureUrl, { signal, onProgress } = {}) {
    const M = this.M;
    let st = await prepareFixture(url, { onProgress });
    if (signal?.aborted) throw signal.reason;

    // Ladder fixtures ship gzipped (1 MB of state compresses to ~40 KB).
    if (st[0] === 0x1f && st[1] === 0x8b) {
      if (typeof DecompressionStream !== "function") {
        throw new Error(t("fixture.gzipUnsupported"));
      }
      const stream = new Blob([st]).stream().pipeThrough(new DecompressionStream("gzip"));
      const data = await withDeadline(new Response(stream).arrayBuffer(), 10_000, t("fixture.gzipTimeout"));
      st = new Uint8Array(data);
    }

    if (signal?.aborted) throw signal.reason;
    if (st.length !== M._retro_serialize_size()
        || new TextDecoder().decode(st.subarray(0, 16)) !== "GENPLUS-GX 1.7.6") {
      throw new Error(t("fixture.invalid"));
    }
    const sp = M._malloc(st.length);
    M.HEAPU8.set(st, sp);
    const ok = M._retro_unserialize(sp, st.length) === 1;
    if (!ok) throw new Error(t("fixture.loadFailed", { url }));
    M._retro_run();
    // Deliberately not freed: the core's fast-savestate path can keep reading the
    // buffer, and one 1 MB allocation per session is not worth the risk.
    return ok;
  }

  async loadPolicy({ signal, onProgress } = {}) {
    const policy = await preparePolicy(this.modelPath, { onProgress });
    if (signal?.aborted) throw signal.reason;
    this.policy = policy;
    return this.policy;
  }

  ram16(off) {
    const h = this.M.HEAPU8;
    return h[this.ramPtr + off] | (h[this.ramPtr + off + 1] << 8);
  }

  get p1hp() { return this.ram16(P1_HP); }
  get p2hp() { return this.ram16(P2_HP); }
  get locked() { return this.ram16(LOCK) === 255; }

  async initAudio() {
    const Audio = window.AudioContext;
    if (!Audio || this.disposed || this.audio || this.pendingAudio) return;
    const av = this.M._malloc(40);
    this.M._retro_get_system_av_info(av);
    const sampleRate = new DataView(this.M.HEAPU8.buffer).getFloat64(av + 32, true);
    this.M._free(av);
    const ctx = new Audio({ sampleRate, latencyHint: "interactive" });
    this.pendingAudio = ctx;
    // Start resuming while a click still has user activation. Some browsers
    // defer the promise until another gesture, so startup never waits for it.
    ctx.resume().catch(() => {});
    try {
      await withDeadline(
        ctx.audioWorklet.addModule(new URL("./audio-worklet.js?v=15", import.meta.url)),
        5_000, t("audio.moduleTimeout"),
      );
      if (this.disposed) return;
      const node = new AudioWorkletNode(ctx, "genesis-audio", { outputChannelCount: [2] });
      this.audioStats = { framesIn: 0, framesOut: 0, underruns: 0 };
      node.port.onmessage = ({ data }) => Object.assign(this.audioStats, data);
      node.connect(ctx.destination);
      this.audio = { ctx, node };
    } finally {
      this.pendingAudio = null;
      if (!this.audio && ctx.state !== "closed") ctx.close().catch(() => {});
    }
  }

  _pushAudio(data, frames) {
    if (this.audio?.ctx.state !== "running" || !frames) return;
    const stereo = new Int16Array(this.M.HEAPU8.buffer, data, frames * 2).slice();
    this.audio.node.port.postMessage(stereo, [stereo.buffer]);
    this.audioStats.framesIn += frames;
  }

  resumeAudio() {
    const ctx = this.audio?.ctx;
    return ctx && ctx.state !== "running" && ctx.state !== "closed" ? ctx.resume() : Promise.resolve();
  }

  /** Current frame as a 256x200 canvas (cropped like the replay pipeline). */
  paint() {
    if (!this.frameCopy || !this.cb) return null;
    const { w, h, pitch } = this.cb;
    const cropW = Math.min(256, w);
    const cropH = Math.min(200, h);
    const x0 = (w - cropW) >> 1;
    // This game's 256x224 frame puts the 200-line playfield at row 16 (16 dark
    // lines above, 8 below).  Centring the crop would keep 4 of those top lines
    // as a black band across the top of the canvas.
    const y0 = h >= cropH + 16 ? h - cropH - 8 : (h - cropH) >> 1;
    const cv = this.pixelCanvas;
    if (cv.width !== cropW || cv.height !== cropH) { cv.width = cropW; cv.height = cropH; }
    const ctx = cv.getContext("2d");
    if (!this.pixelImage || this.pixelImage.width !== cropW || this.pixelImage.height !== cropH) {
      this.pixelImage = ctx.createImageData(cropW, cropH);
    }
    const img = this.pixelImage;
    const d = img.data;
    for (let y = 0; y < cropH; y++) {
      const row = (y + y0) * pitch;
      for (let x = 0; x < cropW; x++) {
        const o = row + (x + x0) * 2;
        const v = this.frameCopy[o] | (this.frameCopy[o + 1] << 8);
        const r = (v >> 11) & 0x1f, g = (v >> 5) & 0x3f, b = v & 0x1f;
        const i = (y * cropW + x) * 4;
        d[i] = (r << 3) | (r >> 2);
        d[i + 1] = (g << 2) | (g >> 4);
        d[i + 2] = (b << 3) | (b >> 2);
        d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return cv;
  }

  async initObserver() {
    const rgb = decodeFrame(this.frameCopy, this.cb.w, this.cb.h, this.cb.pitch, 2);
    this.observer = new Observer();
    this.observer.reset(rgb, this.cb.w, this.cb.h);
    return this.observer;
  }

  setInputPaused(paused) {
    if (this.inputPaused === paused) return;
    this.inputPaused = paused;
    if (paused) {
      this.interruptedMacro = true;
      this.fifo.reset();
      this.joy = 0;
    }
  }

  /** Fixed input delay in frames (0 = none). */
  setDelay(frames) {
    this.baseDelay = Math.max(0, Math.min(30, frames | 0));
    return this.baseDelay;
  }

  /** Probability that a frame is delayed by 1..5 frames. */
  setJitter(p) {
    const v = Math.max(0, Math.min(1, Number(p) || 0));
    this.fifo.p = v;
    return v;
  }

  /**
   * Tear the session down so a fresh one (new stage) does not leave a second
   * core instance and a second audio context running.
   */
  dispose() {
    this.disposed = true;
    if (this.pendingAudio?.state !== "closed") this.pendingAudio?.close().catch(() => {});
    if (this.audio) {
      try { this.audio.node.disconnect(); } catch { /* already gone */ }
      this.audio.ctx.close().catch(() => {});
      this.audio = null;
    }

    this.onFrame = null;
    this.observer = null;
    this.policy = null;
    this.frameCopy = null;
    this.M = null;
  }

  beginMacro() {
    this.interruptedMacro = false;
    this.action = 0;
    if (!this.inputPaused) {
      const { image, hist } = this.observer.obs();
      this.action = this.policy.argmax(this.policy.forward(image, hist).logits);
    }
    const p1x = this.ram16(P1_X), p2x = this.ram16(P2_X);
    this.opponentRight = p2x > p1x;
    this.masks = expandAction(this.action, this.opponentRight);
    this.wasDead2 = this.p2hp <= 0 || this.p2hp >= 0xff00;
    this.wasDead1 = this.p1hp <= 0 || this.p1hp >= 0xff00;
  }

  /** Advance one physical frame, returning a result when the macro completes. */
  stepFrame() {
    if (this.macroFrame === 0) this.beginMacro();
    const neutral = this.inputPaused || this.interruptedMacro;
    this.joy = neutral ? 0 : this.fifo.next(this.masks[this.macroFrame], this.baseDelay);
    this.M._retro_run();
    this.stats.frames++;
    this.macroFrame = (this.macroFrame + 1) % 4;
    if (this.macroFrame !== 0) return null;

    const action = neutral ? 0 : this.action;
    const rgb = decodeFrame(this.frameCopy, this.cb.w, this.cb.h, this.cb.pitch, 2);
    this.observer.push(rgb, this.cb.w, this.cb.h, action);

    const dead2 = this.p2hp <= 0 || this.p2hp >= 0xff00;
    const dead1 = this.p1hp <= 0 || this.p1hp >= 0xff00;
    if (!this.wasDead2 && dead2) this.stats.p2KO++;
    if (!this.wasDead1 && dead1) this.stats.p1KO++;

    this.stats.decisions++;
    return { action, p1hp: this.p1hp, p2hp: this.p2hp, opponentRight: this.opponentRight };
  }

  /** Complete one policy decision in a blocking loop for headless checks. */
  decideOnce() {
    let result;
    do result = this.stepFrame(); while (!result);
    return result;
  }
}
