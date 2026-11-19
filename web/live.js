// Live emulator + policy session for the web page.
//
// Everything here was validated in scripts/rt_play.html: the locally built
// Genesis Plus GX core (which exports retro_get_memory_data), the pure-JS policy
// forward, the 84x84 Observer, and the 27 four-frame action macros.
//
// A session is: boot core -> load ROM from bytes -> unserialize a fixture ->
// then once per 4 emulator frames: observe, decide, act.

import { Policy } from "./policy.js?v=15";
import { Observer, decodeFrame } from "./observer.js?v=9";
import { expandAction } from "./actions.js?v=10";

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
  next(request, base = 0, frozen = false) {
    if (!frozen) {
      const delayed = this.p > 0 && Math.random() < this.p;
      const d = delayed ? 1 + Math.floor(Math.random() * 5) : 0;
      this.release = Math.max(this.tick + base + d, this.release);
      this.lastDelay = this.release - this.tick;
      this.pending.push([this.release, request]);
    }
    while (this.pending.length && this.pending[0][0] <= this.tick) {
      this.current = this.pending.shift()[1];
    }
    this.tick++;
    return this.current;
  }
}

// Bump this whenever a ladder fixture is regenerated.
export const FIXTURE_VERSION = 3;

export class LiveSession {
  constructor(opts = {}) {
    this.corePath = opts.corePath ?? "cores/";
    this.modelPath = opts.modelPath ?? "models/";
    this.fixtureUrl = opts.fixtureUrl ?? `cores/fixtures/ladder/stage01.state.gz?v=${FIXTURE_VERSION}`;
    this.onFrame = opts.onFrame ?? null;

    this.M = null;
    this.cb = null;
    this.frameCopy = null;
    this.ramPtr = 0;
    this.joy = 0;
    this.frozen = false;
    this.frozenMask = 0;
    this.fifo = new InputFifo(0.05);
    this.baseDelay = 0;             // fixed capture->execution delay, frames
    this.spaceDelay = 6;            // extra delay while Space is held
    this.spaceHeld = false;
    this.policy = null;
    this.observer = null;
    this.macroFrame = 0;
    this.pixelCanvas = document.createElement("canvas");
    this.stats = { decisions: 0, frames: 0, p1KO: 0, p2KO: 0, frozen: 0 };
  }

  async loadCore() {
    if (!window.createGenesisPlusGX) {
      await new Promise((res, rej) => {
        const s = document.createElement("script");
        s.src = `${this.corePath}genesis_plus_gx.js`;
        s.onload = res;
        s.onerror = () => rej(new Error("core script failed to load"));
        document.head.appendChild(s);
      });
    }
    const M = await window.createGenesisPlusGX({ print: () => {}, printErr: () => {} });
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
    if (M._retro_load_game(gi) !== 1) throw new Error("retro_load_game failed (unsupported or corrupt ROM)");
    M._retro_reset();
    this.ramPtr = M._retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM);
    return this.ramPtr;
  }

  async loadFixture(url = this.fixtureUrl, file = null) {
    const M = this.M;
    const source = file ?? await fetch(url);
    if (!file && !source.ok) {
      throw Object.assign(new Error(`存档载入失败（HTTP ${source.status}）`), { status: source.status });
    }
    let st = new Uint8Array(await source.arrayBuffer());

    // Ladder fixtures ship gzipped (1 MB of state compresses to ~40 KB).
    if (st[0] === 0x1f && st[1] === 0x8b) {
      if (typeof DecompressionStream !== "function") {
        throw new Error("当前浏览器不支持 gzip 解压，无法载入压缩存档");
      }
      const stream = new Blob([st]).stream().pipeThrough(new DecompressionStream("gzip"));
      st = new Uint8Array(await new Response(stream).arrayBuffer());
    }

    if (st.length !== M._retro_serialize_size()
        || new TextDecoder().decode(st.subarray(0, 16)) !== "GENPLUS-GX 1.7.6") {
      throw new Error("请选择网页专用的 Genesis Plus GX 1.7.6 存档");
    }
    const sp = M._malloc(st.length);
    M.HEAPU8.set(st, sp);
    const ok = M._retro_unserialize(sp, st.length) === 1;
    if (!ok) throw new Error(`存档 ${url} 无法载入（格式或版本不匹配）`);
    M._retro_run();
    // Deliberately not freed: the core's fast-savestate path can keep reading the
    // buffer, and one 1 MB allocation per session is not worth the risk.
    return ok;
  }

  async loadPolicy() {
    this.policy = await Policy.load(this.modelPath);
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
    if (!Audio) return;
    const av = this.M._malloc(40);
    this.M._retro_get_system_av_info(av);
    const sampleRate = new DataView(this.M.HEAPU8.buffer).getFloat64(av + 32, true);
    this.M._free(av);
    const ctx = new Audio({ sampleRate, latencyHint: "interactive" });
    try {
      await ctx.audioWorklet.addModule(new URL("./audio-worklet.js?v=15", import.meta.url));
      const node = new AudioWorkletNode(ctx, "genesis-audio", { outputChannelCount: [2] });
      this.audioStats = { framesIn: 0, framesOut: 0, underruns: 0 };
      node.port.onmessage = ({ data }) => Object.assign(this.audioStats, data);
      node.connect(ctx.destination);
      this.audio = { ctx, node };
    } catch (error) {
      await ctx.close();
      throw error;
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
    const y0 = (h - cropH) >> 1;
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

  setFrozen(on) {
    this.frozen = !!on;
    return this.frozen;
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

  /** Space held: adds spaceDelay frames on top of the base delay. */
  setSpaceHeld(on) {
    this.spaceHeld = !!on;
    return this.spaceHeld;
  }

  get effectiveDelay() {
    return this.baseDelay + (this.spaceHeld ? this.spaceDelay : 0);
  }

  /**
   * Tear the session down so a fresh one (new stage) does not leave a second
   * core instance and a second audio context running.
   */
  dispose() {
    if (this.audio) {
      try { this.audio.node.disconnect(); } catch { /* already gone */ }
      try { this.audio.ctx.close(); } catch { /* already closed */ }
      this.audio = null;
    }

    this.onFrame = null;
    this.observer = null;
    this.policy = null;
    this.frameCopy = null;
    this.M = null;
  }

  beginMacro() {
    const o = this.observer.obs();
    const { logits } = this.policy.forward(o.image, o.hist);
    this.action = this.policy.argmax(logits);
    const p1x = this.ram16(P1_X), p2x = this.ram16(P2_X);
    this.opponentRight = p2x > p1x;
    this.masks = expandAction(this.action, this.opponentRight);
    this.wasDead2 = this.p2hp <= 0 || this.p2hp >= 0xff00;
    this.wasDead1 = this.p1hp <= 0 || this.p1hp >= 0xff00;
  }

  /** Advance one physical frame, returning a result when the macro completes. */
  stepFrame() {
    if (this.macroFrame === 0) this.beginMacro();
    this.joy = this.fifo.next(this.masks[this.macroFrame], this.effectiveDelay, this.frozen);
    this.M._retro_run();
    this.stats.frames++;
    this.macroFrame = (this.macroFrame + 1) % 4;
    if (this.macroFrame !== 0) return null;
    if (this.frozen) this.stats.frozen++;

    const rgb = decodeFrame(this.frameCopy, this.cb.w, this.cb.h, this.cb.pitch, 2);
    this.observer.push(rgb, this.cb.w, this.cb.h, this.action);

    const dead2 = this.p2hp <= 0 || this.p2hp >= 0xff00;
    const dead1 = this.p1hp <= 0 || this.p1hp >= 0xff00;
    if (!this.wasDead2 && dead2) this.stats.p2KO++;
    if (!this.wasDead1 && dead1) this.stats.p1KO++;

    this.stats.decisions++;
    return { action: this.action, p1hp: this.p1hp, p2hp: this.p2hp, opponentRight: this.opponentRight };
  }

  /** Complete one policy decision in a blocking loop for headless checks. */
  decideOnce() {
    let result;
    do result = this.stepFrame(); while (!result);
    return result;
  }
}
