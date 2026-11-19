// Headless campaign runner: emulator + chunppo policy in Node, no browser.
//
// Same decision loop as web/live.js (observe -> argmax -> 27x4-frame macro),
// so results here match what the live page would do, but at headless speed and
// with the input FIFO's randomness controllable by --seed.
//
//   node scripts/web_run_campaign.mjs --rom=... --fixture=... [--decisions=3000]
//        [--jitter=0.05] [--delay=0] [--seed=1] [--quiet]

import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(join(here, "..", "web"));
const coreDir = join(webDir, "cores");

const arg = (name, dflt) => {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : dflt;
};
const quiet = process.argv.includes("--quiet");

const romPath = arg("rom");
const fixturePath = arg("fixture", join(coreDir, "fixtures", "stage01_roundstart.state"));
const decisions = Number(arg("decisions", 3000));
const jitter = Number(arg("jitter", 0));
const delay = Number(arg("delay", 0));
const seed = arg("seed", null);

// Deterministic Math.random for the FIFO when a seed is given.
if (seed !== null) {
  let s = (Number(seed) >>> 0) || 1;
  Math.random = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// policy.js fetches its weights; serve them from disk in Node.
const nativeFetch = globalThis.fetch;
globalThis.fetch = async (url, opts) => {
  if (typeof url === "string" && !/^[a-z][a-z0-9+.-]*:\/\//i.test(url)) {
    return new Response(readFileSync(resolve(url)), { status: 200 });
  }
  return nativeFetch(url, opts);
};

const { Policy } = await import(new URL("../web/policy.js", import.meta.url).href);
const { Observer, decodeFrame } = await import(new URL("../web/observer.js", import.meta.url).href);
const { expandAction } = await import(new URL("../web/actions.js", import.meta.url).href);

// Same ordered delay FIFO as web/live.js InputFifo, duplicated here because
// live.js is only importable in a browser.
class Fifo {
  constructor(p) { this.p = p; this.pending = []; this.current = 0; this.tick = 0; this.release = 0; }
  next(request, base = 0) {
    const delayed = this.p > 0 && Math.random() < this.p;
    const d = delayed ? 1 + Math.floor(Math.random() * 5) : 0;
    this.release = Math.max(this.tick + base + d, this.release);
    this.pending.push([this.release, request]);
    while (this.pending.length && this.pending[0][0] <= this.tick) this.current = this.pending.shift()[1];
    this.tick++;
    return this.current;
  }
}

const RETRO_MEMORY_SYSTEM_RAM = 2;
const P1_X = 0x8006, P2_X = 0x8286, P1_HP = 0x8042, P2_HP = 0x82C2, LOCK = 0x96B2;

const { createRequire } = await import("node:module");
const { tmpdir } = await import("node:os");
const cjsPath = join(tmpdir(), `gx_core_camp_${process.pid}.cjs`);
writeFileSync(cjsPath, readFileSync(join(coreDir, "genesis_plus_gx.js"), "utf8"));
const factory = createRequire(import.meta.url)(cjsPath);
const M = await factory({ locateFile: (p) => join(coreDir, p), print: () => {}, printErr: () => {} });

let frameCopy = null, cb = null;
const envCb = M.addFunction((cmd, data) => {
  switch (cmd) {
    case 9: case 30: { const p = M._malloc(64); M.stringToUTF8("/tmp", p, 64); M.HEAP32[data >> 2] = p; return 1; }
    case 10: return 1;
    case 1: case 11: case 16: case 18: case 34: return 1;
    case 2: M.HEAP32[data >> 2] = 1; return 1;
    case 3: M.HEAP32[data >> 2] = 0; return 1;
    case 36: return 1;
    default: return 0;
  }
}, "iii");
const videoCb = M.addFunction((data, w, h, pitch) => {
  if (!data) return;
  const need = pitch * h;
  if (!frameCopy || frameCopy.length < need) frameCopy = new Uint8Array(need);
  frameCopy.set(M.HEAPU8.subarray(data, data + need));
  cb = { w, h, pitch };
}, "viiii");
let joy = 0;
M._retro_set_environment(envCb);
M._retro_set_video_refresh(videoCb);
M._retro_set_audio_sample_batch(M.addFunction((d, f) => f, "iii"));
M._retro_set_input_poll(M.addFunction(() => {}, "v"));
M._retro_set_input_state(M.addFunction((p, dev, idx, id) => (p === 0 && ((joy >> id) & 1) ? 1 : 0), "iiiii"));

const si = M._malloc(32);
M._retro_get_system_info(si);
M._retro_init();
M.FS.writeFile("/tmp/game.md", new Uint8Array(readFileSync(romPath)));
const pp = M._malloc(64);
M.stringToUTF8("/tmp/game.md", pp, 64);
const gi = M._malloc(16);
M.HEAP32[(gi >> 2) + 0] = pp;
if (M._retro_load_game(gi) !== 1) { console.error("load_game failed"); process.exit(1); }
M._retro_reset();
const ramPtr = M._retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM);
const ram16 = (o) => M.HEAPU8[ramPtr + o] | (M.HEAPU8[ramPtr + o + 1] << 8);

const rawState = readFileSync(fixturePath);
const st = rawState[0] === 0x1f && rawState[1] === 0x8b ? gunzipSync(rawState) : rawState;
const sp = M._malloc(st.length);
M.HEAPU8.set(st, sp);
if (M._retro_unserialize(sp, st.length) !== 1) { console.error("unserialize failed"); process.exit(1); }
M._retro_run();

const policy = await Policy.load(join(webDir, "models") + "/");
const observer = new Observer();
observer.reset(decodeFrame(frameCopy, cb.w, cb.h, cb.pitch, 2), cb.w, cb.h);

const fifo = new Fifo(jitter);
let p2KO = 0, p1KO = 0, frames = 0;
const events = [];
const t0 = Date.now();

for (let d = 0; d < decisions; d++) {
  const o = observer.obs();
  const { logits } = policy.forward(o.image, o.hist);
  let action = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[action]) action = i;

  const opponentRight = ram16(P2_X) > ram16(P1_X);
  const masks = expandAction(action, opponentRight);
  const wasDead2 = ram16(P2_HP) <= 0 || ram16(P2_HP) >= 0xff00;
  const wasDead1 = ram16(P1_HP) <= 0 || ram16(P1_HP) >= 0xff00;

  for (let f = 0; f < 4; f++) {
    joy = fifo.next(masks[f], delay);
    M._retro_run();
    frames++;
  }

  observer.push(decodeFrame(frameCopy, cb.w, cb.h, cb.pitch, 2), cb.w, cb.h, action);

  const dead2 = ram16(P2_HP) <= 0 || ram16(P2_HP) >= 0xff00;
  const dead1 = ram16(P1_HP) <= 0 || ram16(P1_HP) >= 0xff00;
  if (!wasDead2 && dead2) { p2KO++; events.push({ d, ev: "p2KO" }); }
  if (!wasDead1 && dead1) { p1KO++; events.push({ d, ev: "p1KO" }); }

  if (!quiet && d % 500 === 0) {
    console.log(`  d=${d} frames=${frames} hp=${ram16(P1_HP)}/${ram16(P2_HP)} x=${ram16(P1_X)}/${ram16(P2_X)} ko=${p2KO}/${p1KO}`);
  }
}

const summary = {
  decisions, frames, jitter, delay, seed: seed === null ? null : Number(seed),
  p2KO, p1KO, events, ms: Date.now() - t0,
  finalHp: [ram16(P1_HP), ram16(P2_HP)],
};

const ppmPath = arg("ppm");
if (ppmPath && frameCopy && cb) {
  const rgb = Buffer.alloc(cb.w * cb.h * 3);
  for (let y = 0; y < cb.h; y++) {
    for (let x = 0; x < cb.w; x++) {
      const o = y * cb.pitch + x * 2;
      const v = frameCopy[o] | (frameCopy[o + 1] << 8);
      const d = (y * cb.w + x) * 3;
      rgb[d] = ((v >> 11) & 0x1f) << 3;
      rgb[d + 1] = ((v >> 5) & 0x3f) << 2;
      rgb[d + 2] = (v & 0x1f) << 3;
    }
  }
  writeFileSync(ppmPath, Buffer.concat([Buffer.from(`P6\n${cb.w} ${cb.h}\n255\n`), rgb]));
  summary.ppm = ppmPath;
}

console.log(JSON.stringify(summary));
