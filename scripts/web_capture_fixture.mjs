// Headless fixture capture: boot -> menus -> Chun-Li -> round start -> savestate.
//
// Reuses the macro that produced web/cores/fixtures/stage01_roundstart.state, but
// runs it on the Node core harness so timings can be swept without a browser.
//
// Run:
//   node scripts/web_capture_fixture.mjs --rom=... [--boot=600] [--hold=6]
//        [--settle=90] [--out=...] [--ppm=...] [--quiet]

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const coreDir = resolve(join(here, "..", "web", "cores"));

const arg = (name, dflt) => {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : dflt;
};
const quiet = process.argv.includes("--quiet");

const romPath = arg("rom");
const boot = Number(arg("boot", 600));
const hold = Number(arg("hold", 6));
const settle = Number(arg("settle", 90));
const cursor = arg("cursor", "down,right");     // player-select moves after the START taps
const outPath = arg("out");
const ppmPath = arg("ppm");

const RETRO_MEMORY_SYSTEM_RAM = 2;
const P1_X = 0x8006, P2_X = 0x8286, P1_HP = 0x8042, P2_HP = 0x82C2, LOCK = 0x96B2;
const B = { START: 3, DOWN: 5, UP: 4, LEFT: 6, RIGHT: 7, A: 8 };

// node >= 25 refuses to eval the Emscripten glue directly (it sees __dirname plus
// await and calls the module format ambiguous), so load it from a .cjs copy.
const { createRequire } = await import("node:module");
const { tmpdir } = await import("node:os");
const coreSrc = readFileSync(join(coreDir, "genesis_plus_gx.js"), "utf8");
const cjsPath = join(tmpdir(), `gx_core_cap_${process.pid}.cjs`);
writeFileSync(cjsPath, coreSrc);
const factory = createRequire(import.meta.url)(cjsPath);

const M = await factory({ locateFile: (p) => join(coreDir, p), print: () => {}, printErr: () => {} });

let frame = null;
const envCb = M.addFunction((cmd, data) => {
  switch (cmd) {
    case 9: case 30: {
      const p = M._malloc(64); M.stringToUTF8("/tmp", p, 64);
      M.HEAP32[data >> 2] = p; return 1;
    }
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
  frame = { w, h, pitch, data: M.HEAPU8.slice(data, data + pitch * h) };
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
const rd16 = (o) => M.HEAPU8[ramPtr + o] | (M.HEAPU8[ramPtr + o + 1] << 8);
const stats = () => ({
  p1x: rd16(P1_X), p2x: rd16(P2_X), p1hp: rd16(P1_HP), p2hp: rd16(P2_HP), lock: rd16(LOCK),
});

const run = (n, pad = 0) => { joy = pad; for (let i = 0; i < n; i++) M._retro_run(); };
const tap = (pad, h = hold, s = settle) => { run(h, pad); run(s, 0); };

run(boot);
for (let i = 0; i < 5; i++) tap(1 << B.START);
for (const mv of cursor.split(",").filter(Boolean)) {
  tap(1 << B[mv.trim().toUpperCase()]);
}
const cursorStats = stats();
tap(1 << B.A);

let found = -1, frames = 0;
for (; frames < 2000; frames++) {
  run(1, 0);
  const s = stats();
  if (s.p1hp === 176 && s.p2hp === 176 && s.lock === 0) { found = frames; break; }
}
const final = stats();

const hash = (() => {
  let h = 2166136261;
  for (let i = 0; i < 65536; i++) h = ((h ^ M.HEAPU8[ramPtr + i]) * 16777619) >>> 0;
  return h.toString(16).padStart(8, "0");
})();

const size = M._retro_serialize_size();
let saved = null;
if (outPath && found >= 0) {
  if (size !== 1036288) throw new Error(`unexpected serialize size ${size}`);
  const buf = M._malloc(size);
  if (M._retro_serialize(buf, size) !== 1) throw new Error("retro_serialize failed");
  const bytes = Buffer.from(M.HEAPU8.subarray(buf, buf + size));
  mkdirSync(dirname(resolve(outPath)), { recursive: true });
  writeFileSync(outPath, bytes);
  saved = { path: outPath, bytes: bytes.length, header: bytes.subarray(0, 16).toString("latin1") };
}

if (ppmPath && frame) {
  // The instant the intro lock clears the game is mid-transition and renders a
  // black frame, so advance a little before grabbing the verification picture.
  run(30, 0);
  const f = frame;
  const rgb = Buffer.alloc(f.w * f.h * 3);
  for (let y = 0; y < f.h; y++) {
    for (let x = 0; x < f.w; x++) {
      const o = y * f.pitch + x * 2;
      const v = f.data[o] | (f.data[o + 1] << 8);
      const d = (y * f.w + x) * 3;
      rgb[d] = ((v >> 11) & 0x1f) << 3;
      rgb[d + 1] = ((v >> 5) & 0x3f) << 2;
      rgb[d + 2] = (v & 0x1f) << 3;
    }
  }
  mkdirSync(dirname(resolve(ppmPath)), { recursive: true });
  writeFileSync(ppmPath, Buffer.concat([Buffer.from(`P6\n${f.w} ${f.h}\n255\n`), rgb]));
}

const out = {
  boot, hold, settle, cursor,
  cursorStats, foundRoundStart: found >= 0, framesAfterConfirm: frames,
  stats: final, ramHash: hash, frameSize: frame ? `${frame.w}x${frame.h}` : null, saved,
};
if (!quiet) console.log(JSON.stringify(out, null, 2));
else console.log(`${JSON.stringify({ boot, found: found >= 0, stats: final, ramHash: hash })}`);
