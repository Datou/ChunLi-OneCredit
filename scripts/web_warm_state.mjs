// Bake the round-intro out of a fixture.
//
// The training states are captured at the "ROUND 1 / BATTLE 01" banner, where the
// game holds input locked and opens the stage with a wipe.  Loading one and
// starting the policy immediately means the first ~2 s of the page is that
// transition, which reads like a broken background.  Run the emulator forward
// with no input until the lock clears, then re-serialize.
//
//   node scripts/web_warm_state.mjs --in=fx.state.gz --out=fx_warm.state.gz \
//        [--rom=...] [--frames=200] [--gzip]

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const coreDir = resolve(join(here, "..", "web", "cores"));

const arg = (name, dflt) => {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : dflt;
};

const inPath = arg("in");
const outPath = arg("out");
const romPath = arg("rom", "retro_data/StreetFighterIISpecialChampionEdition-Genesis/rom.md");
const maxFrames = Number(arg("frames", 260));
const wantGzip = process.argv.includes("--gzip");

const RETRO_MEMORY_SYSTEM_RAM = 2;
const P1_X = 0x8006, P2_X = 0x8286, P1_HP = 0x8042, P2_HP = 0x82C2, LOCK = 0x96B2;
const STATE_SIZE = 1036288;

const readState = (p) => {
  const raw = readFileSync(p);
  return raw[0] === 0x1f && raw[1] === 0x8b ? gunzipSync(raw) : raw;
};

const { createRequire } = await import("node:module");
const { tmpdir } = await import("node:os");
const cjsPath = join(tmpdir(), `gx_core_warm_${process.pid}.cjs`);
writeFileSync(cjsPath, readFileSync(join(coreDir, "genesis_plus_gx.js"), "utf8"));
const factory = createRequire(import.meta.url)(cjsPath);
const M = await factory({ locateFile: (p) => join(coreDir, p), print: () => {}, printErr: () => {} });

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
M._retro_set_environment(envCb);
M._retro_set_video_refresh(M.addFunction(() => {}, "viiii"));
M._retro_set_audio_sample_batch(M.addFunction((d, f) => f, "iii"));
M._retro_set_input_poll(M.addFunction(() => {}, "v"));
M._retro_set_input_state(M.addFunction(() => 0, "iiiii"));

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

if (M._retro_serialize_size() !== STATE_SIZE) {
  console.error(`unexpected state size ${M._retro_serialize_size()}`);
  process.exit(1);
}

const st = readState(inPath);
if (st.length !== STATE_SIZE) { console.error(`input payload ${st.length}`); process.exit(1); }
const sp = M._malloc(st.length);
M.HEAPU8.set(st, sp);
if (M._retro_unserialize(sp, st.length) !== 1) { console.error("unserialize failed"); process.exit(1); }

const ramPtr = M._retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM);
const rd16 = (o) => M.HEAPU8[ramPtr + o] | (M.HEAPU8[ramPtr + o + 1] << 8);
const lockOf = () => rd16(LOCK);

const before = { lock: lockOf(), p1hp: rd16(P1_HP), p2hp: rd16(P2_HP) };

let frames = 0;
while (lockOf() !== 0 && frames < maxFrames) {
  M._retro_run();
  frames++;
}
const after = { lock: lockOf(), p1hp: rd16(P1_HP), p2hp: rd16(P2_HP), p1x: rd16(P1_X), p2x: rd16(P2_X) };
if (after.lock !== 0 || after.p1hp !== 176 || after.p2hp !== 176) {
  throw new Error(`Expected an unlocked full-health opening: ${JSON.stringify(after)}`);
}

const buf = M._malloc(STATE_SIZE);
if (M._retro_serialize(buf, STATE_SIZE) !== 1) { console.error("serialize failed"); process.exit(1); }
const out = Buffer.from(M.HEAPU8.subarray(buf, buf + STATE_SIZE));
if (out.subarray(0, 16).toString("latin1") !== "GENPLUS-GX 1.7.6") {
  console.error(`bad header ${out.subarray(0, 16).toString("latin1")}`);
  process.exit(1);
}
mkdirSync(dirname(resolve(outPath)), { recursive: true });
writeFileSync(outPath, wantGzip ? gzipSync(out, { level: 9 }) : out);

console.log(JSON.stringify({ in: inPath, out: outPath, frames, before, after, gzip: wantGzip }));
