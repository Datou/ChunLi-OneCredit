// Pack Gym Retro 0.8.0 Genesis Plus GX x64 states for the same core on wasm32.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";

const HEADER = "GENPLUS-GX 1.7.5";
const SIZE = 1_036_288;
const YM_START = 140_651;
const NATIVE_CART = 144_613;
const NATIVE_Z80 = NATIVE_CART - 88;
const WASM_CART = 144_361;
const CHANNELS = 6;
const SLOTS = 4;
const X64_CHANNEL_BYTES = 400;
const WASM_CHANNEL_BYTES = 360;
const X64_SLOT_BYTES = 80;
const WASM_SLOT_BYTES = 76;
const CART_BYTES = 64;
const ROSTER = ["ChunLi", "Zangief", "Dhalsim", "Ryu", "EHonda", "Blanka", "Guile",
  "Ken", "Balrog", "Vega", "Sagat", "Bison"];

function toWasm(native) {
  const validMap = native.subarray(NATIVE_CART, NATIVE_CART + CART_BYTES).every((value, i) => value === i);
  if (native.length !== SIZE || native.toString("ascii", 0, 16) !== HEADER || !validMap) {
    throw new Error("Expected a Gym Retro 0.8.0 x64 savestate for this ROM");
  }
  const wasm = Buffer.alloc(SIZE);
  native.copy(wasm, 0, 0, YM_START);
  for (let channel = 0; channel < CHANNELS; channel++) {
    const source = YM_START + channel * X64_CHANNEL_BYTES;
    const dest = YM_START + channel * WASM_CHANNEL_BYTES;
    for (let slot = 0; slot < SLOTS; slot++) {
      const operator = source + slot * X64_SLOT_BYTES;
      // DT is rebuilt from the serialized table index by YM2612LoadContext.
      native.copy(wasm, dest + slot * WASM_SLOT_BYTES + 4, operator + 8, operator + X64_SLOT_BYTES);
    }
    native.copy(wasm, dest + 304, source + 320, source + 332);
    // The five output pointers are rebound by setup_connection after loading.
    native.copy(wasm, dest + 336, source + 376, source + X64_CHANNEL_BYTES);
  }
  // DAC/OPN fields, sound counters, 68K and non-pointer Z80 fields have the
  // same representation. The Z80 daisy pointer is null on Genesis and the
  // core resets the interrupt callback after loading.
  native.copy(wasm, YM_START + CHANNELS * WASM_CHANNEL_BYTES,
    YM_START + CHANNELS * X64_CHANNEL_BYTES, NATIVE_Z80 + 68);
  native.copy(wasm, WASM_CART, NATIVE_CART, NATIVE_CART + CART_BYTES);
  return wasm;
}

const [states, out] = process.argv.slice(2);
if (!states || !out) {
  console.error("usage: node scripts/web_pack_retro_states.mjs <controllable-state-dir> <output-dir>");
  process.exit(2);
}
const fixtures = ROSTER.map((opponent, index) => {
  const id = String(index + 1).padStart(2, "0");
  const native = gunzipSync(readFileSync(join(states, `Stage${id}.${opponent}.state`)));
  if (native.readUInt16LE(16 + 0x96b2) !== 0
      || native.readUInt16LE(16 + 0x8042) !== 176 || native.readUInt16LE(16 + 0x82c2) !== 176) {
    throw new Error(`Stage ${id} is not an unlocked, full-health opening`);
  }
  return [`stage${id}.state.gz`, gzipSync(toWasm(native), { level: 9 })];
});
mkdirSync(out, { recursive: true });
for (const [file, bytes] of fixtures) writeFileSync(join(out, file), bytes);
