// Convert a gym-retro era savestate (GENPLUS-GX 1.7.5) into one the current
// 1.7.6 core can actually *play*.
//
// The two formats share the same layout through the VDP register array.
// Later fields differ, so loading an unconverted save misaligns the CPU state
// and the machine hangs on a frozen picture.
//
// Keep the legacy RAM, video memory and VDP registers together. The registers
// describe how to render that memory, including the China stage's per-line
// horizontal scrolling. Take the remaining runtime state from a known-good
// 1.7.6 save captured at an equivalent moment.
//
//   node scripts/web_convert_state.mjs --in=legacy.state --out=converted.state \
//        [--template=web/cores/fixtures/stage01_roundstart.state] [--splice=140613]

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const VDP_REGISTER_OFFSET = 140581;
const VDP_REGISTER_BYTES = 0x20;
const SPLICE_DEFAULT = VDP_REGISTER_OFFSET + VDP_REGISTER_BYTES;
const STATE_SIZE = 1036288;

const arg = (name, dflt) => {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : dflt;
};

const inPath = arg("in");
const outPath = arg("out");
const templatePath = arg("template", join(here, "..", "web", "cores", "fixtures", "stage01_roundstart.state"));
const splice = Number(arg("splice", SPLICE_DEFAULT));
const wantGzip = process.argv.includes("--gzip");

if (!inPath || !outPath) {
  console.error("usage: node scripts/web_convert_state.mjs --in=<legacy> --out=<new> [--template=...] [--splice=N]");
  process.exit(2);
}

const readState = (p) => {
  const raw = readFileSync(p);
  return raw[0] === 0x1f && raw[1] === 0x8b ? gunzipSync(raw) : raw;
};

const legacy = readState(inPath);
const template = readState(templatePath);

for (const [name, buf] of [["legacy", legacy], ["template", template]]) {
  if (buf.length !== STATE_SIZE) throw new Error(`${name} payload is ${buf.length}, expected ${STATE_SIZE}`);
}

const header = (buf) => buf.subarray(0, 16).toString("latin1");
const out = Buffer.from(legacy);          // world + header from the old save
out.write("GENPLUS-GX 1.7.6", 0, "latin1");   // last byte stays NUL-free, 16 bytes total
Buffer.from(template).copy(out, splice, splice);      // control block from the template

mkdirSync(dirname(resolve(outPath)), { recursive: true });
writeFileSync(outPath, wantGzip ? gzipSync(out, { level: 9 }) : out);

console.log(JSON.stringify({
  in: inPath, out: outPath, template: templatePath, splice,
  legacyHeader: header(legacy), templateHeader: header(template),
  outHeader: header(out), bytes: out.length, gzip: wantGzip,
}, null, 2));
