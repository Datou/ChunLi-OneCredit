// Run the browser's actual policy loop without rendering or wall-clock delays.
import { mkdtempSync, readFileSync, rmdirSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const web = fileURLToPath(new URL("../web/", import.meta.url));
const { values: args } = parseArgs({ options: {
  rom: { type: "string" },
  fixture: { type: "string", default: join(web, "cores/fixtures/gym-retro/stage08.state.gz") },
  cores: { type: "string", default: join(web, "cores/gym-retro") },
  version: { type: "string", default: "1.7.5" },
  matches: { type: "string", default: "20" },
  seed: { type: "string", default: "81001" },
  jitter: { type: "string", default: "0.05" },
} });
const matches = Number(args.matches);
const seed = Number(args.seed);
const jitter = Number(args.jitter);
if (!args.rom || !Number.isSafeInteger(matches) || matches < 1 || !Number.isSafeInteger(seed)
    || !(jitter >= 0 && jitter <= 1)) {
  throw new Error("Provide --rom, a positive --matches count, an integer --seed and --jitter in [0, 1]");
}
const rom = readFileSync(args.rom);

const cores = resolve(args.cores);
const scratch = mkdtempSync(join(tmpdir(), "chunli-eval-"));
const glue = join(scratch, "core.cjs");
try {
  writeFileSync(glue, readFileSync(join(cores, "genesis_plus_gx.js")));
  globalThis.window = { createGenesisPlusGX: createRequire(import.meta.url)(glue) };
} finally {
  rmSync(glue, { force: true });
  rmdirSync(scratch);
}
globalThis.document = { createElement: () => ({}) };
globalThis.fetch = async url => new Response(readFileSync(String(url).split("?")[0]));
const { LiveSession } = await import(new URL("../web/live.js", import.meta.url));
const fixtureUrl = resolve(args.fixture);

function seedRandom(value) {
  let state = value >>> 0;
  Math.random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

for (let offset = 0; offset < matches; offset++) {
  const matchSeed = seed + offset;
  const session = new LiveSession({
    corePath: cores + "/", modelPath: join(web, "models") + "/", fixtureUrl, stateVersion: args.version,
  });
  try {
    seedRandom(matchSeed);
    await session.loadCore();
    await session.loadRom(rom);
    await session.loadFixture();
    await session.loadPolicy();
    await session.initObserver();
    session.setJitter(jitter);
    while (session.stats.frames < 15_000 && Math.max(session.stats.p1KO, session.stats.p2KO) < 2) {
      session.stepFrame();
    }
    const { frames, p1KO, p2KO } = session.stats;
    const outcome = p2KO >= 2 ? "win" : p1KO >= 2 ? "loss" : "timeout";
    console.log(JSON.stringify({ seed: matchSeed, jitter, outcome, frames, p1KO, p2KO }));
  } finally {
    session.dispose();
  }
}
