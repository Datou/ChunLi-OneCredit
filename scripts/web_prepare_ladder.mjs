// Convert all twelve locally captured training openings to browser fixtures.
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmdirSync, rmSync, copyFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const arg = (name, fallback) => {
  const prefix = `--${name}=`;
  return process.argv.find(value => value.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const rom = resolve(arg("rom", "retro_data/StreetFighterIISpecialChampionEdition-Genesis/rom.md"));
const states = resolve(arg("states", "generated_states/controllable"));
const out = resolve(arg("out", "web/cores/fixtures/ladder"));
const roster = ["ChunLi", "Zangief", "Dhalsim", "Ryu", "EHonda", "Blanka", "Guile", "Ken",
  "Balrog", "Vega", "Sagat", "Bison"];
const run = (script, ...args) => execFileSync(process.execPath, [join(root, "scripts", script), ...args], {
  stdio: "inherit",
});

const scratch = mkdtempSync(join(tmpdir(), "chunli-web-ladder-"));
try {
  const template = join(scratch, "template.state");
  run("web_capture_fixture.mjs", `--rom=${rom}`, `--out=${template}`, "--quiet");
  const filenames = roster.map((opponent, i) => {
    const id = String(i + 1).padStart(2, "0");
    const filename = `stage${id}.state.gz`;
    const fixture = join(scratch, filename);
    run("web_convert_state.mjs", `--in=${join(states, `Stage${id}.${opponent}.state`)}`,
      `--template=${template}`, `--out=${fixture}`, "--gzip");
    run("web_warm_state.mjs", `--rom=${rom}`, `--in=${fixture}`, `--out=${fixture}`, "--gzip");
    return filename;
  });
  mkdirSync(out, { recursive: true });
  for (const name of filenames) copyFileSync(join(scratch, name), join(out, name));
} finally {
  // The temporary directory contains only the known files created by this script.
  for (const name of ["template.state", ...roster.map((_, i) => `stage${String(i + 1).padStart(2, "0")}.state.gz`)]) {
    rmSync(join(scratch, name), { force: true });
  }
  rmdirSync(scratch);
}
