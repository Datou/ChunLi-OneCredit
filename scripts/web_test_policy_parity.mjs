// Parity test: pure-JS policy forward vs the PyTorch reference.
//
// Reads the exported weights plus reference logits produced by
// scripts/export_policy_weights.py and checks that the JS implementation
// reproduces them.  This is the check that makes in-browser inference credible.
//
// Run:  node scripts/web_test_policy_parity.mjs

import { readFileSync } from "node:fs";
import { Policy } from "../web/policy.js";

const ROOT = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");

const meta = JSON.parse(readFileSync(`${ROOT}web/models/chunppo_p27.json`, "utf8"));
const binBuf = readFileSync(`${ROOT}web/models/chunppo_p27.bin`);
const blob = new Float32Array(binBuf.buffer, binBuf.byteOffset, binBuf.byteLength / 4);
const ref = JSON.parse(readFileSync(`${ROOT}_web_verify/chunppo_p27_ref.json`, "utf8"));

const policy = new Policy(blob, meta);

let worstLogit = 0;
let worstValue = 0;
let agree = 0;

for (const [i, c] of ref.cases.entries()) {
  const image = Float32Array.from(c.image);
  const hist = Float32Array.from(c.hist);
  const out = policy.forward(image, hist);

  let maxLogit = 0;
  for (let k = 0; k < c.logits.length; k++) {
    maxLogit = Math.max(maxLogit, Math.abs(out.logits[k] - c.logits[k]));
  }
  const valueDiff = Math.abs(out.value - c.value);
  worstLogit = Math.max(worstLogit, maxLogit);
  worstValue = Math.max(worstValue, valueDiff);

  let refBest = 0;
  for (let k = 1; k < c.logits.length; k++) if (c.logits[k] > c.logits[refBest]) refBest = k;
  const jsBest = policy.argmax(out.logits);
  if (jsBest === refBest) agree++;

  console.log(
    `case ${i}: maxLogitDiff=${maxLogit.toExponential(2)} valueDiff=${valueDiff.toExponential(2)} ` +
      `argmax ref=${refBest} js=${jsBest} ${jsBest === refBest ? "ok" : "MISMATCH"}`
  );
}

console.log("");
console.log(`max |logit diff| = ${worstLogit.toExponential(3)}`);
console.log(`max |value diff| = ${worstValue.toExponential(3)}`);
console.log(`argmax agreement = ${agree}/${ref.cases.length}`);

const ok = worstLogit < 2e-3 && worstValue < 2e-3 && agree === ref.cases.length;
console.log(ok ? "PARITY PASS" : "PARITY FAIL");
process.exit(ok ? 0 : 1);
