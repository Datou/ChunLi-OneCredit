// The chunppo 27-action macro table, ported from chunppo/common.py.
//
// Each action is 4 physical frames; each frame is a set of logical keys.
// FWD/BACK are facing-relative and resolved from RAM x at macro start.
// Attack buttons are pressed for 3 frames and released on the 4th so that
// choosing the same attack twice in a row produces two presses.

// Genesis 12-button order used by gym-retro / libretro.
export const BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"];
export const IDX = Object.fromEntries(BUTTONS.map((b, i) => [b, i]));

const PUNCH = ["X", "Y", "Z"]; // light, medium, heavy punch
const KICK = ["A", "B", "C"];  // light, medium, heavy kick

const hold = (...keys) => [keys, keys, keys, keys];
const press = (btn, held = []) => [[btn], [btn], [btn], held];
const pressDown = (btn) => [[ "DOWN", btn], ["DOWN", btn], ["DOWN", btn], ["DOWN"]];
const seq = (dir, btn) => [[dir], [dir, btn], [dir, btn], [dir]];

function buildActions() {
  const acts = [];
  const names = [];
  const add = (name, frames) => { names.push(name); acts.push(frames); };

  add("neutral", hold());
  for (const [name, keys] of [
    ["back", ["BACK"]], ["fwd", ["FWD"]], ["up", ["UP"]], ["down", ["DOWN"]],
    ["down_back", ["DOWN", "BACK"]], ["down_fwd", ["DOWN", "FWD"]],
    ["up_back", ["UP", "BACK"]], ["up_fwd", ["UP", "FWD"]],
  ]) add(name, hold(...keys));

  for (const b of PUNCH.concat(KICK)) add(`st_${b}`, press(b));
  for (const b of PUNCH.concat(KICK)) add(`cr_${b}`, pressDown(b));

  add("fwd_Y", seq("FWD", "Y"));
  add("fwd_Z", seq("FWD", "Z"));
  add("fwd_C", seq("FWD", "C"));
  add("up_B", seq("UP", "B"));
  add("up_C", seq("UP", "C"));
  add("mash_B", [["B"], [], ["B"], []]);

  return { acts, names };
}

const { acts: ACTIONS, names: ACTION_NAMES } = buildActions();
export { ACTIONS, ACTION_NAMES };
export const N_ACTIONS = ACTIONS.length;
const ATTACK_LABELS = {
  zh: { X: "轻拳", Y: "中拳", Z: "重拳", A: "轻腿", B: "中腿", C: "重腿" },
  en: { X: "LP", Y: "MP", Z: "HP", A: "LK", B: "MK", C: "HK" },
};
const MOVE_LABELS = {
  zh: {
    neutral: "不动", back: "后", fwd: "前", up: "跳", down: "蹲",
    down_back: "蹲防", down_fwd: "蹲＋前", up_back: "后跳", up_fwd: "前跳",
    fwd_Y: "前＋中拳", fwd_Z: "前＋重拳（近身投）", fwd_C: "前＋重腿（近身投）",
    up_B: "跳＋中腿", up_C: "跳＋重腿", mash_B: "连打中腿",
  },
  en: {
    neutral: "Neutral", back: "Back", fwd: "Forward", up: "Jump", down: "Crouch",
    down_back: "Crouch block", down_fwd: "Crouch + forward", up_back: "Back jump", up_fwd: "Forward jump",
    fwd_Y: "Forward + MP", fwd_Z: "Forward + HP (throw)", fwd_C: "Forward + HK (throw)",
    up_B: "Jump + MK", up_C: "Jump + HK", mash_B: "Mash MK",
  },
};

function buildLabels(lang) {
  const attacks = ATTACK_LABELS[lang];
  const moves = MOVE_LABELS[lang];
  return ACTION_NAMES.map(name => {
    const [stance, button] = name.split("_");
    if (stance === "st") return lang === "zh" ? attacks[button] : `Stand ${attacks[button]}`;
    if (stance === "cr") return lang === "zh" ? `蹲＋${attacks[button]}` : `Crouch + ${attacks[button]}`;
    return moves[name];
  });
}

export const ACTION_LABELS = buildLabels("zh");
const ACTION_LABELS_EN = buildLabels("en");

/** Action names for the battle panel, per interface language. */
export function actionLabels(lang = "zh") {
  return lang === "zh" ? ACTION_LABELS : ACTION_LABELS_EN;
}

/** Resolve one macro frame's logical keys into a 12-bit button mask. */
export function resolveFrame(frameKeys, opponentRight) {
  let mask = 0;
  for (let k of frameKeys) {
    if (k === "FWD") k = opponentRight ? "RIGHT" : "LEFT";
    else if (k === "BACK") k = opponentRight ? "LEFT" : "RIGHT";
    mask |= 1 << IDX[k];
  }
  return mask;
}

/**
 * Expand an action index into 4 button masks for a given facing.
 * @returns {number[]} length-4 array of 12-bit masks
 */
export function expandAction(action, opponentRight) {
  const frames = ACTIONS[action];
  if (!frames) throw new Error(`bad action ${action}`);
  const out = new Array(4);
  for (let i = 0; i < 4; i++) out[i] = resolveFrame(frames[i], opponentRight);
  return out;
}
