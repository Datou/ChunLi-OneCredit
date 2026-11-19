// SF2 one-life run, live in the browser.
// Static page: no server, no build step. The Genesis core (WASM) runs the ROM
// in the page, the chunppo policy picks an action every 4 frames, and every
// frame is pushed through a WebGL2 shader with a configurable
// scanline effect.

import { VERT, FRAG, UNIFORMS } from "./shader.js?v=15";
import { FIXTURE_VERSION, LiveSession, prepareCore, prepareFixture, preparePolicy } from "./live.js?v=24";
import { actionLabels } from "./actions.js?v=24";
import { withDeadline } from "./assets.js?v=24";
import { applyI18n, getLang, onLangChange, setLang, t } from "./i18n.js?v=24";

const NATIVE = { w: 256, h: 200 }; // game pixels after the standard crop
const FRAME_MS = 1000 / 59.92274340431231;
const MAX_CATCHUP_FRAMES = 3;

const RENDERS = [
  { label: "256×200 (1×)", r: 1 },
  { label: "512×400 (2×)", r: 2 },
  { label: "768×600 (3×)", r: 3 },
  { label: "1024×800 (4×)", r: 4 },
  { label: "1536×1200 (6×)", r: 6 },
];

const $ = (id) => document.getElementById(id);

const el = {
  canvas: $("view"),
  stage: $("stage"),
  gate: $("gate"),
  note: $("note"),
  res: $("resSelect"),
  plToggle: $("plToggle"),
  settings: $("settings"),
  opacity: $("opacity"),
  dedither: $("dedither"),
  soften: $("soften"),
  vOpacity: $("vOpacity"),
  vDedither: $("vDedither"),
  vSoften: $("vSoften"),
  openSettings: $("openSettings"),
  langToggle: $("langToggle"),
  closeSettings: $("closeSettings"),
  backdrop: $("backdrop"),
  drawer: $("drawer"),
  romFile: $("romFile"),
  gateTitle: $("gateTitle"),
  romSummary: $("romSummary"),
  romStatus: $("romStatus"),
  romName: $("romName"),
  romDetail: $("romDetail"),
  launch: $("launch"),
  launchLabel: $("launchLabel"),
  launchBar: $("launchBar"),
  launchPercent: $("launchPercent"),
  launchHint: $("launchHint"),
  launchError: $("launchError"),
  launchTasks: { core: $("launchCore"), stage: $("launchStage"), model: $("launchModel") },
  retryLaunch: $("retryLaunch"),
  battleStatus: $("battleStatus"),
  actionLabel: $("actionLabel"),
  activeDelay: $("activeDelay"),
  holdInput: $("holdInput"),
  restart: $("restart"),
  fullscreen: $("fullscreen"),
  sound: $("sound"),
  resHint: $("resHint"),
  delayFrames: $("delayFrames"),
  jitterP: $("jitterP"),
  vDelay: $("vDelay"),
  vJitter: $("vJitter"),
  stageSelect: $("stageSelect"),
};

const state = {
  gl: null,
  program: null,
  texture: null,
  uniforms: {},
  renderScale: 3,
  raf: 0,
  failed: null,
  rom: null,
  romBytes: null,
  session: null,
  busy: false,
  validating: false,
  warm: false,
  warming: false,
  nextFrame: null,
  needsDraw: true,
  keyHeld: false,
  pointerHeld: false,
};

// Runtime text, kept as message keys so switching language re-renders it.
const view = {
  gateTitle: { key: "gate.title", params: {} },
  launchLabel: { key: "launch.label.start", params: {} },
  launchHint: { key: "launch.hint.default", params: {} },
  launchError: null,
  rom: null,
  tasks: {},
};

function renderGateTitle() {
  el.gateTitle.textContent = t(view.gateTitle.key, view.gateTitle.params);
}

function setGateTitle(key, params = {}) {
  view.gateTitle = { key, params };
  renderGateTitle();
}

function setLaunchLabel(key, params = {}) {
  view.launchLabel = { key, params };
  el.launchLabel.textContent = t(key, params);
}

function setLaunchHint(key, params = {}) {
  view.launchHint = { key, params };
  el.launchHint.textContent = t(key, params);
}

function renderLaunchError() {
  const error = view.launchError;
  if (!error) return;
  el.launchError.textContent = t(error.key, error.params);
  el.launchError.hidden = el.retryLaunch.hidden = false;
}

function renderRom() {
  const rom = view.rom;
  if (!rom) return;
  el.romSummary.hidden = false;
  el.romSummary.dataset.status = rom.status;
  el.romStatus.textContent = t(rom.key, rom.params);
  el.romName.textContent = rom.file.name;
  el.romName.title = rom.file.name;
  const detail = rom.detailKey ? t(rom.detailKey, rom.detailParams) : "";
  el.romDetail.textContent = detail;
  el.romDetail.hidden = !detail;
}

function renderTasks() {
  for (const [task, row] of Object.entries(view.tasks)) {
    const item = el.launchTasks[task];
    item.dataset.status = row.status;
    item.textContent = t(row.key, row.params);
  }
}

// ------------------------------------------------------------------ WebGL
function initGL() {
  const gl = el.canvas.getContext("webgl2", {
    alpha: false,
    antialias: false,
    desynchronized: true,
    preserveDrawingBuffer: false,
  });
  if (!gl) return null;

  const compile = (type, source) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error(`${type === gl.VERTEX_SHADER ? "vertex" : "fragment"} shader: ${log}`);
    }
    return sh;
  };

  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`link: ${gl.getProgramInfoLog(program)}`);
  }

  // Full-screen triangle; cheaper than a quad and avoids the diagonal seam.
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(program, "aPos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  gl.useProgram(program);
  gl.uniform1i(gl.getUniformLocation(program, "uVideo"), 0);

  const uniforms = {};
  for (const n of UNIFORMS) uniforms[n] = gl.getUniformLocation(program, n);

  return { gl, program, texture, uniforms };
}

function applyResolution() {
  const r = state.renderScale;
  el.canvas.width = NATIVE.w * r;
  el.canvas.height = NATIVE.h * r;
  updateUniforms();
}

function updateUniforms() {
  const { gl, uniforms } = state;
  if (!gl) return;
  const r = state.renderScale;

  gl.useProgram(state.program);
  gl.uniform2f(uniforms.uTexel, 1 / NATIVE.w, 1 / NATIVE.h);
  gl.uniform1f(uniforms.uPeriod, r);
  // Matches the shader/py rule: a scanline needs >= 3 output px per game row.
  gl.uniform1f(uniforms.uActive, el.plToggle.checked && r >= 3 ? 1 : 0);
  gl.uniform1f(uniforms.uOpacity, Number(el.opacity.value));
  gl.uniform1f(uniforms.uDedither, Number(el.dedither.value));
  gl.uniform1f(uniforms.uSoften, Number(el.soften.value));

  el.vOpacity.textContent = Number(el.opacity.value).toFixed(2);
  el.vDedither.textContent = Number(el.dedither.value).toFixed(2);
  el.vSoften.textContent = Number(el.soften.value).toFixed(2);
  state.needsDraw = true;
}

function draw() {
  const { gl } = state;
  if (!gl) return;
  gl.viewport(0, 0, el.canvas.width, el.canvas.height);

  const source = state.session ? state.session.paint() : null;
  if (source) {
    gl.bindTexture(gl.TEXTURE_2D, state.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  }
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function startLoop() {
  if (state.raf) return;
  const tick = (now) => {
    state.raf = requestAnimationFrame(tick);
    if (state.session) {
      if (state.nextFrame === null) state.nextFrame = now;
      state.nextFrame = Math.max(state.nextFrame, now - FRAME_MS * (MAX_CATCHUP_FRAMES - 1));
      try {
        for (let n = 0; n < MAX_CATCHUP_FRAMES && now >= state.nextFrame; n++) {
          state.nextFrame += FRAME_MS;
          const result = state.session.stepFrame();
          state.needsDraw = true;
          if (result) updateLiveStatus(result);
        }
      } catch (err) {
        el.note.className = "note warn";
        el.note.textContent = t("runtime.error", { message: err.message || err });
        state.session.dispose();
        state.session = null;
        syncBattleControls();
      }
    }
    if (state.needsDraw) {
      draw();
      state.needsDraw = false;
    }
  };
  state.raf = requestAnimationFrame(tick);
}

function updateLiveStatus(last) {
  if (!state.session) return;
  if (last) state.lastAction = last.action;
  const s = state.session.stats;
  const hp = value => value >= 0xff00 ? 0 : value;
  const parts = [
    t("note.hp", { p1: hp(state.session.p1hp), p2: hp(state.session.p2hp) }),
    t("note.ko", { p1: s.p2KO, p2: s.p1KO }),
    t("note.frames", { n: s.frames.toLocaleString() }),
  ];
  if (state.session.inputPaused) el.actionLabel.textContent = t("note.inputPaused");
  else if (state.lastAction !== undefined) el.actionLabel.textContent = actionLabels(getLang())[state.lastAction];
  const delay = state.session.baseDelay;
  const queued = state.session.fifo.lastDelay;
  el.activeDelay.textContent = queued > delay
    ? t("delay.queued", { n: delay, q: queued })
    : t("delay.base", { n: delay });
  el.note.className = "note";
  const detail = parts.join(" · ");
  if (el.note.textContent !== detail) el.note.textContent = detail;
}

function syncBattleControls() {
  el.battleStatus.hidden = !state.session;
  el.fullscreen.disabled = !state.session;
  const session = state.session;
  el.sound.hidden = !session || !window.AudioContext || !!session.pendingAudio
    || session.audio?.ctx.state === "running";
  const loading = state.busy || state.validating;
  el.romFile.disabled = loading;
  el.restart.disabled = loading || !state.romBytes;
  el.stageSelect.disabled = loading;
}

function syncInputHold() {
  const held = state.keyHeld || state.pointerHeld;
  el.holdInput.setAttribute("aria-pressed", String(held));
  const session = state.session;
  if (session?.inputPaused && !held) el.actionLabel.textContent = "—";
  session?.setInputPaused(held);
  updateLiveStatus();
}

function clearInputHold() {
  state.keyHeld = state.pointerHeld = false;
  syncInputHold();
}

// ----------------------------------------------------------------- drawer
function openDrawer() {
  clearInputHold();
  el.drawer.classList.add("open");
  el.drawer.setAttribute("aria-hidden", "false");
  el.backdrop.hidden = false;
  el.openSettings.setAttribute("aria-expanded", "true");
  el.closeSettings.focus();
}

function closeDrawer() {
  el.drawer.classList.remove("open");
  el.drawer.setAttribute("aria-hidden", "true");
  el.backdrop.hidden = true;
  el.openSettings.setAttribute("aria-expanded", "false");
  (state.session ? el.stage : el.openSettings).focus({ preventScroll: true });
}

// -------------------------------------------------------------------- ROM
const fmtBytes = (n) => (n >= 1048576 ? `${(n / 1048576).toFixed(2)} MB` : `${(n / 1024).toFixed(1)} KB`);

// The one dump this build is trained and tested against.
const ROM_SPEC = {
  name: "Street Fighter II' - Special Champion Edition (USA).md",
  size: 3145728,
  sha256: "d9224423c80754a4ee945f7348d7478d56a134654a9389e02cd82caab76dddef",
};
const fixtureUrlFor = (stage) => `cores/fixtures/ladder/${stage}.state.gz?v=${FIXTURE_VERSION}`;

function showRomStatus(file, key, params = {}, status = "pending", detailKey = null, detailParams = {}) {
  view.rom = { file, key, params, status, detailKey, detailParams };
  renderRom();
}

async function sha256Hex(buffer) {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** Pick a ROM: verify it, then start the fight straight away. */
async function handleRom(file) {
  if (!file || state.busy || state.validating) return;
  el.launch.hidden = true;
  setGateTitle("gate.title");
  showRomStatus(file, "rom.verifying");
  if (file.size !== ROM_SPEC.size) {
    showRomStatus(file, "rom.mismatch", {}, "error", "rom.badSize.detail", {
      size: file.size.toLocaleString(),
      required: ROM_SPEC.size.toLocaleString(),
    });
    syncBattleControls();
    return;
  }

  state.validating = true;
  syncBattleControls();
  let verified = false;
  try {
    const data = await withDeadline(file.arrayBuffer(), 15_000, t("rom.readTimeout"));
    const bytes = new Uint8Array(data);
    const hex = await withDeadline(sha256Hex(bytes), 15_000, t("rom.shaTimeout"));
    if (!hex) {
      showRomStatus(file, "rom.unverified", {}, "error", "rom.noCrypto.detail");
      return;
    }
    if (hex !== ROM_SPEC.sha256) {
      showRomStatus(file, "rom.mismatch", {}, "error", "rom.badHash.detail", {
        name: ROM_SPEC.name,
        expected: ROM_SPEC.sha256,
        actual: hex,
      });
      return;
    }
    state.romBytes = bytes;
    state.rom = { name: file.name, size: file.size, hash: hex };
    verified = true;
    showRomStatus(file, "rom.ok", {}, "ok");
  } catch (err) {
    showRomStatus(file, "rom.unreadable", {}, "error", "error.raw", { message: String(err.message || err) });
  } finally {
    state.validating = false;
    syncBattleControls();
  }
  // Only a ROM verified just now starts a load; a failed attempt must not
  // silently restart the battle with a previously verified one.
  if (verified) await startBattle();
}

// Rough byte sizes of the three launch downloads: they weight the overall bar
// before Content-Length arrives, and keep it honest for tiny files.
const TASK_EXPECTED = { core: 3_004_000, stage: 48_000, model: 13_614_320 };
const taskBytes = Object.fromEntries(
  Object.entries(TASK_EXPECTED).map(([task, total]) => [task, { got: 0, total }]),
);

function showTask(task, key, params = {}, status = "pending", progress = null) {
  view.tasks[task] = { key, params, status };
  el.launchTasks[task].dataset.status = status;
  el.launchTasks[task].textContent = t(key, params);
  const bytes = taskBytes[task];
  if (progress) {
    bytes.got = progress.got;
    if (progress.total > 0) bytes.total = progress.total;
  }
  if (status === "ok") bytes.got = bytes.total;
  refreshProgress();
}

/** Weighted overall bar so the 13 MB policy does not look like a new stall. */
function refreshProgress() {
  let got = 0;
  let total = 0;
  for (const bytes of Object.values(taskBytes)) {
    got += Math.min(bytes.got, bytes.total);
    total += bytes.total;
  }
  // Hold 1% back while a launch is running, so "100%" never sits there.
  const cap = state.busy ? 99 : 100;
  const pct = total > 0 ? Math.min(cap, Math.round((got / total) * 100)) : 0;
  el.launchBar.value = pct;
  el.launchPercent.textContent = `${pct}%`;
}

/** One row of the launch table: "download with bytes" -> "ready" or the error. */
function runTask(task, work, { signal, abort } = {}) {
  showTask(task, "task.loading");
  const onProgress = (bytes, total) => {
    if (signal?.aborted) return;
    const progress = total > 0 ? `${fmtBytes(bytes)} / ${fmtBytes(total)}` : fmtBytes(bytes);
    showTask(task, "task.downloading", { progress }, "pending", { got: bytes, total });
  };
  return work(onProgress).then(
    () => { if (!signal?.aborted) showTask(task, "task.ready", {}, "ok"); },
    (error) => {
      if (!signal?.aborted) {
        showTask(task, "error.raw", { message: String(error.message || error) }, "error");
        abort?.(error);
      }
      throw error;
    },
  );
}

/**
 * Warm the emulator, the current stage save and the policy while the visitor is
 * still looking for the ROM. Nothing is shown while this runs: the launch table
 * is only revealed once a ROM is picked. live.js shares these promises with
 * startBattle, so whichever path runs second reuses the first one's work.
 */
async function warmUp() {
  if (navigator.connection?.saveData) return;     // respect data-saver
  state.warming = true;
  for (const task of Object.keys(el.launchTasks)) showTask(task, "task.wait");
  try {
    await Promise.all([
      runTask("core", onProgress => prepareCore("cores/", { onProgress })),
      runTask("stage", onProgress => prepareFixture(fixtureUrlFor(el.stageSelect.value), { onProgress })),
      runTask("model", onProgress => preparePolicy("models/", { onProgress })),
    ]);
    state.warm = true;
  } catch (error) {
    // Keep the page usable; selecting a ROM retries only what is missing.
    state.warm = false;
    console.warn("预载失败", error);
  } finally {
    state.warming = false;
  }
}

async function startAudio(session) {
  try {
    const setup = session.initAudio();
    syncBattleControls();
    await setup;
    if (state.session !== session) return;
    const ctx = session.audio?.ctx;
    if (ctx) ctx.onstatechange = syncBattleControls;
    session.resumeAudio().catch(error => console.warn("音频启动失败", error));
  } catch (error) {
    console.warn("音频初始化失败", error);
  } finally {
    syncBattleControls();
  }
}

/**
 * Load the selected stage and start (or restart) the battle.  A verified ROM and
 * the stage picker both call this directly, so no extra button is needed.
 */
async function startBattle() {
  if (!state.romBytes || state.busy || state.validating) return;
  state.busy = true;
  state.session?.dispose?.();
  state.session = null;
  state.lastAction = undefined;
  clearInputHold();
  syncBattleControls();

  el.gate.classList.remove("hidden");
  setGateTitle("gate.starting");
  el.launch.hidden = false;
  setLaunchLabel("launch.label.start");
  setLaunchHint("launch.hint.start");
  el.launchError.hidden = el.retryLaunch.hidden = true;
  view.launchError = null;
  el.note.textContent = "";
  if (!state.warm && !state.warming) {
    for (const [task, bytes] of Object.entries(taskBytes)) {
      bytes.got = 0;
      bytes.total = TASK_EXPECTED[task];
    }
    for (const task of Object.keys(el.launchTasks)) showTask(task, "task.wait");
  }

  const controller = new AbortController();
  const { signal } = controller;
  const loadTask = (task, work) => runTask(task, work, {
    signal,
    abort: error => controller.abort(error),
  });
  let sess;
  try {
    if (!state.gl) throw new Error(t("webgl.noContext"));
    const stage = el.stageSelect ? el.stageSelect.value : "stage01";
    // The .gz fixtures are regenerated in place, so version their URL or a
    // browser may keep serving a stale save (which looks like a broken stage).
    const fixtureUrl = fixtureUrlFor(stage);
    sess = new LiveSession({ fixtureUrl });
    await Promise.all([
      (async () => {
        await loadTask("core", async onProgress => {
          await sess.loadCore({ signal, onProgress });
          await sess.loadRom(state.romBytes);
        });
        if (signal.aborted) throw signal.reason;
        await loadTask("stage", async onProgress => {
          sess.stats.fixture = await sess.loadFixture(fixtureUrl, { signal, onProgress });
        });
      })(),
      loadTask("model", onProgress => sess.loadPolicy({ signal, onProgress })),
    ]);
    await sess.initObserver();

    state.session = sess;
    state.nextFrame = null;
    state.needsDraw = true;
    sess.setDelay(Number(el.delayFrames.value));
    sess.setJitter(Number(el.jitterP.value));
    el.launch.hidden = true;
    el.gate.classList.add("hidden");
    el.actionLabel.textContent = "—";
    syncInputHold();
    if (!el.drawer.classList.contains("open")) el.stage.focus({ preventScroll: true });
    startAudio(sess);
  } catch (err) {
    controller.abort(err);
    sess?.dispose();
    for (const [task, item] of Object.entries(el.launchTasks)) {
      if (item.dataset.status === "pending") showTask(task, "task.stopped");
    }
    const taskError = Object.values(el.launchTasks).some(item => item.dataset.status === "error");
    setGateTitle(taskError ? "gate.failed" : "gate.title");
    setLaunchLabel("launch.label.failed");
    setLaunchHint(taskError ? "launch.hint.failed" : "launch.hint.stopped");
    view.launchError = taskError
      ? { key: "launch.error.retry", params: {} }
      : { key: "error.raw", params: { message: String(err.message || err) } };
    renderLaunchError();
  } finally {
    state.busy = false;
    syncBattleControls();
  }
}

// ------------------------------------------------------------------- boot
function currentLanguageButton() {
  el.langToggle.textContent = getLang() === "zh" ? "EN" : "中文";
  el.langToggle.setAttribute("aria-label", t("lang.switch"));
}

/** Re-render the text that lives outside the static data-i18n attributes. */
function renderDynamic() {
  renderGateTitle();
  renderLaunchLabels();
  renderRom();
  renderTasks();
  renderLaunchError();
  if (state.session) updateLiveStatus();
  else el.activeDelay.textContent = t("delay.base", { n: 0 });
}

function renderLaunchLabels() {
  el.launchLabel.textContent = t(view.launchLabel.key, view.launchLabel.params);
  el.launchHint.textContent = t(view.launchHint.key, view.launchHint.params);
}

async function boot() {
  applyI18n();
  currentLanguageButton();
  el.activeDelay.textContent = t("delay.base", { n: 0 });
  onLangChange(() => {
    currentLanguageButton();
    renderDynamic();
  });
  el.langToggle.addEventListener("click", () => setLang(getLang() === "zh" ? "en" : "zh"));

  el.romFile.addEventListener("change", () => {
    const file = el.romFile.files?.[0];
    el.romFile.value = "";
    handleRom(file);
  });
  el.retryLaunch.addEventListener("click", startBattle);
  for (const [i, r] of RENDERS.entries()) {
    el.res.add(new Option(r.label, String(r.r), i === 2, i === 2));
  }
  state.renderScale = 3;

  let glc;
  try {
    glc = initGL();
  } catch (err) {
    state.failed = String(err.message || err);
    el.note.className = "note warn";
    el.note.innerHTML = `<strong>${escapeHtml(t("webgl.initFailed"))}</strong>: ${escapeHtml(state.failed)}`;
    return;
  }
  if (!glc) {
    state.failed = "no-webgl2";
    el.note.className = "note warn";
    el.note.textContent = `${t("webgl.missing")}${t("webgl.missingHint")}`;
    return;
  }
  Object.assign(state, glc);

  applyResolution();
  startLoop();
  // Fetch the emulator, this stage's save and the policy now, so choosing a ROM
  // afterwards only costs load_game + unserialize.
  warmUp();

  el.res.addEventListener("change", () => {
    state.renderScale = Number(el.res.value);
    applyResolution();
    el.settings.classList.toggle("disabled", state.renderScale < 3);
  });
  el.plToggle.addEventListener("change", updateUniforms);
  for (const s of [el.opacity, el.dedither, el.soften]) {
    s.addEventListener("input", updateUniforms);
  }

  el.openSettings.addEventListener("click", openDrawer);
  el.closeSettings.addEventListener("click", closeDrawer);
  el.backdrop.addEventListener("click", closeDrawer);
  el.drawer.addEventListener("click", (e) => e.stopPropagation());
  el.restart.addEventListener("click", startBattle);
  el.sound.addEventListener("click", () => {
    if (state.session) startAudio(state.session);
  });
  const fullscreen = () => {
    const result = document.fullscreenElement ? document.exitFullscreen() : el.stage.requestFullscreen?.();
    result?.catch(() => {});
  };
  el.fullscreen.addEventListener("click", fullscreen);
  el.fullscreen.hidden = !document.fullscreenEnabled;

  // Picking another stage restarts the fight right away.
  el.stageSelect.addEventListener("change", () => {
    prepareFixture(fixtureUrlFor(el.stageSelect.value)).catch(() => {});
    if (!state.romBytes) return;
    startBattle();
  });

  // Delay / jitter controls act on the running session and stick for later runs.
  const applyDelay = () => {
    const v = Number(el.delayFrames.value);
    el.vDelay.textContent = String(v);
    state.session?.setDelay(v);
    updateLiveStatus();
  };
  const applyJitter = () => {
    const v = Number(el.jitterP.value);
    el.vJitter.textContent = `${Math.round(v * 100)}%`;
    state.session?.setJitter(v);
  };
  el.delayFrames.addEventListener("input", applyDelay);
  el.jitterP.addEventListener("input", applyJitter);

  el.holdInput.addEventListener("pointerdown", e => {
    if (!e.isPrimary || e.button !== 0) return;
    e.preventDefault();
    el.holdInput.setPointerCapture(e.pointerId);
    state.pointerHeld = true;
    syncInputHold();
  });
  el.holdInput.addEventListener("contextmenu", e => e.preventDefault());
  el.holdInput.addEventListener("lostpointercapture", () => {
    state.pointerHeld = false;
    syncInputHold();
  });
  el.holdInput.addEventListener("keydown", e => {
    if (e.code !== "Space" && e.code !== "Enter") return;
    e.preventDefault();
    state.keyHeld = true;
    syncInputHold();
  });
  el.holdInput.addEventListener("keyup", e => {
    if (e.code !== "Space" && e.code !== "Enter") return;
    e.preventDefault();
    state.keyHeld = false;
    syncInputHold();
  });
  el.holdInput.addEventListener("blur", () => {
    state.keyHeld = false;
    syncInputHold();
  });
  window.addEventListener("blur", clearInputHold);
  document.addEventListener("keydown", (e) => {
    if (e.code === "Escape" && el.drawer.classList.contains("open")) {
      closeDrawer();
      return;
    }
    if (e.code === "KeyF" && state.session && !el.drawer.classList.contains("open")) fullscreen();
  });

  el.canvas.addEventListener("dblclick", fullscreen);

  // Exposed so headless checks can assert the page really initialised.
  window.__theater = { state, el };
}

boot();
