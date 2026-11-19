// SF2 one-life run, live in the browser.
// Static page: no server, no build step. The Genesis core (WASM) runs the ROM
// in the page, the chunppo policy picks an action every 4 frames, and every
// frame is pushed through a WebGL2 shader with a configurable
// scanline effect.

import { VERT, FRAG, UNIFORMS } from "./shader.js?v=15";
import { FIXTURE_VERSION, LiveSession } from "./live.js?v=18";
import { ACTION_LABELS } from "./actions.js?v=10";

const NATIVE = { w: 256, h: 200 }; // game pixels after the standard crop
const FRAME_MS = 1000 / 59.92274340431231;
const MAX_CATCHUP_FRAMES = 3;

const RENDERS = [
  { label: "256×200（1×）", r: 1 },
  { label: "512×400（2×）", r: 2 },
  { label: "768×600（3×）", r: 3 },
  { label: "1024×800（4×）", r: 4 },
  { label: "1536×1200（6×）", r: 6 },
];

const $ = (id) => document.getElementById(id);

const el = {
  canvas: $("view"),
  stage: $("stage"),
  gate: $("gate"),
  note: $("note"),
  model: null,
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
  closeSettings: $("closeSettings"),
  backdrop: $("backdrop"),
  drawer: $("drawer"),
  romFile: $("romFile"),
  romStatus: $("romStatus"),
  fixtureFiles: $("fixtureFiles"),
  fixtureStatus: $("fixtureStatus"),
  battleStatus: $("battleStatus"),
  actionLabel: $("actionLabel"),
  activeDelay: $("activeDelay"),
  holdDelay: $("holdDelay"),
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
  fixtures: new Map(),
  session: null,
  busy: false,
  nextFrame: null,
  needsDraw: true,
  spaceHeld: false,
  pointerHeld: false,
};

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
        el.note.textContent = `实时运行出错：${err.message || err}`;
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
  const s = state.session.stats;
  const hp = value => value >= 0xff00 ? 0 : value;
  const parts = [
    `HP ${hp(state.session.p1hp)} : ${hp(state.session.p2hp)}`,
    `KO ${s.p2KO} : ${s.p1KO}`,
    `${s.frames.toLocaleString()} 帧`,
  ];
  if (state.session.frozen) parts.push("输入暂停");
  if (last) el.actionLabel.textContent = ACTION_LABELS[last.action];
  const delay = state.session.effectiveDelay;
  const queued = state.session.fifo.lastDelay;
  el.activeDelay.textContent = queued > delay ? `${delay} 帧 · 最近排队 ${queued} 帧` : `${delay} 帧`;
  el.note.className = "note";
  const detail = parts.join(" · ");
  if (el.note.textContent !== detail) el.note.textContent = detail;
}

function syncBattleControls() {
  el.battleStatus.hidden = !state.session;
  el.fullscreen.disabled = !state.session;
  const audio = state.session?.audio?.ctx;
  el.sound.hidden = !audio || audio.state === "running";
  el.restart.disabled = state.busy || !state.romBytes;
  el.stageSelect.disabled = state.busy;
  el.fixtureFiles.disabled = state.busy;
}

function syncDelayHold() {
  const held = state.spaceHeld || state.pointerHeld;
  el.holdDelay.setAttribute("aria-pressed", String(held));
  state.session?.setSpaceHeld(held);
  updateLiveStatus();
}

// ----------------------------------------------------------------- drawer
function openDrawer() {
  state.spaceHeld = false;
  state.session?.setFrozen(false);
  syncDelayHold();
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

function romError(html) {
  el.romStatus.hidden = false;
  el.romStatus.className = "rom-status warn";
  el.romStatus.innerHTML = html;
}

async function sha256Hex(buffer) {
  if (!crypto?.subtle) return null;
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** Pick a ROM: verify it, then start the fight straight away. */
async function handleRom(file) {
  if (!file) return;
  el.romStatus.hidden = true;
  el.romStatus.className = "rom-status";

  let bytes, hex;
  try {
    bytes = new Uint8Array(await file.arrayBuffer());
    hex = await sha256Hex(bytes);
  } catch (err) {
    romError(`<b>读不了这个文件</b>：${escapeHtml(String(err.message || err))}`);
    return;
  }

  if (!hex) {
    romError("<b>这个浏览器无法计算 sha256</b>（需要安全上下文 HTTPS 或 localhost），换用 Chrome / Edge 再试。");
    return;
  }

  if (hex !== ROM_SPEC.sha256) {
    const sizeNote = file.size === ROM_SPEC.size ? "" : `，大小 ${fmtBytes(file.size)} 也不对（应为 3.00 MB）`;
    romError([
      "<b>ROM 与要求的不一致</b>",
      `你选的是：${escapeHtml(file.name)}${sizeNote}`,
      `实际 sha256：<code>${hex}</code>`,
      `需要：<code>${ROM_SPEC.sha256}</code>`,
      `对应文件：<code>${ROM_SPEC.name}</code>`,
    ].join("<br>"));
    return;
  }

  state.romBytes = bytes;
  state.rom = { name: file.name, size: file.size, hash: hex };
  await startBattle();
}

/**
 * Load the selected stage and start (or restart) the battle.  A verified ROM and
 * the stage picker both call this directly, so no extra button is needed.
 */
async function startBattle() {
  if (!state.romBytes || state.busy) return;
  state.busy = true;
  state.session?.dispose?.();
  state.session = null;
  syncBattleControls();

  el.romStatus.hidden = true;
  el.romStatus.className = "rom-status";
  el.note.className = "note";
  el.note.textContent = "启动模拟器并载入策略…";

  let sess;
  try {
    const stage = el.stageSelect ? el.stageSelect.value : "stage01";
    // The .gz fixtures are regenerated in place, so version their URL or a
    // browser may keep serving a stale save (which looks like a broken stage).
    const fixtureUrl = `cores/fixtures/ladder/${stage}.state.gz?v=${FIXTURE_VERSION}`;
    sess = new LiveSession({ fixtureUrl });
    await sess.loadCore();
    await sess.loadRom(state.romBytes);
    const fixtureOk = await sess.loadFixture(fixtureUrl, state.fixtures.get(stage));
    await sess.loadPolicy();
    await sess.initObserver();
    await sess.initAudio().catch(error => console.warn("音频初始化失败", error));
    sess.resumeAudio().catch(error => console.warn("音频启动失败", error));
    sess.audio?.ctx.addEventListener("statechange", syncBattleControls);

    sess.stats.fixture = fixtureOk;
    state.session = sess;
    state.nextFrame = null;
    state.needsDraw = true;
    sess.setDelay(Number(el.delayFrames.value));
    sess.setJitter(Number(el.jitterP.value));
    el.gate.classList.add("hidden");
    el.actionLabel.textContent = "—";
    syncDelayHold();
    if (!el.drawer.classList.contains("open")) el.stage.focus({ preventScroll: true });
  } catch (err) {
    sess?.dispose();
    el.gate.classList.remove("hidden");
    el.note.textContent = "";
    if (err.status === 404) {
      romError(`此关需要本地存档。请在<strong>设置 → 本地关卡存档</strong>中选择 <code>${el.stageSelect.value}.state.gz</code>。 <a href="https://github.com/Datou/ChunLi-OneCredit/blob/main/docs/savestates.md">存档生成方法</a>`);
    } else {
      romError(`<b>启动失败</b>：${escapeHtml(String(err.message || err))}`);
    }
  } finally {
    state.busy = false;
    syncBattleControls();
  }
}

// ------------------------------------------------------------------- boot
async function boot() {
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
    el.note.innerHTML = `<strong>WebGL 初始化失败</strong>：${escapeHtml(state.failed)}`;
    return;
  }
  if (!glc) {
    state.failed = "no-webgl2";
    el.note.className = "note warn";
    el.note.innerHTML = "<strong>浏览器不支持 WebGL2</strong>，无法应用荧光线滤镜。";
    return;
  }
  Object.assign(state, glc);

  applyResolution();
  startLoop();

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
  el.romFile.addEventListener("change", () => handleRom(el.romFile.files?.[0]));
  el.fixtureFiles.addEventListener("change", () => {
    for (const file of el.fixtureFiles.files) {
      const match = /^(stage(?:0[1-9]|1[0-2]))\.state(?:\.gz)?$/i.exec(file.name);
      if (match) state.fixtures.set(match[1].toLowerCase(), file);
    }
    const stages = [...state.fixtures.keys()].sort().map(name => name.slice(-2));
    el.fixtureStatus.textContent = stages.length ? `已选择关卡 ${stages.join("、")}` : "请选择命名为 stageNN.state.gz 的网页存档";
    el.fixtureFiles.value = "";
    if (state.fixtures.has(el.stageSelect.value)) startBattle();
  });
  el.restart.addEventListener("click", startBattle);
  el.sound.addEventListener("click", () => state.session?.resumeAudio());
  const fullscreen = () => {
    const result = document.fullscreenElement ? document.exitFullscreen() : el.stage.requestFullscreen?.();
    result?.catch(() => {});
  };
  el.fullscreen.addEventListener("click", fullscreen);
  el.fullscreen.hidden = !document.fullscreenEnabled;

  // Picking another stage restarts the fight right away.
  el.stageSelect.addEventListener("change", () => {
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

  el.holdDelay.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    e.preventDefault();
    el.holdDelay.setPointerCapture(e.pointerId);
    state.pointerHeld = true;
    syncDelayHold();
  });
  el.holdDelay.addEventListener("contextmenu", e => e.preventDefault());
  el.holdDelay.addEventListener("lostpointercapture", () => {
    state.pointerHeld = false;
    syncDelayHold();
  });
  window.addEventListener("blur", () => {
    state.spaceHeld = state.pointerHeld = false;
    state.session?.setFrozen(false);
    syncDelayHold();
  });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Escape" && el.drawer.classList.contains("open")) {
      closeDrawer();
      return;
    }
    if (!state.session || el.drawer.classList.contains("open")
        || e.target.closest?.("input, select, textarea, button, a, [contenteditable]") && e.target !== el.holdDelay) return;
    if (e.code === "Space") {
      e.preventDefault();
      if (!e.repeat) {
        state.spaceHeld = true;
        syncDelayHold();
      }
    } else if (e.code === "ShiftLeft" || e.code === "ShiftRight") {
      e.preventDefault();
      state.session.setFrozen(true);
      updateLiveStatus();
    } else if (e.code === "KeyF") {
      fullscreen();
    }
  });
  document.addEventListener("keyup", (e) => {
    if (e.code === "Space") {
      if (state.spaceHeld) e.preventDefault();
      state.spaceHeld = false;
      syncDelayHold();
    } else if (e.code === "ShiftLeft" || e.code === "ShiftRight") {
      state.session?.setFrozen(false);
      updateLiveStatus();
    }
  });

  el.canvas.addEventListener("dblclick", fullscreen);

  // Exposed so headless checks can assert the page really initialised.
  window.__theater = { state, el };
}

boot();
