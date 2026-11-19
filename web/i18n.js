// Two-language UI for the browser demo.
//
// Static markup carries data-i18n keys; everything built at runtime (ROM
// verdict, launch rows, battle note) goes through t().  The choice is stored in
// localStorage and falls back to the browser language, Chinese when zh-*.

const STORAGE_KEY = "chunli.lang";

const MESSAGES = {
  zh: {
    "app.title": "Chun-Li One Credit · 春丽一命通关",
    "app.subtitle": "春丽一命通关",
    "cabinet.label": "实时对战",
    "lang.switch": "切换界面语言",
    "stage.label": "起始关卡",
    "stage.screen": "游戏画面",
    "stage.01": "01 · 春丽（镜像）",
    "stage.02": "02 · 桑吉尔夫",
    "stage.03": "03 · 达尔锡",
    "stage.04": "04 · 隆",
    "stage.05": "05 · 本田",
    "stage.06": "06 · 布兰卡",
    "stage.07": "07 · 古烈",
    "stage.08": "08 · 肯",
    "stage.09": "09 · 拜森",
    "stage.10": "10 · 维加",
    "stage.11": "11 · 沙加特",
    "stage.12": "12 · 将军",
    "restart": "重开",
    "fullscreen": "全屏",
    "sound.enable": "启用声音",
    "settings.open": "设置",
    "settings.title": "设置",
    "settings.close": "关闭",
    "settings.video": "画面",
    "settings.res": "分辨率",
    "settings.resHint": "扫描线需要 3× 及以上；1×/2× 只放大画面。",
    "settings.scanlines": "扫描线",
    "settings.scanEnable": "启用扫描线滤镜",
    "settings.opacity": "扫描线强度",
    "settings.dedither": "去抖动",
    "settings.soften": "线条柔化",
    "settings.battle": "对战",
    "settings.delay": "输入延迟",
    "settings.jitter": "抖动概率",
    "settings.battleHint": "按住画面下方的按钮可切断春丽的输入，松开后恢复。固定延迟和抖动可单独调整。",
    "gate.title": "载入 ROM，观看春丽实战",
    "gate.picker": "选择 ROM",
    "gate.hint": "3,145,728 字节 · 文件仅在浏览器本地读取",
    "gate.starting": "正在启动…",
    "gate.failed": "启动未完成，可重试",
    "launch.label": "预载与启动进度",
    "launch.core": "模拟器",
    "launch.stage": "关卡存档",
    "launch.model": "策略模型",
    "launch.hint.default": "模拟器、关卡存档与策略模型（神经网络）会提前下载，首次约 16 MB，之后由浏览器缓存复用。",
    "launch.hint.start": "三项资源全部就绪后立即开打。",
    "launch.hint.failed": "已校验的 ROM 会保留，可直接重试启动。",
    "launch.hint.stopped": "启动没有完成，可以重试。",
    "launch.label.start": "启动中",
    "launch.label.failed": "启动未完成",
    "launch.retry": "重试启动",
    "launch.error.retry": "启动失败，可以使用已校验的 ROM 重试。",
    "battle.action": "策略动作",
    "battle.delay": "输入延迟",
    "hold.ready": "按住切断输入",
    "hold.active": "松开恢复",
    "hold.aria": "按住切断输入，松开恢复",
    "unit.frames": "帧",
    "rom.verifying": "正在校验 ROM",
    "rom.ok": "ROM 校验通过",
    "rom.mismatch": "ROM 不符合要求",
    "rom.unverified": "无法校验 ROM",
    "rom.unreadable": "无法读取 ROM",
    "rom.badSize.detail": "文件大小为 {size} 字节，要求 {required} 字节。",
    "rom.badHash.detail": "SHA-256 不匹配。需要 {name}（{expected}），当前文件是 {actual}。",
    "rom.noCrypto.detail": "SHA-256 校验需要 HTTPS 或 localhost。",
    "rom.readTimeout": "本地文件读取超时，请重新选择。",
    "rom.shaTimeout": "SHA-256 校验超时，请重试。",
    "task.wait": "等待",
    "task.loading": "加载中…",
    "task.downloading": "下载中 · {progress}",
    "task.ready": "已就绪",
    "task.stopped": "已停止",
    "note.hp": "HP {p1} : {p2}",
    "note.ko": "KO {p1} : {p2}",
    "note.frames": "{n} 帧",
    "note.inputPaused": "输入已切断",
    "runtime.error": "实时运行出错：{message}",
    "delay.base": "{n} 帧",
    "delay.queued": "{n} 帧 · 最近排队 {q} 帧",
    "webgl.initFailed": "WebGL 初始化失败",
    "webgl.missing": "浏览器不支持 WebGL2",
    "webgl.missingHint": "，无法应用荧光线滤镜。",
    "webgl.noContext": "当前浏览器无法初始化 WebGL2",
    "core.scriptTimeout": "模拟器脚本下载超时",
    "core.scriptInvalid": "模拟器脚本格式错误",
    "core.scriptFailed": "无法下载模拟器脚本",
    "core.compileTimeout": "模拟器编译超时",
    "core.initTimeout": "模拟器初始化超时",
    "core.romFailed": "ROM 无法载入（不受支持或已损坏）",
    "audio.moduleTimeout": "音频组件载入超时",
    "fixture.gzipUnsupported": "当前浏览器不支持 gzip 解压，无法载入压缩存档",
    "fixture.gzipTimeout": "关卡存档解压超时",
    "fixture.invalid": "关卡存档的 Genesis Plus GX 格式无效",
    "fixture.loadFailed": "存档 {url} 无法载入（格式或版本不匹配）",
    "download.failed": "下载失败（HTTP {status}）",
    "download.stalled": "连接或下载长时间未响应",
    "policy.incomplete": "策略模型下载不完整",
    "error.raw": "{message}",
  },
  en: {
    "app.title": "Chun-Li One Credit",
    "app.subtitle": "One credit with Chun-Li",
    "cabinet.label": "Live match",
    "lang.switch": "Switch interface language",
    "stage.label": "Start stage",
    "stage.screen": "Game screen",
    "stage.01": "01 · Chun-Li (mirror)",
    "stage.02": "02 · Zangief",
    "stage.03": "03 · Dhalsim",
    "stage.04": "04 · Ryu",
    "stage.05": "05 · E. Honda",
    "stage.06": "06 · Blanka",
    "stage.07": "07 · Guile",
    "stage.08": "08 · Ken",
    "stage.09": "09 · Balrog",
    "stage.10": "10 · Vega",
    "stage.11": "11 · Sagat",
    "stage.12": "12 · M. Bison",
    "restart": "Restart",
    "fullscreen": "Fullscreen",
    "sound.enable": "Enable sound",
    "settings.open": "Settings",
    "settings.title": "Settings",
    "settings.close": "Close",
    "settings.video": "Video",
    "settings.res": "Resolution",
    "settings.resHint": "Scanlines need 3× or more; 1×/2× only scale the picture.",
    "settings.scanlines": "Scanlines",
    "settings.scanEnable": "Enable scanline filter",
    "settings.opacity": "Scanline strength",
    "settings.dedither": "De-dither",
    "settings.soften": "Line softening",
    "settings.battle": "Match",
    "settings.delay": "Input delay",
    "settings.jitter": "Jitter chance",
    "settings.battleHint": "Hold the button under the screen to cut Chun-Li's input and release to resume. Fixed delay and jitter can be tuned separately.",
    "gate.title": "Load the ROM and watch Chun-Li fight",
    "gate.picker": "Choose ROM",
    "gate.hint": "3,145,728 bytes · the file is read locally in your browser",
    "gate.starting": "Starting…",
    "gate.failed": "Start did not finish — you can retry",
    "launch.label": "Preload and start progress",
    "launch.core": "Emulator",
    "launch.stage": "Savestate",
    "launch.model": "Policy model",
    "launch.hint.default": "The emulator, this stage's savestate and the policy network are fetched up front, about 16 MB the first time, then the browser cache takes over.",
    "launch.hint.start": "The match starts as soon as all three resources are ready.",
    "launch.hint.failed": "Your verified ROM is kept — retry the start.",
    "launch.hint.stopped": "The start did not finish; feel free to retry.",
    "launch.label.start": "Starting",
    "launch.label.failed": "Start failed",
    "launch.retry": "Retry start",
    "launch.error.retry": "Start failed; you can retry with the verified ROM.",
    "battle.action": "Policy action",
    "battle.delay": "Input delay",
    "hold.ready": "Hold to cut input",
    "hold.active": "Release to resume",
    "hold.aria": "Hold to cut the policy input, release to resume",
    "unit.frames": "frames",
    "rom.verifying": "Checking ROM",
    "rom.ok": "ROM verified",
    "rom.mismatch": "Unsupported ROM",
    "rom.unverified": "Cannot verify ROM",
    "rom.unreadable": "Cannot read ROM",
    "rom.badSize.detail": "This file is {size} bytes; {required} bytes are required.",
    "rom.badHash.detail": "SHA-256 mismatch. Expected {name} ({expected}); this file hashes to {actual}.",
    "rom.noCrypto.detail": "SHA-256 needs HTTPS or localhost.",
    "rom.readTimeout": "Reading the local file timed out; pick it again.",
    "rom.shaTimeout": "The SHA-256 check timed out; try again.",
    "task.wait": "Waiting",
    "task.loading": "Loading…",
    "task.downloading": "Downloading · {progress}",
    "task.ready": "Ready",
    "task.stopped": "Stopped",
    "note.hp": "HP {p1} : {p2}",
    "note.ko": "KO {p1} : {p2}",
    "note.frames": "{n} frames",
    "note.inputPaused": "Input cut",
    "runtime.error": "Live run error: {message}",
    "delay.base": "{n} frames",
    "delay.queued": "{n} frames · last queued {q}",
    "webgl.initFailed": "WebGL initialisation failed",
    "webgl.missing": "This browser has no WebGL2",
    "webgl.missingHint": ", so the phosphor filter cannot be applied.",
    "webgl.noContext": "WebGL2 could not be initialised in this browser",
    "core.scriptTimeout": "The emulator script download timed out",
    "core.scriptInvalid": "The emulator script is malformed",
    "core.scriptFailed": "Could not download the emulator script",
    "core.compileTimeout": "Compiling the emulator timed out",
    "core.initTimeout": "Initialising the emulator timed out",
    "core.romFailed": "retro_load_game failed (unsupported or corrupt ROM)",
    "audio.moduleTimeout": "Loading the audio worklet timed out",
    "fixture.gzipUnsupported": "This browser cannot gunzip the compressed savestate",
    "fixture.gzipTimeout": "Decompressing the savestate timed out",
    "fixture.invalid": "The savestate is not a Genesis Plus GX state",
    "fixture.loadFailed": "Could not load {url} (format or version mismatch)",
    "download.failed": "Download failed (HTTP {status})",
    "download.stalled": "The connection stalled",
    "policy.incomplete": "The policy weights did not download completely",
    "error.raw": "{message}",
  },
};

const listeners = new Set();
let lang = detect();

function detect() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "zh" || saved === "en") return saved;
  } catch {
    // private mode: fall through to the browser language
  }
  const preferred = (navigator.languages?.[0] || navigator.language || "").toLowerCase();
  return preferred.startsWith("zh") ? "zh" : "en";
}

export function getLang() {
  return lang;
}

export function t(key, params = null) {
  const template = MESSAGES[lang]?.[key] ?? MESSAGES.zh[key] ?? key;
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (match, name) => (name in params ? String(params[name]) : match));
}

/** Re-render every static node; runtime text re-renders through onLangChange. */
export function applyI18n(root = document) {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.title = t("app.title");
  for (const node of root.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const node of root.querySelectorAll("[data-i18n-title]")) node.title = t(node.dataset.i18nTitle);
  for (const node of root.querySelectorAll("[data-i18n-aria]")) node.setAttribute("aria-label", t(node.dataset.i18nAria));
}

export function setLang(next) {
  if (next !== "zh" && next !== "en") return;
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // storage can be unavailable; the switch still works for this page load
  }
  if (next === lang) return;
  lang = next;
  applyI18n();
  for (const listener of listeners) listener(lang);
}

export function onLangChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
