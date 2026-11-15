"use strict";

const STORAGE_KEY = "danglish.desktop.settings.v3";
const HISTORY_KEY = "danglish.desktop.history.v1";
const DEFAULT_ASR_MODEL = "lucasw89/whisper-small-danish-ct2-int8";

const DEFAULT_SETTINGS = {
  context: "general",
  fullDuplex: true,
  ttsEnabled: true,
  inputDevice: "",
  outputDevice: "",
  inputChannel: 0,
  outputChannel: null,
  silenceMs: 300,
  maxSegmentS: 4.0,
  minEnergy: 0.006,
  asrModel: DEFAULT_ASR_MODEL,
  voice: "M2",
  broadcastEnabled: false,
};

const FALLBACK_CONTEXTS = [
  { name: "general", label: "General", description: "Neutral everyday translation." },
  {
    name: "sermon",
    label: "Church / sermon",
    description: "Prefers congregation, sermon, pastor, and worship terminology.",
  },
];

const VOICE_LABELS = {
  M2: "M2 — US English (male)",
  F2: "F2 — US English (female)",
};

const state = {
  settings: loadSettings(),
  contexts: FALLBACK_CONTEXTS.slice(),
  devices: [],
  deviceDefaults: { input: null, output: null },
  appInfo: { packaged: false, voices: ["M2", "F2"] },
  status: "idle",
  running: false,
  busy: false,
  rss: "",
  inputInfo: "",
  outputInfo: "",
  broadcastUrl: "",
  broadcastQr: "",
  feedStarted: false,
  emptyMarkup: "",
  history: loadHistory(),
};

const $ = (id) => document.getElementById(id);
const els = {
  statusCard: $("statusCard"),
  dot: $("dot"),
  statusText: $("statusText"),
  statusDetail: $("statusDetail"),
  talkBtn: $("talkBtn"),
  talkLabel: $("talkLabel"),
  contextSelect: $("contextSelect"),
  contextHint: $("contextHint"),
  contextBadge: $("contextBadge"),
  fullDuplex: $("fullDuplex"),
  ttsEnabled: $("ttsEnabled"),
  broadcastEnabled: $("broadcastEnabled"),
  broadcastLinks: $("broadcastLinks"),
  broadcastUrl: $("broadcastUrl"),
  broadcastQr: $("broadcastQr"),
  duplexNote: $("duplexNote"),
  settingsBtn: $("settingsBtn"),
  legalBtn: $("legalBtn"),
  clearBtn: $("clearBtn"),
  logsBtn: $("logsBtn"),
  logs: $("logs"),
  feed: $("feed"),
  deviceInfo: $("deviceInfo"),
  lastEvent: $("lastEvent"),
  rssInfo: $("rssInfo"),
  modelInfo: $("modelInfo"),
  dialog: $("settingsDialog"),
  form: $("settingsForm"),
  closeSettings: $("closeSettings"),
  cancelSettings: $("cancelSettings"),
  refreshDevices: $("refreshDevices"),
  inputDevice: $("inputDevice"),
  outputDevice: $("outputDevice"),
  inputChannel: $("inputChannel"),
  outputChannel: $("outputChannel"),
  silenceMs: $("silenceMs"),
  maxSegmentS: $("maxSegmentS"),
  minEnergy: $("minEnergy"),
  asrModel: $("asrModel"),
  voice: $("voice"),
  toast: $("toast"),
};

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_SETTINGS, ...parsed };
  } catch (_) {
    return { ...DEFAULT_SETTINGS };
  }
}

function saveSettings() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.settings));
}

function loadHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(parsed) ? parsed.slice(-100) : [];
  } catch (_) {
    return [];
  }
}

function saveHistory() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history.slice(-100)));
}

function shortDeviceName(name) {
  if (!name) return "";
  return name
    .replace(/^alsa_(input|output)\./, "")
    .replace(/\.analog-surround-\d+$/, "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function truncate(value, length = 48) {
  if (!value) return "";
  return value.length > length ? `${value.slice(0, length - 1)}…` : value;
}

function formatMs(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

function showToast(message, duration = 3800) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => els.toast.classList.remove("show"), duration);
}

function setStatus(kind, message = "") {
  state.status = kind;
  state.running = kind === "listening";
  state.busy = kind === "starting" || kind === "stopping";

  const dotClass = {
    idle: "",
    starting: "is-starting",
    listening: "is-listening",
    stopping: "is-stopping",
    error: "is-error",
  }[kind];
  els.dot.className = `dot ${dotClass || ""}`.trim();
  els.statusCard.classList.toggle("is-error", kind === "error");

  const labels = {
    idle: "Ready",
    starting: "Starting",
    listening: "Listening",
    stopping: "Stopping",
    error: "Error",
  };
  const details = {
    idle: "Everything runs on this machine.",
    starting: message || "Loading the local models…",
    listening: "Speak Danish. Pause when you finish a thought.",
    stopping: "Closing the audio streams…",
    error: message || "Something went wrong.",
  };
  const buttonLabels = {
    idle: "Start listening",
    starting: "Starting…",
    listening: "Stop listening",
    stopping: "Stopping…",
    error: "Start listening",
  };

  els.statusText.textContent = labels[kind];
  els.statusDetail.textContent = details[kind];
  els.talkLabel.textContent = buttonLabels[kind];
  els.talkBtn.disabled = state.busy;
  els.talkBtn.classList.toggle("is-running", kind === "listening");

  if (kind === "listening") {
    showLiveHint();
  } else if (kind === "idle" || kind === "error") {
    removeLiveHint();
  }
}

function showLiveHint() {
  if ($("liveHint")) return;
  const hint = document.createElement("div");
  hint.className = "live-hint";
  hint.id = "liveHint";
  hint.innerHTML =
    '<div class="live-bars" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></div>' +
    "<p>Listening for Danish …</p>";
  els.feed.appendChild(hint);
  els.feed.scrollTop = els.feed.scrollHeight;
}

function removeLiveHint() {
  $("liveHint")?.remove();
}

function updateContextUi() {
  const profile =
    state.contexts.find((item) => item.name === state.settings.context) ||
    state.contexts[0] ||
    FALLBACK_CONTEXTS[0];
  els.contextBadge.textContent = profile.label || profile.name;
  els.contextHint.textContent = profile.description || "";
}

function updateDuplexUi() {
  els.duplexNote.textContent = state.settings.fullDuplex
    ? "Full-duplex is on — use headphones or isolated audio I/O. Turn it off for laptop speakers."
    : "Half-duplex — the mic pauses while English audio plays.";
}

function updateBroadcastUi() {
  const url = state.broadcastUrl;
  const qr = state.broadcastQr;
  els.broadcastLinks.classList.toggle("hidden", !url && !qr);
  if (url) {
    els.broadcastUrl.href = url;
    els.broadcastUrl.title = url;
  }
  if (qr) {
    els.broadcastQr.href = qr;
    els.broadcastQr.title = qr;
  }
}

function populateContexts() {
  els.contextSelect.replaceChildren();
  for (const context of state.contexts) {
    const option = document.createElement("option");
    option.value = context.name;
    option.textContent = context.label;
    els.contextSelect.appendChild(option);
  }
  if (!state.contexts.some((item) => item.name === state.settings.context)) {
    state.settings.context = state.contexts[0]?.name || "general";
  }
  els.contextSelect.value = state.settings.context;
  updateContextUi();
}

function populateVoices() {
  const voices =
    state.appInfo.voices && state.appInfo.voices.length
      ? state.appInfo.voices
      : ["M2", "F2"];
  els.voice.replaceChildren();
  for (const voice of voices) {
    const option = document.createElement("option");
    option.value = voice;
    option.textContent = VOICE_LABELS[voice] || voice;
    els.voice.appendChild(option);
  }
  if (!voices.includes(state.settings.voice)) {
    state.settings.voice = voices[0];
  }
  els.voice.value = state.settings.voice;
}

function populateAsrOptions() {
  if (!state.appInfo.packaged) {
    return;
  }
  for (const option of Array.from(els.asrModel.options)) {
    option.disabled = option.value !== DEFAULT_ASR_MODEL;
  }
  els.asrModel.value = DEFAULT_ASR_MODEL;
}

function addDeviceOption(select, value, label, title) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  option.title = title || label;
  select.appendChild(option);
}

function populateDeviceSelects() {
  const inputs = state.devices.filter((device) => device.max_input_channels > 0);
  const outputs = state.devices.filter((device) => device.max_output_channels > 0);

  const defaultInput = inputs.find((d) => d.index === state.deviceDefaults.input);
  const defaultOutput = outputs.find((d) => d.index === state.deviceDefaults.output);

  els.inputDevice.replaceChildren();
  addDeviceOption(
    els.inputDevice,
    "",
    defaultInput
      ? `System default — ${shortDeviceName(defaultInput.name)}`
      : "System default",
    defaultInput?.name || "System default"
  );
  for (const device of inputs) {
    if (device.name.toLowerCase().includes("monitor")) continue;
    addDeviceOption(
      els.inputDevice,
      device.name,
      `${shortDeviceName(device.name)} (${device.max_input_channels} ch)`,
      `${device.name} — ${device.hostapi}`
    );
  }
  if (
    state.settings.inputDevice &&
    !inputs.some((d) => d.name === state.settings.inputDevice)
  ) {
    addDeviceOption(els.inputDevice, state.settings.inputDevice, state.settings.inputDevice);
  }
  els.inputDevice.value = state.settings.inputDevice || "";

  els.outputDevice.replaceChildren();
  addDeviceOption(
    els.outputDevice,
    "",
    defaultOutput
      ? `System default — ${shortDeviceName(defaultOutput.name)}`
      : "System default",
    defaultOutput?.name || "System default"
  );
  for (const device of outputs) {
    addDeviceOption(
      els.outputDevice,
      device.name,
      `${shortDeviceName(device.name)} (${device.max_output_channels} ch)`,
      `${device.name} — ${device.hostapi}`
    );
  }
  if (
    state.settings.outputDevice &&
    !outputs.some((d) => d.name === state.settings.outputDevice)
  ) {
    addDeviceOption(els.outputDevice, state.settings.outputDevice, state.settings.outputDevice);
  }
  els.outputDevice.value = state.settings.outputDevice || "";
}

function updateDeviceInfo() {
  const parts = [];
  if (state.inputInfo) parts.push(`In: ${shortDeviceName(state.inputInfo)}`);
  if (state.outputInfo) parts.push(`Out: ${shortDeviceName(state.outputInfo)}`);
  els.deviceInfo.textContent = parts.length ? parts.join(" · ") : "Audio: system default";
  els.deviceInfo.title = [state.inputInfo, state.outputInfo].filter(Boolean).join("\n");
}

async function loadAppInfo() {
  try {
    const info = await window.danglish.appInfo();
    state.appInfo = info || state.appInfo;
    populateVoices();
    populateAsrOptions();
    if (state.appInfo.packaged) {
      els.modelInfo.textContent = "Bundled models · Danish Whisper · OPUS-MT · tiny-voca";
    }
  } catch (error) {
    els.modelInfo.textContent = "Backend offline";
  }
}

async function loadContexts() {
  try {
    const contexts = await window.danglish.listContexts();
    if (Array.isArray(contexts) && contexts.length) state.contexts = contexts;
  } catch (_) {
    state.contexts = FALLBACK_CONTEXTS.slice();
  }
  populateContexts();
}

async function loadDevices(showError = false) {
  try {
    const payload = await window.danglish.listDevices();
    state.devices = payload?.devices || [];
    state.deviceDefaults = {
      input: payload?.input_default ?? null,
      output: payload?.output_default ?? null,
    };
  } catch (error) {
    state.devices = [];
    if (showError) showToast(`Could not read audio devices: ${error.message}`);
  }
  populateDeviceSelects();
  updateDeviceInfo();
}

function setSettingsFromControls() {
  state.settings.context = els.contextSelect.value;
  state.settings.fullDuplex = els.fullDuplex.checked;
  state.settings.ttsEnabled = els.ttsEnabled.checked;
  state.settings.broadcastEnabled = els.broadcastEnabled.checked;
  saveSettings();
  updateContextUi();
  updateDuplexUi();
}

function addSessionDivider(label) {
  const empty = $("empty");
  if (empty) empty.remove();
  const divider = document.createElement("div");
  divider.className = "session-divider";
  divider.textContent = label;
  els.feed.appendChild(divider);
  els.feedStarted = true;
}

function clearFeed() {
  state.history = [];
  saveHistory();
  els.feed.innerHTML = state.emptyMarkup;
  els.feedStarted = false;
  els.lastEvent.textContent = "—";
  if (state.status === "listening") {
    $("empty")?.remove();
    showLiveHint();
  }
}

function appendUtterance(payload, { persist = true } = {}) {
  const segments = Array.isArray(payload.segments) ? payload.segments : [];
  const target = segments
    .map((segment) => segment.target)
    .filter(Boolean)
    .join(" ");
  if (!payload.transcript && !target) return;

  removeLiveHint();
  const empty = $("empty");
  if (empty) empty.remove();

  const nearBottom =
    els.feed.scrollHeight - els.feed.scrollTop - els.feed.clientHeight < 200;

  const article = document.createElement("article");
  article.className = "utterance";

  const head = document.createElement("div");
  head.className = "utterance-head";
  const who = document.createElement("span");
  who.textContent = "Danish → English";
  const time = document.createElement("span");
  time.textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  head.append(who, time);

  const source = document.createElement("p");
  source.className = "da";
  source.textContent = payload.transcript || "";

  const rule = document.createElement("div");
  rule.className = "divider";

  const translation = document.createElement("p");
  translation.className = "en";
  translation.textContent = target;

  article.append(head, source, rule, translation);

  const timings = document.createElement("div");
  timings.className = "timings";
  for (const [label, value] of [
    ["ASR", payload.asr_ms],
    ["MT", payload.mt_ms],
    ["TTS", payload.tts_ms],
    ["Total", payload.total_ms],
  ]) {
    if (typeof value === "number") {
      const chip = document.createElement("span");
      chip.textContent = `${label} ${formatMs(value)}`;
      timings.appendChild(chip);
    }
  }
  if (timings.childElementCount) article.appendChild(timings);

  els.feed.appendChild(article);
  els.feedStarted = true;

  if (persist) {
    state.history.push(payload);
    state.history = state.history.slice(-100);
    saveHistory();
  }

  if (nearBottom) {
    els.feed.scrollTop = els.feed.scrollHeight;
  }
  if (typeof payload.total_ms === "number") {
    els.lastEvent.textContent = `last turn ${formatMs(payload.total_ms)}`;
  }
  if (state.status === "listening") {
    els.statusDetail.textContent = "Speak Danish. Pause when you finish a thought.";
  }
}

function appendLog(stream, line) {
  if (!line) return;
  const prefix = stream === "stderr" ? "" : `[${stream}] `;
  els.logs.textContent += `${prefix}${line}\n`;
  const lines = els.logs.textContent.split("\n");
  if (lines.length > 500) {
    els.logs.textContent = lines.slice(-450).join("\n");
  }
  if (!els.logs.classList.contains("hidden")) {
    els.logs.scrollTop = els.logs.scrollHeight;
  }
}

function handleInfo(payload) {
  if (!payload || !payload.key) return;
  if (payload.key === "rss") {
    state.rss = payload.value;
    els.rssInfo.textContent = payload.value ? `RSS ${payload.value}` : "";
  } else if (payload.key === "input") {
    state.inputInfo = payload.value;
    updateDeviceInfo();
  } else if (payload.key === "output") {
    state.outputInfo = payload.value;
    updateDeviceInfo();
  } else if (payload.key === "broadcast") {
    state.broadcastUrl = payload.value || "";
    updateBroadcastUi();
  } else if (payload.key === "broadcastQr") {
    state.broadcastQr = payload.value || "";
    updateBroadcastUi();
  } else if (payload.key === "capture" || payload.key === "playback") {
    els.logs.title = `${payload.key}: ${payload.value}`;
  }
}

async function startListening() {
  if (!window.danglish || state.busy) return;
  setSettingsFromControls();
  if (state.settings.broadcastEnabled && !state.settings.ttsEnabled) {
    showToast("Broadcast needs Speak English enabled.");
    return;
  }
  state.broadcastUrl = "";
  state.broadcastQr = "";
  updateBroadcastUi();
  addSessionDivider(`Session ${new Date().toLocaleTimeString()}`);
  setStatus("starting", "Starting the local models…");
  try {
    const result = await window.danglish.start(state.settings);
    if (!result?.ok) {
      setStatus("error", result?.error || "Could not start the backend.");
    }
  } catch (error) {
    setStatus("error", error.message || String(error));
  }
}

async function stopListening() {
  if (!window.danglish) return;
  state.broadcastUrl = "";
  state.broadcastQr = "";
  updateBroadcastUi();
  setStatus("stopping", "Closing the audio streams…");
  try {
    await window.danglish.stop();
  } catch (error) {
    setStatus("error", error.message || String(error));
    return;
  }
  setTimeout(() => {
    if (state.status === "stopping") setStatus("idle");
  }, 3300);
}

async function toggleTalk() {
  if (state.status === "listening") {
    await stopListening();
  } else if (!state.busy) {
    await startListening();
  }
}

function openSettings() {
  els.inputChannel.value = String(state.settings.inputChannel ?? 0);
  els.outputChannel.value =
    state.settings.outputChannel === null || state.settings.outputChannel === undefined
      ? ""
      : String(state.settings.outputChannel);
  els.silenceMs.value = String(state.settings.silenceMs ?? 300);
  els.maxSegmentS.value = String(state.settings.maxSegmentS ?? 4.0);
  els.minEnergy.value = String(state.settings.minEnergy ?? 0.006);
  els.asrModel.value = state.settings.asrModel || DEFAULT_ASR_MODEL;
  populateVoices();
  els.voice.value = state.settings.voice;
  populateDeviceSelects();
  els.dialog.showModal();
}

function saveDialogSettings(event) {
  event.preventDefault();
  const silence = Number(els.silenceMs.value);
  const maxSegment = Number(els.maxSegmentS.value);
  const energy = Number(els.minEnergy.value);
  const inputChannel = Number(els.inputChannel.value);
  const outputChannel = els.outputChannel.value === "" ? null : Number(els.outputChannel.value);

  state.settings.inputDevice = els.inputDevice.value || "";
  state.settings.outputDevice = els.outputDevice.value || "";
  state.settings.inputChannel = Number.isFinite(inputChannel) && inputChannel >= 0 ? inputChannel : 0;
  state.settings.outputChannel =
    outputChannel !== null && Number.isFinite(outputChannel) && outputChannel >= 0
      ? outputChannel
      : null;
  state.settings.silenceMs = Number.isFinite(silence) ? Math.max(150, silence) : 300;
  state.settings.maxSegmentS =
    Number.isFinite(maxSegment) && maxSegment >= 1 ? Math.min(30, maxSegment) : 4.0;
  state.settings.minEnergy = Number.isFinite(energy) && energy > 0 ? energy : 0.006;
  state.settings.asrModel = els.asrModel.value;
  state.settings.voice = els.voice.value;

  saveSettings();
  els.dialog.close();
  if (state.running) {
    showToast("Settings saved. Stop and start listening to apply them.");
  }
}

function bindEvents() {
  els.talkBtn.addEventListener("click", toggleTalk);
  els.settingsBtn.addEventListener("click", openSettings);
  els.legalBtn.addEventListener("click", () => window.danglish?.openLegal());
  els.clearBtn.addEventListener("click", clearFeed);
  els.logsBtn.addEventListener("click", () => {
    const hidden = els.logs.classList.toggle("hidden");
    els.logsBtn.textContent = hidden ? "Logs" : "Hide logs";
    if (!hidden) els.logs.scrollTop = els.logs.scrollHeight;
  });

  els.contextSelect.addEventListener("change", () => {
    setSettingsFromControls();
    if (state.running) showToast("Context changed. Restart listening to apply it.");
  });
  els.fullDuplex.addEventListener("change", () => {
    setSettingsFromControls();
    if (state.running) showToast("Full-duplex changed. Restart listening to apply it.");
  });
  els.ttsEnabled.addEventListener("change", () => {
    if (!els.ttsEnabled.checked && els.broadcastEnabled.checked) {
      els.broadcastEnabled.checked = false;
      showToast("Broadcast turned off because Speak English is off.");
    }
    setSettingsFromControls();
    if (state.running) showToast("Voice output changed. Restart listening to apply it.");
  });
  els.broadcastEnabled.addEventListener("change", () => {
    if (els.broadcastEnabled.checked && !els.ttsEnabled.checked) {
      els.ttsEnabled.checked = true;
      showToast("Speak English enabled for broadcast.");
    }
    setSettingsFromControls();
    if (state.running) showToast("Broadcast changed. Restart listening to apply it.");
  });

  els.form.addEventListener("submit", saveDialogSettings);
  els.closeSettings.addEventListener("click", () => els.dialog.close());
  els.cancelSettings.addEventListener("click", () => els.dialog.close());
  els.refreshDevices.addEventListener("click", async () => {
    els.refreshDevices.disabled = true;
    await loadDevices(true);
    els.refreshDevices.disabled = false;
    showToast("Audio devices refreshed.");
  });
  els.dialog.addEventListener("click", (event) => {
    if (event.target === els.dialog) els.dialog.close();
  });

  if (window.danglish?.onStatus) {
    window.danglish.onStatus((payload) => {
      if (!payload) return;
      setStatus(payload.state || "idle", payload.message || "");
    });
    window.danglish.onEvent((payload) => {
      if (payload?.type === "utterance") {
        appendUtterance(payload);
      } else if (payload?.type === "speech" && state.status === "listening") {
        const duration = Number(payload.duration_s || 0).toFixed(1);
        els.statusDetail.textContent = `Translating ${duration}s of speech…`;
        els.lastEvent.textContent = `processing ${duration}s`;
      }
    });
    window.danglish.onLog((payload) => {
      if (payload) appendLog(payload.stream, payload.line);
    });
    window.danglish.onInfo(handleInfo);
  }
}

async function bootstrap() {
  state.emptyMarkup = els.feed.innerHTML;
  if (state.history.length) {
    $("empty")?.remove();
    for (const item of state.history) {
      appendUtterance(item, { persist: false });
    }
    els.feed.scrollTop = els.feed.scrollHeight;
  }
  els.fullDuplex.checked = state.settings.fullDuplex !== false;
  els.ttsEnabled.checked = state.settings.ttsEnabled !== false;
  els.broadcastEnabled.checked = state.settings.broadcastEnabled === true;
  els.contextSelect.value = state.settings.context;
  updateContextUi();
  updateDuplexUi();
  updateBroadcastUi();

  if (!window.danglish) {
    els.modelInfo.textContent = "Open in Electron";
    showToast("This interface needs the Electron shell. Run `npm start` in desktop/.");
    return;
  }

  await loadAppInfo();
  await loadContexts();
  await loadDevices();
  setStatus("idle");
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  bootstrap();
});
