"use strict";

const { app, BrowserWindow, ipcMain, Menu, shell, systemPreferences } = require("electron");
const fs = require("fs");
const path = require("path");
const readline = require("readline");
const { spawn } = require("child_process");

const isWindows = process.platform === "win32";
const isMac = process.platform === "darwin";
const repoRoot = path.resolve(__dirname, "..");
const DEFAULT_ASR_MODEL = "lucasw89/whisper-small-danish-ct2-int8";

let mainWindow = null;
let backendProcess = null;
let backendState = "idle";
let backendStateMessage = "Ready";
let backendLogTail = [];
let stopRequested = false;
let backendMuted = false;

function send(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

function sendStatus(state, message) {
  backendState = state;
  backendStateMessage = message;
  send("backend:status", { state, message });
}

function sendLog(stream, line) {
  backendLogTail.push(line);
  if (backendLogTail.length > 80) backendLogTail.shift();
  send("backend:log", { stream, line });
}

function backendCommand() {
  if (process.env.DANGLISH_BACKEND) {
    return { command: process.env.DANGLISH_BACKEND, prefix: [] };
  }

  if (app.isPackaged) {
    const base = path.join(process.resourcesPath, "backend", "danglish-backend");
    const nested = path.join(base, isWindows ? "danglish-backend.exe" : "danglish-backend");
    const direct = isWindows ? `${base}.exe` : base;
    if (fs.existsSync(nested)) return { command: nested, prefix: [] };
    return { command: direct, prefix: [] };
  }

  const python =
    process.env.DANGLISH_PYTHON ||
    (isWindows
      ? path.join(repoRoot, ".venv", "Scripts", "python.exe")
      : path.join(repoRoot, ".venv", "bin", "python"));
  return { command: python, prefix: ["-m", "danglish"] };
}

function modelsRoot() {
  return app.isPackaged
    ? path.join(process.resourcesPath, "models")
    : path.join(repoRoot, "desktop", "resources", "models");
}

function modelPaths(settings) {
  const bundled = modelsRoot();
  const bundledSermon = path.join(bundled, "sermon-opus-mt-int8");
  const bundledGeneral = path.join(bundled, "opus-mt-da-en-int8");
  const useBundled = fs.existsSync(bundledSermon) || fs.existsSync(bundledGeneral);
  const devModels = path.join(repoRoot, "models");

  const mt = fs.existsSync(bundledSermon)
    ? bundledSermon
    : fs.existsSync(bundledGeneral)
      ? bundledGeneral
      : path.join(devModels, "opus-mt-da-en-int8");

  const bundledAsr = path.join(bundled, "whisper-small-danish");
  const legacyAsr = path.join(bundled, "whisper-small");
  const asr = useBundled
    ? fs.existsSync(bundledAsr)
      ? bundledAsr
      : legacyAsr
    : null;

  return { bundled, useBundled, mt, asr, tinyVoca: path.join(bundled, "tiny-voca") };
}

function buildBackendArgs(settings) {
  const models = modelPaths(settings);
  const args = ["talk", "--json"];

  if (fs.existsSync(models.mt)) args.push("--model-path", models.mt);
  if (models.asr) args.push("--asr-model", models.asr);
  else args.push("--asr-model", settings.asrModel || DEFAULT_ASR_MODEL);

  args.push("--tts-backend", settings.ttsBackend || "tiny-voca");
  args.push("--tts-voice", settings.voice || "M2");
  if (settings.broadcastEnabled) args.push("--broadcast");

  args.push("--context", "sermon");

  args.push("--full-duplex");
  if (settings.inputDevice) args.push("--input-device", String(settings.inputDevice));
  if (settings.outputDevice) {
    args.push("--output-device", String(settings.outputDevice));
  }
  if (settings.inputChannel !== undefined && settings.inputChannel !== null) {
    args.push("--input-channel", String(settings.inputChannel));
  }
  if (
    settings.outputChannel !== undefined &&
    settings.outputChannel !== null &&
    settings.outputChannel !== ""
  ) {
    args.push("--output-channel", String(settings.outputChannel));
  }
  if (settings.silenceMs !== undefined) {
    args.push("--silence-ms", String(settings.silenceMs));
  }
  if (settings.maxSegmentS !== undefined) {
    args.push("--max-segment-s", String(settings.maxSegmentS));
  }
  if (settings.minEnergy !== undefined) {
    args.push("--min-energy", String(settings.minEnergy));
  }

  return args;
}

function handleBackendLine(line) {
  const trimmed = line.trim();
  if (!trimmed) return;

  if (trimmed.startsWith("loading Whisper")) {
    sendStatus("starting", "Loading speech model…");
  } else if (trimmed.startsWith("loading English voice")) {
    sendStatus("starting", "Loading English voice…");
  } else if (trimmed.startsWith("ready —")) {
    const match = trimmed.match(/RSS ([0-9,.]+) MB/);
    send("backend:info", {
      key: "rss",
      value: match ? `${match[1]} MB` : "",
    });
    sendStatus("starting", "Models ready");
  } else if (trimmed.startsWith("listening —")) {
    sendStatus("listening", "Listening");
  } else if (trimmed.startsWith("broadcast:")) {
    send("backend:info", {
      key: "broadcast",
      value: trimmed.replace(/^broadcast:\s*/, ""),
    });
  } else if (trimmed.startsWith("operator QR:")) {
    send("backend:info", {
      key: "broadcastQr",
      value: trimmed.replace(/^operator QR:\s*/, ""),
    });
  } else if (trimmed.startsWith("audio input:")) {
    send("backend:info", { key: "input", value: trimmed.replace(/^audio input:\s*/, "") });
  } else if (trimmed.startsWith("audio output:")) {
    send("backend:info", { key: "output", value: trimmed.replace(/^audio output:\s*/, "") });
  } else if (trimmed.startsWith("capture:")) {
    send("backend:info", { key: "capture", value: trimmed.replace(/^capture:\s*/, "") });
  } else if (trimmed.startsWith("playback:")) {
    send("backend:info", { key: "playback", value: trimmed.replace(/^playback:\s*/, "") });
  } else if (trimmed.startsWith("danglish: error:")) {
    sendStatus("error", trimmed.replace(/^danglish: error:\s*/, ""));
  }
}

async function stopBackend() {
  if (!backendProcess) {
    sendStatus("idle", "Stopped");
    return;
  }

  stopRequested = true;
  const child = backendProcess;
  sendStatus("stopping", "Stopping…");

  if (isWindows) {
    try {
      child.kill();
    } catch (_) {
      /* already gone */
    }
    setTimeout(() => {
      try {
        spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], {
          windowsHide: true,
        });
      } catch (_) {
        /* ignore */
      }
    }, 900);
  } else {
    try {
      child.kill("SIGINT");
    } catch (_) {
      /* already gone */
    }
    setTimeout(() => {
      try {
        child.kill("SIGKILL");
      } catch (_) {
        /* already gone */
      }
    }, 2500);
  }

  await new Promise((resolve) => {
    if (child.exitCode !== null) {
      resolve();
      return;
    }
    const timer = setTimeout(resolve, 3200);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });

  if (backendProcess === child) backendProcess = null;
  if (stopRequested) sendStatus("idle", "Stopped");
  stopRequested = false;
}

function startBackend(settings) {
  return new Promise((resolve) => {
    if (backendProcess) {
      resolve({ ok: false, error: "The backend is already running." });
      return;
    }

    const { command, prefix } = backendCommand();
    if (!fs.existsSync(command)) {
      const message = app.isPackaged
        ? `Bundled backend not found at ${command}.`
        : `Python environment not found at ${command}. Run "uv sync --extra speech" or set DANGLISH_PYTHON.`;
      sendLog("app", message);
      sendStatus("error", message);
      resolve({ ok: false, error: message });
      return;
    }

    const args = [...prefix, ...buildBackendArgs(settings)];
    const models = modelPaths(settings);
    const tinyVocaDir = fs.existsSync(path.join(models.tinyVoca, "onnx", "model.onnx"))
      ? models.tinyVoca
      : null;
    sendLog("app", `$ ${command} ${args.join(" ")}`);
    sendStatus("starting", "Starting…");

    let child;
    try {
      child = spawn(command, args, {
        cwd: app.isPackaged ? process.resourcesPath : repoRoot,
        env: {
          ...process.env,
          ...(tinyVocaDir ? { DANGLISH_TINY_VOCA_DIR: tinyVocaDir } : {}),
          DANGLISH_CONTROL_STDIN: "1",
          PYTHONUNBUFFERED: "1",
          PYTHONIOENCODING: "utf-8",
          PYTHONUTF8: "1",
        },
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true,
      });
    } catch (error) {
      const message = `Could not start backend: ${error.message}`;
      sendStatus("error", message);
      resolve({ ok: false, error: message });
      return;
    }

    backendProcess = child;
    backendLogTail = [];
    stopRequested = false;
    backendMuted = false;

    const stdout = readline.createInterface({ input: child.stdout });
    stdout.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      try {
        const event = JSON.parse(trimmed);
        send("backend:event", event);
      } catch (_) {
        sendLog("stdout", trimmed);
      }
    });

    const stderr = readline.createInterface({ input: child.stderr });
    stderr.on("line", (line) => {
      handleBackendLine(line);
      sendLog("stderr", line);
    });

    child.on("error", (error) => {
      sendLog("app", `backend error: ${error.message}`);
      sendStatus("error", error.message);
    });

    child.on("exit", (code, signal) => {
      if (backendProcess === child) backendProcess = null;
      const wasStop = stopRequested;
      stopRequested = false;

      if (wasStop || code === 0) {
        sendStatus("idle", "Stopped");
      } else {
        const tail = backendLogTail.slice(-3).join("\n");
        sendStatus(
          "error",
          `Backend exited (${code ?? signal ?? "unknown"}).${tail ? `\n${tail}` : ""}`
        );
      }
    });

    resolve({ ok: true });
  });
}

function runBackendJson(args, timeoutMs = 20000) {
  return new Promise((resolve, reject) => {
    const { command, prefix } = backendCommand();
    if (!fs.existsSync(command)) {
      reject(
        new Error(
          app.isPackaged
            ? `Bundled backend not found at ${command}.`
            : `Python environment not found at ${command}. Run "uv sync --extra speech".`
        )
      );
      return;
    }

    let stdout = "";
    let stderr = "";
    let settled = false;

    const child = spawn(command, [...prefix, ...args], {
      cwd: app.isPackaged ? process.resourcesPath : repoRoot,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
        PYTHONUTF8: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try {
        child.kill();
      } catch (_) {
        /* ignore */
      }
      reject(new Error("Backend command timed out."));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(error);
    });
    child.on("exit", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(stderr.trim() || `Backend exited with code ${code}.`));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim() || "null"));
      } catch (error) {
        reject(new Error(`Invalid backend JSON: ${error.message}`));
      }
    });
  });
}

function legalPath() {
  if (app.isPackaged) return path.join(process.resourcesPath, "legal", "NOTICE");
  return path.join(repoRoot, "NOTICE");
}

function bundledVoiceNames() {
  // Both voices ship in the same tiny-voca model; no per-voice files.
  return ["M2", "F2"];
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 980,
    minHeight: 660,
    backgroundColor: "#0c0f14",
    title: "Danglish",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));

  mainWindow.webContents.setWindowOpenHandler(() => ({ action: "deny" }));

  if (process.env.DANGLISH_DEVTOOLS === "1" && !app.isPackaged) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }

  if (process.env.DANGLISH_AUTOSTART === "1") {
    mainWindow.webContents.once("did-finish-load", () => {
      setTimeout(() => {
        mainWindow.webContents
          .executeJavaScript("document.getElementById('talkBtn').click()")
          .catch(() => {});
      }, 1200);
    });
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
    stopBackend();
  });
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!mainWindow) {
      createWindow();
    } else {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    }
  });

  app.whenReady().then(() => {
    if (!isMac) Menu.setApplicationMenu(null);

    if (isMac) {
      // The Python sidecar owns audio capture. Asking here associates the app
      // identity with the microphone TCC prompt before the first turn.
      systemPreferences.askForMediaAccess("microphone").catch(() => {});
    }

    ipcMain.handle("backend:start", async (_event, settings) => startBackend(settings || {}));
    ipcMain.handle("backend:stop", async () => {
      await stopBackend();
      return { ok: true };
    });
    ipcMain.handle("backend:setMuted", async (_event, muted) => {
      backendMuted = Boolean(muted);
      if (backendProcess && backendProcess.stdin && !backendProcess.stdin.destroyed) {
        backendProcess.stdin.write(backendMuted ? "mute\n" : "unmute\n");
      }
      return { ok: true, muted: backendMuted };
    });
    ipcMain.handle("backend:state", async () => ({
      state: backendState,
      message: backendStateMessage,
    }));

    ipcMain.handle("audio:list", async () => runBackendJson(["devices", "--json"]));
    ipcMain.handle("contexts:list", async () => runBackendJson(["contexts", "--json"]));
    ipcMain.handle("app:info", async () => ({
      packaged: app.isPackaged,
      voices: bundledVoiceNames(),
    }));
    ipcMain.handle("legal:open", async () => shell.openPath(legalPath()));

    createWindow();

    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on("window-all-closed", async () => {
    await stopBackend();
    if (!isMac) app.quit();
  });

  let quitting = false;
  app.on("before-quit", async (event) => {
    if (quitting || !backendProcess) return;
    event.preventDefault();
    quitting = true;
    await stopBackend();
    app.quit();
  });
}
