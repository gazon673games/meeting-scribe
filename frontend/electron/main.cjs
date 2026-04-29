const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const { execFile, spawn } = require("node:child_process");
const { existsSync, readFileSync } = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const appRoot = app.isPackaged ? path.join(process.resourcesPath, "app") : path.resolve(__dirname, "..", "..");
const backendRoot = app.isPackaged ? path.join(process.resourcesPath, "backend") : appRoot;
const runtimeRoot = app.isPackaged ? path.dirname(process.execPath) : appRoot;
const windowIconPath = path.join(__dirname, "assets", process.platform === "win32" ? "icon.ico" : "icon.png");

if (process.platform === "win32") {
  app.setAppUserModelId("com.meetingscribe.app");
}

class PythonBackend {
  constructor({ root, backendRoot, runtimeRoot }) {
    this.root = root;
    this.backendRoot = backendRoot;
    this.runtimeRoot = runtimeRoot;
    this.process = null;
    this.nextId = 1;
    this.pending = new Map();
    this.stdoutBuffer = "";
    this.windows = new Set();
    this.eventBuffer = [];
    this.ready = false;
    this.lastError = "";
    this.activeDownloads = 0;
    this.activeModelDownloads = 0;
    this.activeSpeakerIdDownloads = 0;
    this.activeLlmDownloads = 0;
    this.stoppingProcess = null;
  }

  attachWindow(window) {
    if (!window) {
      return;
    }
    this.windows.add(window);
    window.on("closed", () => this.detachWindow(window));
  }

  detachWindow(window) {
    this.windows.delete(window);
  }

  recentEvents() {
    return this.eventBuffer.map((event) => ({ ...event }));
  }

  start() {
    if (this.process) {
      return;
    }
    const command = this.findBackendCommand();
    const child = spawn(command.executable, command.args, {
      cwd: command.cwd,
      env: {
        ...process.env,
        MEETING_SCRIBE_RUNTIME_ROOT: this.runtimeRoot,
        PYTHONIOENCODING: "utf-8",
        PYTHONUNBUFFERED: "1",
        PYTHONUTF8: "1"
      },
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"]
    });
    this.process = child;
    this.stoppingProcess = null;

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => this.handleStdout(chunk));
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      this.lastError = String(chunk);
      console.error("[python-backend]", String(chunk).trim());
      this.sendEvent({ type: "backend_stderr", text: String(chunk) });
    });
    child.on("exit", (code, signal) => {
      const expectedStop = this.stoppingProcess === child || app.isQuitting;
      this.ready = false;
      if (this.process === child) {
        this.process = null;
      }
      if (this.stoppingProcess === child) {
        this.stoppingProcess = null;
      }
      this.activeDownloads = 0;
      this.activeModelDownloads = 0;
      this.activeSpeakerIdDownloads = 0;
      this.activeLlmDownloads = 0;
      const message = `Python backend exited (${code ?? "null"}, ${signal ?? "null"})`;
      for (const pending of this.pending.values()) {
        if (expectedStop) {
          pending.resolve(stoppedResultForMethod(pending.method));
        } else {
          pending.reject(new Error(message));
        }
      }
      this.pending.clear();
      if (!expectedStop) {
        this.sendEvent({ type: "backend_exit", code, signal });
      }
    });
  }

  stop() {
    if (!this.process) {
      return;
    }
    const child = this.process;
    this.stoppingProcess = child;
    this.process = null;
    this.ready = false;
    for (const pending of this.pending.values()) {
      pending.resolve(stoppedResultForMethod(pending.method));
    }
    this.pending.clear();
    child.kill();
  }

  status() {
    return {
      ready: this.ready,
      running: Boolean(this.process),
      lastError: this.lastError
    };
  }

  request(method, params = {}) {
    if (app.isQuitting) {
      return Promise.resolve(stoppedResultForMethod(method));
    }
    this.start();
    if (!this.process || !this.process.stdin.writable) {
      return Promise.reject(new Error("Python backend is not writable"));
    }

    const id = String(this.nextId++);
    const payload = JSON.stringify({ id, method, params });
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, method });
      this.process.stdin.write(`${payload}\n`, "utf8", (error) => {
        if (error) {
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  handleStdout(chunk) {
    this.stdoutBuffer += chunk;
    while (true) {
      const newlineIndex = this.stdoutBuffer.indexOf("\n");
      if (newlineIndex < 0) {
        break;
      }
      const line = this.stdoutBuffer.slice(0, newlineIndex).trim();
      this.stdoutBuffer = this.stdoutBuffer.slice(newlineIndex + 1);
      if (line) {
        this.handleMessageLine(line);
      }
    }
  }

  handleMessageLine(line) {
    let message;
    try {
      message = JSON.parse(line);
    } catch (error) {
      this.lastError = `Invalid backend JSON: ${line}`;
      console.error(this.lastError, error);
      return;
    }

    if (message.event) {
      if (message.event.type === "backend_ready") {
        this.ready = true;
      }
      if (message.event.type === "model_download_updated") {
        this.activeModelDownloads = toNumber(message.event.activeDownloads);
        this.activeDownloads = this.activeModelDownloads + this.activeSpeakerIdDownloads + this.activeLlmDownloads;
      }
      if (message.event.type === "diarization_model_download_updated") {
        this.activeSpeakerIdDownloads = toNumber(message.event.activeDownloads);
        this.activeDownloads = this.activeModelDownloads + this.activeSpeakerIdDownloads + this.activeLlmDownloads;
      }
      if (message.event.type === "llm_model_download_updated") {
        this.activeLlmDownloads = toNumber(message.event.activeDownloads);
        this.activeDownloads = this.activeModelDownloads + this.activeSpeakerIdDownloads + this.activeLlmDownloads;
      }
      this.sendEvent(message.event);
      return;
    }

    const pending = this.pending.get(String(message.id));
    if (!pending) {
      return;
    }
    this.pending.delete(String(message.id));
    if (message.ok) {
      pending.resolve(message.result);
    } else {
      const backendError = new Error(message.error?.message || "Backend request failed");
      backendError.name = message.error?.type || "BackendError";
      pending.reject(backendError);
    }
  }

  sendEvent(event) {
    const record = { ts: Date.now() / 1000, ...event };
    this.eventBuffer.push(record);
    if (this.eventBuffer.length > 1200) {
      this.eventBuffer.splice(0, this.eventBuffer.length - 1200);
    }
    for (const window of [...this.windows]) {
      if (window.isDestroyed()) {
        this.windows.delete(window);
        continue;
      }
      window.webContents.send("backend:event", record);
    }
  }

  findPython() {
    const candidates =
      process.platform === "win32"
        ? [
            path.join(this.root, ".venv", "Scripts", "python.exe"),
            path.join(this.root, ".venv", "Scripts", "python3.exe"),
            "python"
          ]
        : [path.join(this.root, ".venv", "bin", "python"), "python3", "python"];
    return candidates.find((candidate) => candidate.includes(path.sep) ? existsSync(candidate) : true) || "python";
  }

  findBackendCommand() {
    if (app.isPackaged) {
      const executableName = process.platform === "win32" ? "meeting-scribe-backend.exe" : "meeting-scribe-backend";
      const executable = path.join(this.backendRoot, executableName);
      if (!existsSync(executable)) {
        throw new Error(`Packaged Python backend was not found: ${executable}`);
      }
      return { executable, args: [], cwd: this.runtimeRoot };
    }

    return {
      executable: this.findPython(),
      args: [path.join(this.root, "backend", "main_electron_backend.py")],
      cwd: this.root
    };
  }
}

const backend = new PythonBackend({ root: appRoot, backendRoot, runtimeRoot });
let gpuUsageCache = { ts: 0, gpus: [] };
let mainWindow = null;
let debugWindow = null;

function readSavedContentProtection() {
  const configPath = path.join(runtimeRoot, "config.json");
  if (!existsSync(configPath)) {
    return false;
  }
  try {
    const config = JSON.parse(readFileSync(configPath, "utf8"));
    return Boolean(config?.ui?.screen_capture_protection);
  } catch (error) {
    console.warn("[window] failed to read content protection setting", error);
    return false;
  }
}

function setWindowContentProtection(window, enabled) {
  if (!window || window.isDestroyed()) {
    return { enabled: false };
  }
  const nextEnabled = Boolean(enabled);
  window.setContentProtection(nextEnabled);
  return { enabled: nextEnabled };
}

function reloadWindow(window) {
  if (!window || window.isDestroyed()) {
    return { reloaded: false };
  }
  window.webContents.reloadIgnoringCache();
  return { reloaded: true };
}

function electronResourceUsage() {
  const metrics = app.getAppMetrics();
  const memoryBytes = metrics.reduce((sum, metric) => {
    const workingSetSize = Number(metric.memory?.workingSetSize || 0);
    return sum + Math.max(0, workingSetSize * 1024);
  }, 0);
  const cpuPct = metrics.reduce((sum, metric) => sum + Math.max(0, Number(metric.cpu?.percentCPUUsage || 0)), 0);
  return {
    cpuPct,
    memoryBytes: memoryBytes || process.memoryUsage().rss,
    processCount: metrics.length
  };
}

function nvidiaGpuUsage() {
  const now = Date.now();
  if (now - gpuUsageCache.ts < 2000) {
    return Promise.resolve(gpuUsageCache.gpus);
  }
  return new Promise((resolve) => {
    execFile(
      "nvidia-smi",
      [
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits"
      ],
      { timeout: 2000, windowsHide: true },
      (error, stdout) => {
        if (error) {
          gpuUsageCache = { ts: now, gpus: [] };
          resolve([]);
          return;
        }
        const gpus = String(stdout || "")
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line) => {
            const [name, total, used, gpuUtil, memoryUtil] = line.split(",").map((part) => part.trim());
            const memoryTotalMiB = toNumber(total);
            const memoryUsedMiB = toNumber(used);
            return {
              name: name || "NVIDIA GPU",
              memoryTotalMiB,
              memoryUsedMiB,
              memoryFreeMiB: Math.max(0, memoryTotalMiB - memoryUsedMiB),
              gpuUtilizationPct: toNumber(gpuUtil),
              memoryUtilizationPct: toNumber(memoryUtil)
            };
          });
        gpuUsageCache = { ts: now, gpus };
        resolve(gpus);
      }
    );
  });
}

function toNumber(value) {
  const parsed = Number(String(value || "").trim());
  return Number.isFinite(parsed) ? parsed : 0;
}

function stoppedResultForMethod(method) {
  if (method === "get_resource_usage") {
    return { pid: 0, cpuPct: 0, memoryBytes: 0 };
  }
  if (method === "list_devices") {
    return { loopback: [], input: [], errors: ["Python backend is stopping"] };
  }
  return null;
}

async function getResourceUsage() {
  const electron = electronResourceUsage();
  let backendUsage = { pid: backend.process?.pid || 0, cpuPct: 0, memoryBytes: 0 };
  try {
    backendUsage = (await backend.request("get_resource_usage", {})) || backendUsage;
  } catch (error) {
    backendUsage = { ...backendUsage, error: String(error?.message || error) };
  }
  const backendMemoryBytes = Math.max(0, Number(backendUsage.memoryBytes || 0));
  const electronMemoryBytes = Math.max(0, Number(electron.memoryBytes || 0));
  const backendCpuPct = Math.max(0, Number(backendUsage.cpuPct || 0));
  const electronCpuPct = Math.max(0, Number(electron.cpuPct || 0));
  return {
    ts: Date.now(),
    app: {
      cpuPct: electronCpuPct + backendCpuPct,
      memoryBytes: electronMemoryBytes + backendMemoryBytes,
      electronCpuPct,
      electronMemoryBytes,
      backendCpuPct,
      backendMemoryBytes,
      backendPid: Number(backendUsage.pid || backend.process?.pid || 0),
      electronProcessCount: Number(electron.processCount || 0)
    },
    system: {
      totalMemoryBytes: os.totalmem(),
      freeMemoryBytes: os.freemem(),
      logicalCores: os.cpus().length
    },
    gpus: await nvidiaGpuUsage()
  };
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1320,
    height: 880,
    minWidth: 980,
    minHeight: 640,
    backgroundColor: "#111111",
    title: "Meeting Scribe",
    icon: windowIconPath,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  mainWindow = window;
  setWindowContentProtection(window, readSavedContentProtection());
  backend.attachWindow(window);
  backend.start();

  window.on("close", (event) => {
    if (backend.activeDownloads <= 0) {
      return;
    }
    const choice = dialog.showMessageBoxSync(window, {
      type: "warning",
      buttons: ["Keep Downloading", "Close Anyway"],
      defaultId: 0,
      cancelId: 0,
      message: "Model download is still running",
      detail: "Closing the app now will stop the Python backend and interrupt the model download."
    });
    if (choice === 0) {
      event.preventDefault();
    }
  });

  loadRenderer(window);
  window.on("closed", () => {
    if (mainWindow === window) {
      mainWindow = null;
    }
    if (debugWindow && !debugWindow.isDestroyed()) {
      debugWindow.destroy();
    }
  });
}

function showDebugConsole() {
  if (!debugWindow || debugWindow.isDestroyed()) {
    debugWindow = createDebugWindow();
  }
  if (debugWindow.isMinimized()) {
    debugWindow.restore();
  }
  debugWindow.showInactive();
  return { shown: true };
}

function createDebugWindow() {
  const window = new BrowserWindow({
    width: 980,
    height: 720,
    minWidth: 720,
    minHeight: 520,
    backgroundColor: "#0a0a0a",
    title: "Meeting Scribe Diagnostics",
    icon: windowIconPath,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  backend.attachWindow(window);
  window.setAlwaysOnTop(false);
  window.on("close", (event) => {
    if (app.isQuitting) {
      return;
    }
    event.preventDefault();
    window.hide();
  });
  window.on("closed", () => {
    if (debugWindow === window) {
      debugWindow = null;
    }
  });
  loadRenderer(window, { debugConsole: "1" });
  return window;
}

function loadRenderer(window, query = {}) {
  const rendererUrl = process.env.ELECTRON_RENDERER_URL || "http://127.0.0.1:5173";
  if (app.isPackaged && !process.env.ELECTRON_RENDERER_URL) {
    window.loadFile(path.join(appRoot, "build", "electron-renderer", "index.html"), { query });
    return;
  }
  const url = new URL(rendererUrl);
  for (const [key, value] of Object.entries(query)) {
    url.searchParams.set(key, String(value));
  }
  window.loadURL(url.toString());
}

app.whenReady().then(() => {
  ipcMain.handle("backend:request", (_event, method, params) => backend.request(method, params));
  ipcMain.handle("backend:status", () => backend.status());
  ipcMain.handle("backend:recent-events", () => backend.recentEvents());
  ipcMain.handle("debug:show", () => showDebugConsole());
  ipcMain.handle("window:set-content-protection", (event, enabled) => {
    return setWindowContentProtection(BrowserWindow.fromWebContents(event.sender), enabled);
  });
  ipcMain.handle("window:reload", (event) => {
    return reloadWindow(BrowserWindow.fromWebContents(event.sender));
  });
  ipcMain.handle("app:get-resource-usage", () => getResourceUsage());
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("before-quit", () => {
  app.isQuitting = true;
  backend.stop();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
