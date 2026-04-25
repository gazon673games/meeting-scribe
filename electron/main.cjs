const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("node:child_process");
const { existsSync } = require("node:fs");
const path = require("node:path");

const projectRoot = path.resolve(__dirname, "..");

class PythonBackend {
  constructor(root) {
    this.root = root;
    this.process = null;
    this.nextId = 1;
    this.pending = new Map();
    this.stdoutBuffer = "";
    this.window = null;
    this.ready = false;
    this.lastError = "";
  }

  attachWindow(window) {
    this.window = window;
  }

  start() {
    if (this.process) {
      return;
    }
    const python = this.findPython();
    const entrypoint = path.join(this.root, "main_electron_backend.py");
    this.process = spawn(python, [entrypoint], {
      cwd: this.root,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1"
      },
      stdio: ["pipe", "pipe", "pipe"]
    });

    this.process.stdout.setEncoding("utf8");
    this.process.stdout.on("data", (chunk) => this.handleStdout(chunk));
    this.process.stderr.setEncoding("utf8");
    this.process.stderr.on("data", (chunk) => {
      this.lastError = String(chunk);
      console.error("[python-backend]", String(chunk).trim());
      this.sendEvent({ type: "backend_stderr", text: String(chunk) });
    });
    this.process.on("exit", (code, signal) => {
      this.ready = false;
      this.process = null;
      const message = `Python backend exited (${code ?? "null"}, ${signal ?? "null"})`;
      for (const { reject } of this.pending.values()) {
        reject(new Error(message));
      }
      this.pending.clear();
      this.sendEvent({ type: "backend_exit", code, signal });
    });
  }

  stop() {
    if (!this.process) {
      return;
    }
    this.process.kill();
    this.process = null;
  }

  status() {
    return {
      ready: this.ready,
      running: Boolean(this.process),
      lastError: this.lastError
    };
  }

  request(method, params = {}) {
    this.start();
    if (!this.process || !this.process.stdin.writable) {
      return Promise.reject(new Error("Python backend is not writable"));
    }

    const id = String(this.nextId++);
    const payload = JSON.stringify({ id, method, params });
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
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
    if (this.window && !this.window.isDestroyed()) {
      this.window.webContents.send("backend:event", event);
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
}

const backend = new PythonBackend(projectRoot);

function createWindow() {
  const window = new BrowserWindow({
    width: 1320,
    height: 880,
    minWidth: 980,
    minHeight: 640,
    backgroundColor: "#111111",
    title: "Meeting Scribe",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  backend.attachWindow(window);
  backend.start();

  const rendererUrl = process.env.ELECTRON_RENDERER_URL || "http://127.0.0.1:5173";
  if (app.isPackaged && !process.env.ELECTRON_RENDERER_URL) {
    window.loadFile(path.join(projectRoot, "build", "electron-renderer", "index.html"));
  } else {
    window.loadURL(rendererUrl);
  }
}

app.whenReady().then(() => {
  ipcMain.handle("backend:request", (_event, method, params) => backend.request(method, params));
  ipcMain.handle("backend:status", () => backend.status());
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("before-quit", () => {
  backend.stop();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

