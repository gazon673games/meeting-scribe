import { createRequire } from "node:module";

import { afterEach, describe, expect, test, vi } from "vitest";

const require = createRequire(import.meta.url);
const mainPath = require.resolve("../main.cjs");

function makeStream() {
  const handlers = new Map();
  return {
    setEncoding: vi.fn(),
    on: vi.fn((event, handler) => handlers.set(event, handler)),
    emit: (event, ...args) => handlers.get(event)?.(...args)
  };
}

function makeChild() {
  const handlers = new Map();
  return {
    pid: 4321,
    stdout: makeStream(),
    stderr: makeStream(),
    stdin: {
      writable: true,
      write: vi.fn((payload, encoding, callback) => callback?.())
    },
    kill: vi.fn(),
    on: vi.fn((event, handler) => handlers.set(event, handler)),
    emit: (event, ...args) => handlers.get(event)?.(...args)
  };
}

function makeElectron() {
  const appEvents = new Map();
  const windows = [];

  class FakeBrowserWindow {
    constructor(options = {}) {
      this.options = options;
      this.destroyed = false;
      this.events = new Map();
      this.webContents = {
        reloadIgnoringCache: vi.fn(),
        send: vi.fn()
      };
      this.hide = vi.fn();
      this.isMinimized = vi.fn(() => false);
      this.loadFile = vi.fn();
      this.loadURL = vi.fn();
      this.restore = vi.fn();
      this.setAlwaysOnTop = vi.fn();
      this.setContentProtection = vi.fn();
      this.showInactive = vi.fn();
      windows.push(this);
    }

    on(event, handler) {
      this.events.set(event, handler);
    }

    emit(event, ...args) {
      return this.events.get(event)?.(...args);
    }

    isDestroyed() {
      return this.destroyed;
    }

    destroy() {
      this.destroyed = true;
      this.emit("closed");
    }
  }

  FakeBrowserWindow.fromWebContents = vi.fn(() => windows[0] ?? null);
  FakeBrowserWindow.getAllWindows = vi.fn(() => windows.filter((window) => !window.destroyed));

  const app = {
    isPackaged: false,
    isQuitting: false,
    getAppMetrics: vi.fn(() => [
      { cpu: { percentCPUUsage: 3 }, memory: { workingSetSize: 2 } },
      { cpu: { percentCPUUsage: 4 }, memory: { workingSetSize: 5 } }
    ]),
    on: vi.fn((event, handler) => appEvents.set(event, handler)),
    quit: vi.fn(),
    setAppUserModelId: vi.fn(),
    whenReady: vi.fn(() => Promise.resolve())
  };

  return {
    appEvents,
    electron: {
      app,
      BrowserWindow: FakeBrowserWindow,
      dialog: { showMessageBoxSync: vi.fn(() => 0) },
      ipcMain: { handle: vi.fn() }
    },
    windows
  };
}

function unloadMain() {
  delete require.cache[mainPath];
  delete global.__MEETING_SCRIBE_MAIN_DEPS__;
  delete global.__MEETING_SCRIBE_MAIN_DISABLE_BOOT__;
}

function loadMain(overrides = {}) {
  unloadMain();
  const child = overrides.child || makeChild();
  const electronData = overrides.electronData || makeElectron();
  const deps = {
    childProcess: {
      execFile: vi.fn(),
      spawn: vi.fn(() => child)
    },
    electron: electronData.electron,
    fs: {
      existsSync: vi.fn(() => false),
      readFileSync: vi.fn()
    },
    ipcHandlers: {
      registerIpcHandlers: vi.fn()
    },
    os: {
      cpus: vi.fn(() => [{}, {}, {}, {}]),
      freemem: vi.fn(() => 4000),
      totalmem: vi.fn(() => 8000)
    },
    rendererUrl: {
      rendererUrlFromEnv: vi.fn(() => "http://127.0.0.1:5173")
    },
    ...overrides.deps
  };

  global.__MEETING_SCRIBE_MAIN_DISABLE_BOOT__ = true;
  global.__MEETING_SCRIBE_MAIN_DEPS__ = deps;

  return {
    child,
    deps,
    electronData,
    main: require(mainPath)
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  unloadMain();
});

describe("electron main process", () => {
  test("starts the Python backend, resolves JSONL replies, and forwards events to windows", async () => {
    const { child, deps, electronData, main } = loadMain();
    const backend = new main.PythonBackend({
      backendRoot: "C:/app",
      root: "C:/app",
      runtimeRoot: "C:/runtime"
    });
    const window = new electronData.electron.BrowserWindow();

    backend.attachWindow(window);
    const response = backend.request("ping", { fresh: true });

    expect(deps.childProcess.spawn).toHaveBeenCalledWith(
      expect.stringMatching(/^python3?$/),
      expect.arrayContaining([expect.stringContaining("main_electron_backend.py")]),
      expect.objectContaining({
        cwd: "C:/app",
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true
      })
    );
    expect(JSON.parse(child.stdin.write.mock.calls[0][0])).toEqual({
      id: "1",
      method: "ping",
      params: { fresh: true }
    });

    child.stdout.emit(
      "data",
      [
        JSON.stringify({ event: { type: "backend_ready" } }),
        JSON.stringify({ event: { type: "model_download_updated", activeDownloads: "2" } }),
        JSON.stringify({ id: "1", ok: true, result: { pong: true } })
      ].join("\n") + "\n"
    );

    await expect(response).resolves.toEqual({ pong: true });
    expect(backend.status()).toEqual({ ready: true, running: true, lastError: "" });
    expect(backend.activeDownloads).toBe(2);
    expect(window.webContents.send).toHaveBeenCalledWith(
      "backend:event",
      expect.objectContaining({ type: "backend_ready" })
    );

    const recentEvents = backend.recentEvents();
    recentEvents[0].type = "mutated";
    expect(backend.recentEvents()[0].type).toBe("backend_ready");
  });

  test("rejects backend errors and resolves in-flight requests during an intentional stop", async () => {
    const { child, main } = loadMain();
    const backend = new main.PythonBackend({
      backendRoot: "C:/app",
      root: "C:/app",
      runtimeRoot: "C:/runtime"
    });

    const failed = backend.request("fail");
    child.stdout.emit(
      "data",
      JSON.stringify({
        id: "1",
        ok: false,
        error: { message: "decode failed", type: "DecodeError" }
      }) + "\n"
    );
    await expect(failed).rejects.toMatchObject({ message: "decode failed", name: "DecodeError" });

    const stopping = backend.request("get_resource_usage");
    backend.stop();

    await expect(stopping).resolves.toEqual({ pid: 0, cpuPct: 0, memoryBytes: 0 });
    expect(child.kill).toHaveBeenCalled();
    expect(backend.status()).toEqual({ ready: false, running: false, lastError: "" });
  });

  test("exposes window and resource helpers without requiring a real Electron app", async () => {
    const { deps, electronData, main } = loadMain();
    deps.fs.existsSync.mockReturnValue(true);
    deps.fs.readFileSync.mockReturnValue(JSON.stringify({ ui: { screen_capture_protection: true } }));
    deps.childProcess.execFile.mockImplementationOnce((command, args, options, callback) => {
      callback(null, "RTX 4070, 12000, 3000, 65, 20\n");
    });
    const window = new electronData.electron.BrowserWindow();

    expect(main.readSavedContentProtection()).toBe(true);
    expect(main.setWindowContentProtection(window, 1)).toEqual({ enabled: true });
    expect(main.reloadWindow(window)).toEqual({ reloaded: true });
    expect(window.setContentProtection).toHaveBeenCalledWith(true);
    expect(window.webContents.reloadIgnoringCache).toHaveBeenCalled();
    expect(main.electronResourceUsage()).toEqual({
      cpuPct: 7,
      memoryBytes: 7168,
      processCount: 2
    });
    await expect(main.nvidiaGpuUsage()).resolves.toEqual([
      {
        gpuUtilizationPct: 65,
        memoryFreeMiB: 9000,
        memoryTotalMiB: 12000,
        memoryUsedMiB: 3000,
        memoryUtilizationPct: 20,
        name: "RTX 4070"
      }
    ]);
    expect(main.stoppedResultForMethod("list_devices")).toEqual({
      errors: ["Python backend is stopping"],
      input: [],
      loopback: []
    });
    expect(main.toNumber("not a number")).toBe(0);
  });

  test("creates app windows, debug console, renderer URLs, and lifecycle handlers", async () => {
    const { deps, electronData, main } = loadMain();

    main.createWindow();
    const mainWindow = electronData.windows[0];

    expect(mainWindow.options.title).toBe("Meeting Scribe");
    expect(mainWindow.options.webPreferences.preload).toContain("preload.cjs");
    expect(mainWindow.loadURL).toHaveBeenCalledWith("http://127.0.0.1:5173/");

    main.backend.activeDownloads = 1;
    const blockedClose = { preventDefault: vi.fn() };
    mainWindow.emit("close", blockedClose);
    expect(blockedClose.preventDefault).toHaveBeenCalled();

    expect(main.showDebugConsole()).toEqual({ shown: true });
    const debugWindow = electronData.windows[1];
    expect(debugWindow.loadURL.mock.calls[0][0]).toBe("http://127.0.0.1:5173/?debugConsole=1");
    expect(debugWindow.showInactive).toHaveBeenCalled();

    const debugClose = { preventDefault: vi.fn() };
    debugWindow.emit("close", debugClose);
    expect(debugClose.preventDefault).toHaveBeenCalled();
    expect(debugWindow.hide).toHaveBeenCalled();

    main.boot();
    await Promise.resolve();

    expect(deps.ipcHandlers.registerIpcHandlers).toHaveBeenCalledWith(
      expect.objectContaining({
        backend: main.backend,
        ipcMain: electronData.electron.ipcMain,
        showDebugConsole: main.showDebugConsole
      })
    );
    electronData.appEvents.get("before-quit")();
    expect(electronData.electron.app.isQuitting).toBe(true);

    electronData.appEvents.get("window-all-closed")();
    if (process.platform === "darwin") {
      expect(electronData.electron.app.quit).not.toHaveBeenCalled();
    } else {
      expect(electronData.electron.app.quit).toHaveBeenCalled();
    }
  });
});
