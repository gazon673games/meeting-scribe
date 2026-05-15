import { createRequire } from "node:module";

import { describe, expect, test, vi } from "vitest";

const require = createRequire(import.meta.url);
const { registerIpcHandlers } = require("../ipc-handlers.cjs");

function registeredHandlers() {
  const handlers = new Map();
  const ipcMain = {
    handle: vi.fn((channel, handler) => handlers.set(channel, handler))
  };
  return { handlers, ipcMain };
}

describe("main-process IPC handlers", () => {
  test("registers backend and app handlers", async () => {
    const { handlers, ipcMain } = registeredHandlers();
    const backend = {
      recentEvents: vi.fn(() => [{ type: "backend_ready" }]),
      request: vi.fn(async (method, params) => ({ method, params })),
      status: vi.fn(() => ({ ready: true }))
    };
    const window = {};
    const BrowserWindow = { fromWebContents: vi.fn(() => window) };
    const getResourceUsage = vi.fn(async () => ({ app: {} }));
    const reloadWindow = vi.fn(() => ({ reloaded: true }));
    const setWindowContentProtection = vi.fn(() => ({ enabled: true }));
    const showDebugConsole = vi.fn(() => ({ shown: true }));

    registerIpcHandlers({
      BrowserWindow,
      backend,
      getResourceUsage,
      ipcMain,
      reloadWindow,
      setWindowContentProtection,
      showDebugConsole
    });

    expect(ipcMain.handle).toHaveBeenCalledTimes(7);
    await expect(handlers.get("backend:request")({}, "get_state", { a: 1 })).resolves.toEqual({
      method: "get_state",
      params: { a: 1 }
    });
    expect(handlers.get("backend:status")()).toEqual({ ready: true });
    expect(handlers.get("backend:recent-events")()).toEqual([{ type: "backend_ready" }]);
    expect(handlers.get("debug:show")()).toEqual({ shown: true });
    await expect(handlers.get("app:get-resource-usage")()).resolves.toEqual({ app: {} });

    const event = { sender: {} };
    expect(handlers.get("window:set-content-protection")(event, true)).toEqual({ enabled: true });
    expect(handlers.get("window:reload")(event)).toEqual({ reloaded: true });
    expect(setWindowContentProtection).toHaveBeenCalledWith(window, true);
    expect(reloadWindow).toHaveBeenCalledWith(window);
  });
});
