import { createRequire } from "node:module";

import { describe, expect, test, vi } from "vitest";

const require = createRequire(import.meta.url);
const { createMeetingScribeApi } = require("../preload-api.cjs");

function fakeIpcRenderer() {
  return {
    invoke: vi.fn(async (channel, ...args) => ({ channel, args })),
    on: vi.fn(),
    removeListener: vi.fn()
  };
}

describe("preload API", () => {
  test("maps public calls to stable IPC channels", async () => {
    const ipcRenderer = fakeIpcRenderer();
    const api = createMeetingScribeApi(ipcRenderer);

    await api.request("get_state", { fresh: true });
    await api.status();
    await api.resourceUsage();
    await api.setContentProtection(1);

    expect(ipcRenderer.invoke).toHaveBeenNthCalledWith(1, "backend:request", "get_state", { fresh: true });
    expect(ipcRenderer.invoke).toHaveBeenNthCalledWith(2, "backend:status");
    expect(ipcRenderer.invoke).toHaveBeenNthCalledWith(3, "app:get-resource-usage");
    expect(ipcRenderer.invoke).toHaveBeenNthCalledWith(4, "window:set-content-protection", true);
  });

  test("subscribes and unsubscribes backend events", () => {
    const ipcRenderer = fakeIpcRenderer();
    const api = createMeetingScribeApi(ipcRenderer);
    const callback = vi.fn();

    const unsubscribe = api.onBackendEvent(callback);
    const listener = ipcRenderer.on.mock.calls[0][1];
    listener({}, { type: "backend_ready" });
    unsubscribe();

    expect(ipcRenderer.on).toHaveBeenCalledWith("backend:event", listener);
    expect(callback).toHaveBeenCalledWith({ type: "backend_ready" });
    expect(ipcRenderer.removeListener).toHaveBeenCalledWith("backend:event", listener);
  });
});
