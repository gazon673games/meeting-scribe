import { createRequire } from "node:module";

import { afterEach, describe, expect, test, vi } from "vitest";

const require = createRequire(import.meta.url);
const preloadPath = require.resolve("../preload.cjs");

function unloadPreload() {
  delete require.cache[preloadPath];
  delete global.__MEETING_SCRIBE_PRELOAD_DEPS__;
}

afterEach(() => {
  vi.restoreAllMocks();
  unloadPreload();
});

describe("electron preload entry", () => {
  test("exposes the Meeting Scribe API through the isolated context bridge", () => {
    const api = { request: vi.fn() };
    const ipcRenderer = { invoke: vi.fn() };
    const contextBridge = { exposeInMainWorld: vi.fn() };
    const createMeetingScribeApi = vi.fn(() => api);

    unloadPreload();
    global.__MEETING_SCRIBE_PRELOAD_DEPS__ = {
      electron: { contextBridge, ipcRenderer },
      preloadApi: { createMeetingScribeApi }
    };

    const preload = require(preloadPath);

    expect(createMeetingScribeApi).toHaveBeenCalledWith(ipcRenderer);
    expect(contextBridge.exposeInMainWorld).toHaveBeenCalledWith("meetingScribe", api);
    expect(preload.meetingScribeApi).toBe(api);
  });
});
