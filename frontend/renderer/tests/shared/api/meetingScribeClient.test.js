/* @vitest-environment jsdom */
import { beforeEach, describe, expect, test, vi } from "vitest";

describe("meetingScribeClient", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
    Reflect.deleteProperty(window, "meetingScribe");
  });

  test("uses safe fallbacks when Electron preload is unavailable", async () => {
    const { meetingScribeClient } = await import("../../../src/shared/api/meetingScribeClient");

    await expect(meetingScribeClient.request()).rejects.toThrow("Electron preload unavailable");
    await expect(meetingScribeClient.status()).resolves.toMatchObject({ ready: false, running: false });
    await expect(meetingScribeClient.resourceUsage()).resolves.toMatchObject({
      unavailable: true,
      reason: "Electron preload unavailable"
    });
    expect(meetingScribeClient.onBackendEvent()).toEqual(expect.any(Function));
  });

  test("wraps preload APIs and falls back to backend resource usage", async () => {
    const preload = {
      request: vi.fn(async (method) => {
        if (method === "get_resource_usage") {
          return { cpuPct: 12, memoryBytes: 4096, pid: 77, system: { cpuPct: 50 }, gpus: [{ name: "GPU" }] };
        }
        return {};
      }),
      resourceUsage: vi.fn(async () => {
        throw new Error("ipc down");
      }),
      status: vi.fn(async () => ({ ready: true }))
    };
    Object.defineProperty(window, "meetingScribe", { configurable: true, value: preload });

    const { meetingScribeClient } = await import("../../../src/shared/api/meetingScribeClient");
    const usage = await meetingScribeClient.resourceUsage();

    expect(usage).toMatchObject({
      source: "backend",
      app: { backendPid: 77, backendCpuPct: 12, backendMemoryBytes: 4096 },
      system: { cpuPct: 50 },
      gpus: [{ name: "GPU" }]
    });
    expect(await meetingScribeClient.recentBackendEvents()).toEqual([]);
    expect(await meetingScribeClient.showDebugConsole()).toEqual({ shown: false });
  });
});
