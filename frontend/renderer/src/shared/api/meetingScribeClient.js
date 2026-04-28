const preloadApi = globalThis.window?.meetingScribe;

export const meetingScribeClient = preloadApi
  ? {
      ...preloadApi,
      reloadApp: preloadApi.reloadApp || (async () => ({ reloaded: false })),
      resourceUsage: () => electronResourceUsageWithFallback(preloadApi),
      setContentProtection: preloadApi.setContentProtection || (async () => ({ enabled: false }))
    }
  : {
    reloadApp: async () => {
      globalThis.location?.reload();
      return { reloaded: true };
    },
    resourceUsage: async () => emptyResourceUsage("Electron preload unavailable"),
    status: async () => ({ ready: false, running: false, lastError: "Electron preload unavailable" }),
    setContentProtection: async () => ({ enabled: false }),
    request: async () => {
      throw new Error("Electron preload unavailable");
    },
    onBackendEvent: () => () => {}
  };

async function electronResourceUsageWithFallback(api) {
  if (api?.resourceUsage) {
    try {
      return await api.resourceUsage();
    } catch {
      return backendResourceUsageFallback(api.request);
    }
  }
  return backendResourceUsageFallback(api?.request);
}

async function backendResourceUsageFallback(request) {
  if (!request) {
    return emptyResourceUsage("Resource IPC unavailable");
  }
  try {
    const usage = await request("get_resource_usage", {});
    return {
      source: "backend",
      app: {
        cpuPct: usage.cpuPct,
        memoryBytes: usage.memoryBytes,
        backendCpuPct: usage.cpuPct,
        backendMemoryBytes: usage.memoryBytes,
        backendPid: usage.pid
      },
      system: usage.system || {},
      gpus: usage.gpus || []
    };
  } catch (error) {
    return emptyResourceUsage(error?.message || "Resource usage unavailable");
  }
}

function emptyResourceUsage(reason) {
  return {
    unavailable: true,
    reason: reason || "Resource usage unavailable",
    app: {},
    system: {},
    gpus: []
  };
}
