const preloadApi = globalThis.window?.meetingScribe;

export const meetingScribeClient = preloadApi
  ? {
      ...preloadApi,
      reloadApp: preloadApi.reloadApp || (async () => ({ reloaded: false })),
      setContentProtection: preloadApi.setContentProtection || (async () => ({ enabled: false }))
    }
  : {
    reloadApp: async () => {
      globalThis.location?.reload();
      return { reloaded: true };
    },
    status: async () => ({ ready: false, running: false, lastError: "Electron preload unavailable" }),
    setContentProtection: async () => ({ enabled: false }),
    request: async () => {
      throw new Error("Electron preload unavailable");
    },
    onBackendEvent: () => () => {}
  };
