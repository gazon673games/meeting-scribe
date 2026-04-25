const preloadApi = globalThis.window?.meetingScribe;

export const meetingScribeClient =
  preloadApi ?? {
    status: async () => ({ ready: false, running: false, lastError: "Electron preload unavailable" }),
    request: async () => {
      throw new Error("Electron preload unavailable");
    },
    onBackendEvent: () => () => {}
  };
