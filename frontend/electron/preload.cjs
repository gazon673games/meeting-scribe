const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("meetingScribe", {
  request: (method, params = {}) => ipcRenderer.invoke("backend:request", method, params),
  status: () => ipcRenderer.invoke("backend:status"),
  recentBackendEvents: () => ipcRenderer.invoke("backend:recent-events"),
  resourceUsage: () => ipcRenderer.invoke("app:get-resource-usage"),
  showDebugConsole: () => ipcRenderer.invoke("debug:show"),
  setContentProtection: (enabled) => ipcRenderer.invoke("window:set-content-protection", Boolean(enabled)),
  reloadApp: () => ipcRenderer.invoke("window:reload"),
  onBackendEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("backend:event", listener);
    return () => ipcRenderer.removeListener("backend:event", listener);
  }
});
