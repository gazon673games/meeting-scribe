const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("meetingScribe", {
  request: (method, params = {}) => ipcRenderer.invoke("backend:request", method, params),
  status: () => ipcRenderer.invoke("backend:status"),
  onBackendEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("backend:event", listener);
    return () => ipcRenderer.removeListener("backend:event", listener);
  }
});
