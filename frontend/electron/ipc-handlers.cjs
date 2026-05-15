function registerIpcHandlers({
  BrowserWindow,
  backend,
  getResourceUsage,
  ipcMain,
  reloadWindow,
  setWindowContentProtection,
  showDebugConsole
}) {
  ipcMain.handle("backend:request", (_event, method, params) => backend.request(method, params));
  ipcMain.handle("backend:status", () => backend.status());
  ipcMain.handle("backend:recent-events", () => backend.recentEvents());
  ipcMain.handle("debug:show", () => showDebugConsole());
  ipcMain.handle("window:set-content-protection", (event, enabled) => {
    return setWindowContentProtection(BrowserWindow.fromWebContents(event.sender), enabled);
  });
  ipcMain.handle("window:reload", (event) => {
    return reloadWindow(BrowserWindow.fromWebContents(event.sender));
  });
  ipcMain.handle("app:get-resource-usage", () => getResourceUsage());
}

module.exports = { registerIpcHandlers };
