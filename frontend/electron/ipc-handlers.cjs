'use strict';

const registerIpcHandlers = ({
  BrowserWindow,
  backend,
  getResourceUsage,
  ipcMain,
  reloadWindow,
  setWindowContentProtection,
  showDebugConsole,
}) => {
  ipcMain.handle('backend:request', (_event, method, params) =>
    backend.request(method, params),
  );
  ipcMain.handle('backend:status', () => backend.status());
  ipcMain.handle('backend:recent-events', () => backend.recentEvents());
  ipcMain.handle('debug:show', () => showDebugConsole());
  ipcMain.handle('window:set-content-protection', (event, enabled) => {
    const window = BrowserWindow.fromWebContents(event.sender);
    return setWindowContentProtection(window, enabled);
  });
  ipcMain.handle('window:reload', (event) => {
    const window = BrowserWindow.fromWebContents(event.sender);
    return reloadWindow(window);
  });
  ipcMain.handle('app:get-resource-usage', () => getResourceUsage());
};

module.exports = { registerIpcHandlers };
