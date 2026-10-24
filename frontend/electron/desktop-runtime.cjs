'use strict';

const path = require('node:path');
const { createPythonBackendClass } = require('./python-backend.cjs');
const { createResourceMonitor } = require('./resource-monitor.cjs');
const { stoppedResultForMethod, toNumber } = require('./backend-protocol.cjs');

const createDesktopRuntime = ({
  electron,
  childProcess,
  fs,
  os,
  ipcHandlers,
  rendererUrl,
  logger = console,
}) => {
  const { app, BrowserWindow, dialog, ipcMain } = electron;
  const { execFile, spawn } = childProcess;
  const { existsSync, readFileSync } = fs;
  const { registerIpcHandlers } = ipcHandlers;
  const { rendererUrlFromEnv } = rendererUrl;

  const appRoot = app.isPackaged
    ? path.join(process.resourcesPath, 'app')
    : path.resolve(__dirname, '..', '..');
  const backendRoot = app.isPackaged
    ? path.join(process.resourcesPath, 'backend')
    : appRoot;
  const runtimeRoot = app.isPackaged ? path.dirname(process.execPath) : appRoot;
  const windowIconPath = path.join(
    __dirname,
    'assets',
    process.platform === 'win32' ? 'icon.ico' : 'icon.png',
  );

  if (process.platform === 'win32') {
    app.setAppUserModelId('com.meetingscribe.app');
  }

  let isQuitting = false;
  const PythonBackend = createPythonBackendClass({
    spawn,
    existsSync,
    isPackaged: () => app.isPackaged,
    isQuitting: () => isQuitting,
    logger,
  });

  const backend = new PythonBackend({
    root: appRoot,
    backendRoot,
    runtimeRoot,
  });
  const { electronResourceUsage, nvidiaGpuUsage, getResourceUsage } =
    createResourceMonitor({ app, backend, execFile, os, logger });
  let mainWindow = null;
  let debugWindow = null;

  const readSavedContentProtection = () => {
    const configPath = path.join(runtimeRoot, 'config.json');
    if (!existsSync(configPath)) {
      return false;
    }
    try {
      const config = JSON.parse(readFileSync(configPath, 'utf8'));
      return Boolean(config?.ui?.screen_capture_protection);
    } catch (error) {
      logger.warn('[window] failed to read content protection setting', error);
      return false;
    }
  };

  const setWindowContentProtection = (window, enabled) => {
    if (!window || window.isDestroyed()) {
      return { enabled: false };
    }
    const nextEnabled = Boolean(enabled);
    window.setContentProtection(nextEnabled);
    return { enabled: nextEnabled };
  };

  const reloadWindow = (window) => {
    if (!window || window.isDestroyed()) {
      return { reloaded: false };
    }
    window.webContents.reloadIgnoringCache();
    return { reloaded: true };
  };

  const loadRenderer = (window, query = {}) => {
    if (app.isPackaged && !process.env.ELECTRON_RENDERER_URL) {
      window.loadFile(
        path.join(appRoot, 'build', 'electron-renderer', 'index.html'),
        { query },
      );
      return;
    }
    const rendererUrl = rendererUrlFromEnv(process.env);
    const url = new URL(rendererUrl);
    for (const key of Object.keys(query)) {
      url.searchParams.set(key, String(query[key]));
    }
    window.loadURL(url.toString());
  };

  const createDebugWindow = () => {
    const window = new BrowserWindow({
      width: 980,
      height: 720,
      minWidth: 720,
      minHeight: 520,
      backgroundColor: '#0a0a0a',
      title: 'Meeting Scribe Diagnostics',
      icon: windowIconPath,
      show: false,
      webPreferences: {
        preload: path.join(__dirname, 'preload.cjs'),
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: false,
      },
    });

    backend.attachWindow(window);
    window.setAlwaysOnTop(false);
    window.on('close', (event) => {
      if (isQuitting) {
        return;
      }
      event.preventDefault();
      window.hide();
    });
    window.on('closed', () => {
      if (debugWindow === window) {
        debugWindow = null;
      }
    });
    loadRenderer(window, { debugConsole: '1' });
    return window;
  };

  const showDebugConsole = () => {
    if (!debugWindow || debugWindow.isDestroyed()) {
      debugWindow = createDebugWindow();
    }
    if (debugWindow.isMinimized()) {
      debugWindow.restore();
    }
    debugWindow.showInactive();
    return { shown: true };
  };

  const createWindow = () => {
    const window = new BrowserWindow({
      width: 1320,
      height: 880,
      minWidth: 980,
      minHeight: 640,
      backgroundColor: '#111111',
      title: 'Meeting Scribe',
      icon: windowIconPath,
      webPreferences: {
        preload: path.join(__dirname, 'preload.cjs'),
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: false,
      },
    });

    mainWindow = window;
    setWindowContentProtection(window, readSavedContentProtection());
    backend.attachWindow(window);
    backend.start();

    window.on('close', (event) => {
      if (backend.activeDownloads <= 0) {
        return;
      }
      const choice = dialog.showMessageBoxSync(window, {
        type: 'warning',
        buttons: ['Keep Downloading', 'Close Anyway'],
        defaultId: 0,
        cancelId: 0,
        message: 'Model download is still running',
        detail:
          'Closing the app now will stop the Python backend ' +
          'and interrupt the model download.',
      });
      if (choice === 0) {
        event.preventDefault();
      }
    });

    loadRenderer(window);
    window.on('closed', () => {
      if (mainWindow === window) {
        mainWindow = null;
      }
      if (debugWindow && !debugWindow.isDestroyed()) {
        debugWindow.destroy();
      }
    });
  };

  const boot = () => {
    app
      .whenReady()
      .then(() => {
        registerIpcHandlers({
          BrowserWindow,
          backend,
          getResourceUsage,
          ipcMain,
          reloadWindow,
          setWindowContentProtection,
          showDebugConsole,
        });
        createWindow();

        app.on('activate', () => {
          if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
          }
        });
      })
      .catch((error) => {
        logger.error('[desktop] startup failed', error);
        app.quit();
      });

    app.on('before-quit', () => {
      isQuitting = true;
      backend.stop();
    });

    app.on('window-all-closed', () => {
      if (process.platform !== 'darwin') {
        app.quit();
      }
    });
  };

  return {
    PythonBackend,
    backend,
    boot,
    createDebugWindow,
    createWindow,
    electronResourceUsage,
    getResourceUsage,
    loadRenderer,
    nvidiaGpuUsage,
    readSavedContentProtection,
    reloadWindow,
    setWindowContentProtection,
    showDebugConsole,
    stoppedResultForMethod,
    toNumber,
  };
};

module.exports = { createDesktopRuntime };
