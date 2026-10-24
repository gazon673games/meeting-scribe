'use strict';

const { createDesktopRuntime } = require('./desktop-runtime.cjs');

const runtime = createDesktopRuntime({
  electron: require('electron'),
  childProcess: require('node:child_process'),
  fs: require('node:fs'),
  os: require('node:os'),
  ipcHandlers: require('./ipc-handlers.cjs'),
  rendererUrl: require('./renderer-url.cjs'),
});

runtime.boot();

module.exports = { ...runtime, createDesktopRuntime };
