'use strict';

const { spawn } = require('node:child_process');
const { existsSync, readdirSync } = require('node:fs');
const path = require('node:path');
const { rendererUrlFromEnv } = require('./renderer-url.cjs');

const projectRoot = path.resolve(__dirname, '..', '..');

const resolveElectronExecutable = () => {
  try {
    const electronPath = require('electron');
    if (typeof electronPath === 'string' && existsSync(electronPath)) {
      return electronPath;
    }
  } catch (error) {
    console.warn('Cannot resolve installed Electron runtime:', error.message);
  }

  const runtimeRoot = path.join(projectRoot, 'build', 'electron-runtime');
  if (existsSync(runtimeRoot)) {
    const executableName =
      process.platform === 'win32' ? 'electron.exe' : 'electron';
    const dirs = readdirSync(runtimeRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => path.join(runtimeRoot, entry.name))
      .sort()
      .reverse();
    for (const dir of dirs) {
      const candidate = path.join(dir, executableName);
      if (existsSync(candidate)) {
        return candidate;
      }
    }
  }

  throw new Error(
    'Electron binary is unavailable. ' +
      'Run `npm rebuild electron`, then try again.',
  );
};

const electronExe = resolveElectronExecutable();
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;
env.ELECTRON_RENDERER_URL ||= rendererUrlFromEnv(env);

const child = spawn(electronExe, [projectRoot], {
  cwd: projectRoot,
  env,
  stdio: 'inherit',
});

child.once('error', (error) => {
  console.error('Cannot start Electron:', error.message);
  process.exit(1);
});

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
