import { EventEmitter } from 'node:events';
import { createRequire } from 'node:module';
import { afterEach, expect, test, vi } from 'vitest';

const require = createRequire(import.meta.url);
const { runDev } = require('../dev-runner.cjs');
const makeChild = () =>
  Object.assign(new EventEmitter(), {
    exitCode: null,
    killed: false,
    kill: vi.fn(),
  });
const chooseEndpoint = async () => ({
  host: 'localhost',
  port: 5173,
  url: 'http://localhost:5173',
});

afterEach(() => vi.restoreAllMocks());

test('cleans up Vite and signal handlers when readiness fails', async () => {
  vi.spyOn(console, 'log').mockImplementation(() => {});
  const vite = makeChild();
  const processEvents = new EventEmitter();
  const exit = vi.fn();
  await expect(
    runDev({
      chooseEndpoint,
      spawnProcess: () => vite,
      processEvents,
      exit,
      waitForRenderer: async () => {
        throw new Error('not ready');
      },
    }),
  ).rejects.toThrow('not ready');
  expect(vite.kill).toHaveBeenCalledOnce();
  expect(exit).toHaveBeenCalledWith(1);
  expect(processEvents.listenerCount('SIGINT')).toBe(0);
});

test('handles an Electron spawn error and stops its renderer', async () => {
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
  const vite = makeChild();
  const electron = makeChild();
  const spawnProcess = vi
    .fn()
    .mockReturnValueOnce(vite)
    .mockReturnValueOnce(electron);
  const exit = vi.fn();
  await runDev({
    chooseEndpoint,
    spawnProcess,
    exit,
    processEvents: new EventEmitter(),
    waitForRenderer: async () => {},
  });
  electron.emit('error', new Error('ENOENT'));
  expect(exit).toHaveBeenCalledWith(1);
  expect(vite.kill).toHaveBeenCalledOnce();
});
