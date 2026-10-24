import { EventEmitter } from 'node:events';
import { createRequire } from 'node:module';

import { afterEach, expect, test, vi } from 'vitest';

const require = createRequire(import.meta.url);
const { createPythonBackendClass } = require('../python-backend.cjs');
const backends = [];

const makeChild = () => {
  const child = new EventEmitter();
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.stdout.setEncoding = vi.fn();
  child.stderr.setEncoding = vi.fn();
  child.stdin = new EventEmitter();
  child.stdin.writable = true;
  child.stdin.write = vi.fn((line, encoding, done) => done());
  child.kill = vi.fn();
  return child;
};

const setup = (options = {}) => {
  const children = [makeChild(), makeChild()];
  const logger = { error: vi.fn() };
  let nextChild = 0;
  const Backend = createPythonBackendClass({
    spawn: () => children[nextChild++],
    existsSync: () => false,
    isPackaged: () => false,
    isQuitting: () => false,
    logger,
  });
  const backend = new Backend({
    root: '.',
    backendRoot: '.',
    runtimeRoot: '.',
    ...options,
  });
  backends.push(backend);
  return { backend, children, logger };
};

afterEach(() => {
  for (const backend of backends) backend.stop();
  backends.length = 0;
  vi.useRealTimers();
});

test('rejects malformed envelopes without losing a valid pending response', async () => {
  const { backend, children, logger } = setup();
  const response = backend.request('ping');
  for (const line of ['null', '[]', '{"event":{}}', '{"id":"1","ok":"yes"}']) {
    expect(() => backend.handleMessageLine(line)).not.toThrow();
  }
  expect(logger.error).toHaveBeenCalledTimes(4);
  children[0].stdout.emit(
    'data',
    '{"id":"1","ok":true,"result":{"pong":true}}\n',
  );
  await expect(response).resolves.toEqual({ pong: true });
});

test('bounds pending requests and frees capacity when a response arrives', async () => {
  const { backend } = setup({ maxPendingRequests: 1 });
  const first = backend.request('ping');
  await expect(backend.request('ping')).rejects.toThrow('queue is full');
  backend.handleMessageLine('{"id":"1","ok":true,"result":1}');
  await expect(first).resolves.toBe(1);
  const next = backend.request('ping');
  backend.handleMessageLine('{"id":"2","ok":true,"result":2}');
  await expect(next).resolves.toBe(2);
});

test('expires unanswered requests and accepts another request afterwards', async () => {
  vi.useFakeTimers();
  const { backend } = setup({ maxPendingRequests: 1, requestTimeoutMs: 50 });
  const failure = expect(backend.request('ping')).rejects.toThrow('timed out');
  await vi.advanceTimersByTimeAsync(50);
  await failure;
  const next = backend.request('ping');
  backend.handleMessageLine('{"id":"2","ok":true,"result":true}');
  await expect(next).resolves.toBe(true);
});

test('releases request capacity when stdin throws synchronously', async () => {
  const { backend, children } = setup({ maxPendingRequests: 1 });
  children[0].stdin.write.mockImplementationOnce(() => {
    throw new Error('pipe closed');
  });
  await expect(backend.request('ping')).rejects.toThrow('pipe closed');
  const next = backend.request('ping');
  backend.handleMessageLine('{"id":"2","ok":true,"result":true}');
  await expect(next).resolves.toBe(true);
});

test('reports process startup errors and rejects pending requests', async () => {
  const { backend, children } = setup();
  const failure = expect(backend.request('ping')).rejects.toThrow(
    'spawn failed',
  );
  children[0].emit('error', new Error('spawn failed'));
  await failure;
  expect(backend.status()).toEqual({
    ready: false,
    running: false,
    lastError: 'spawn failed',
  });
});

test('handles broken stdin streams without an uncaught error event', async () => {
  const { backend, children } = setup();
  const failure = expect(backend.request('ping')).rejects.toThrow(
    'broken pipe',
  );
  expect(() =>
    children[0].stdin.emit('error', new Error('broken pipe')),
  ).not.toThrow();
  await failure;
  expect(backend.status().running).toBe(false);
  expect(children[0].kill).toHaveBeenCalled();
});

test('ignores late output and exit from a process replaced after stop', async () => {
  const { backend, children } = setup();
  const first = backend.request('ping');
  children[0].stdout.emit('data', '{"unfinished":');
  backend.stop();
  await expect(first).resolves.toBeNull();
  const next = backend.request('ping');
  children[1].stdout.emit('data', '{"event":{"type":"backend_ready"}}\n');
  children[0].emit('exit', 0, null);
  children[0].stdout.emit('data', '{"event":{"type":"backend_exit"}}\n');
  expect(backend.status()).toMatchObject({ running: true, ready: true });
  children[1].stdout.emit('data', '{"id":"2","ok":true,"result":42}\n');
  await expect(next).resolves.toBe(42);
});
