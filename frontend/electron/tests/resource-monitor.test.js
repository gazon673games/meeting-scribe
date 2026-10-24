import { createRequire } from 'node:module';

import { afterEach, expect, test, vi } from 'vitest';

const require = createRequire(import.meta.url);
const { createResourceMonitor } = require('../resource-monitor.cjs');

const setup = (output = '') => {
  let elapsedMs = 0;
  const logger = { warn: vi.fn() };
  const backend = { process: null, request: vi.fn() };
  const execFile = vi.fn((_command, _args, _options, done) => {
    done(null, output);
  });
  const monitor = createResourceMonitor({
    app: { getAppMetrics: () => [] },
    backend,
    execFile,
    os: { totalmem: () => 8000, freemem: () => 4000, cpus: () => [] },
    logger,
    monotonicNow: () => elapsedMs,
  });
  return {
    backend,
    execFile,
    logger,
    monitor,
    advance: (milliseconds) => {
      elapsedMs += milliseconds;
    },
  };
};

afterEach(() => vi.restoreAllMocks());

test('returns independent GPU snapshots while retaining the query cache', async () => {
  const { execFile, monitor } = setup(' GPU , 12000 , 3000 , 65 , 20\n');
  const first = await monitor.nvidiaGpuUsage();
  first[0].name = 'mutated';
  first.length = 0;

  await expect(monitor.nvidiaGpuUsage()).resolves.toEqual([
    {
      name: 'GPU',
      memoryTotalMiB: 12000,
      memoryUsedMiB: 3000,
      memoryFreeMiB: 9000,
      gpuUtilizationPct: 65,
      memoryUtilizationPct: 20,
    },
  ]);
  expect(execFile).toHaveBeenCalledOnce();
});

test('shares an in-flight GPU query and returns separate snapshots to waiters', async () => {
  const { execFile, monitor } = setup();
  let finish;
  execFile.mockImplementation((_command, _args, _options, done) => {
    finish = done;
  });

  const firstRequest = monitor.nvidiaGpuUsage();
  const secondRequest = monitor.nvidiaGpuUsage();
  expect(execFile).toHaveBeenCalledOnce();
  finish(null, 'GPU, 12000, 3000, 65, 20');
  const [first, second] = await Promise.all([firstRequest, secondRequest]);
  first[0].name = 'mutated';
  first.length = 0;
  expect(second[0].name).toBe('GPU');
  expect(await monitor.nvidiaGpuUsage()).toEqual(second);
});

test('expires GPU results from completion using elapsed time across wall-clock jumps', async () => {
  const { advance, execFile, monitor } = setup('GPU, 12000, 3000, 65, 20');
  const wallClock = vi.spyOn(Date, 'now').mockReturnValue(10000);
  let finish;
  execFile.mockImplementationOnce((_command, _args, _options, done) => {
    finish = done;
  });
  const pending = monitor.nvidiaGpuUsage();
  advance(2000);
  finish(null, 'GPU, 12000, 3000, 65, 20');
  await pending;

  wallClock.mockReturnValue(-100000);
  advance(4999);
  await monitor.nvidiaGpuUsage();
  expect(execFile).toHaveBeenCalledOnce();

  wallClock.mockReturnValue(1000000);
  advance(1);
  await monitor.nvidiaGpuUsage();
  expect(execFile).toHaveBeenCalledTimes(2);
});

test('caches an unavailable GPU query and retries after its TTL', async () => {
  const { advance, execFile, monitor } = setup('GPU, 12000, 3000, 65, 20');
  execFile.mockImplementationOnce((_command, _args, _options, done) => {
    done(new Error('nvidia-smi unavailable'), '');
  });
  await expect(monitor.nvidiaGpuUsage()).resolves.toEqual([]);
  await expect(monitor.nvidiaGpuUsage()).resolves.toEqual([]);
  expect(execFile).toHaveBeenCalledOnce();
  advance(5000);
  expect(await monitor.nvidiaGpuUsage()).toHaveLength(1);
  expect(execFile).toHaveBeenCalledTimes(2);
});

test('releases the pending query after a synchronous process-launch failure', async () => {
  const { execFile, monitor } = setup();
  execFile.mockImplementationOnce(() => {
    throw new Error('launch failed');
  });
  await expect(monitor.nvidiaGpuUsage()).rejects.toThrow('launch failed');
  await expect(monitor.nvidiaGpuUsage()).resolves.toEqual([]);
  expect(execFile).toHaveBeenCalledTimes(2);
});

test('logs backend metric failures and preserves the fallback snapshot', async () => {
  const { backend, logger, monitor } = setup();
  const error = new Error('backend exited');
  backend.request.mockRejectedValue(error);

  const snapshot = await monitor.getResourceUsage();

  expect(logger.warn).toHaveBeenCalledWith(
    '[resources] failed to read backend metrics',
    error,
  );
  expect(snapshot.app).toMatchObject({
    backendPid: 0,
    backendCpuPct: 0,
    backendMemoryBytes: 0,
  });
});
