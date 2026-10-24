'use strict';

const { performance } = require('node:perf_hooks');

const { toNumber } = require('./backend-protocol.cjs');

const GPU_USAGE_CACHE_MS = 5000;
const GPU_QUERY = [
  'name',
  'memory.total',
  'memory.used',
  'utilization.gpu',
  'utilization.memory',
].join(',');

const gpuUsageFromLine = (line) => {
  const columns = line.split(',');
  const memoryTotalMiB = toNumber(columns[1]);
  const memoryUsedMiB = toNumber(columns[2]);
  return {
    name: columns[0].trim() || 'NVIDIA GPU',
    memoryTotalMiB,
    memoryUsedMiB,
    memoryFreeMiB: Math.max(0, memoryTotalMiB - memoryUsedMiB),
    gpuUtilizationPct: toNumber(columns[3]),
    memoryUtilizationPct: toNumber(columns[4]),
  };
};

const gpuSnapshot = (gpus) => gpus.map((gpu) => ({ ...gpu }));

const createResourceMonitor = ({
  app,
  backend,
  execFile,
  os,
  logger = console,
  monotonicNow = () => performance.now(),
}) => {
  let gpuUsageCache = { expiresAt: -Infinity, gpus: [] };
  let gpuUsageRequest = null;

  const electronResourceUsage = () => {
    const metrics = app.getAppMetrics();
    const memoryBytes = metrics.reduce((sum, metric) => {
      const workingSetSize = Number(metric.memory?.workingSetSize || 0);
      return sum + Math.max(0, workingSetSize * 1024);
    }, 0);
    const cpuPct = metrics.reduce(
      (sum, metric) =>
        sum + Math.max(0, Number(metric.cpu?.percentCPUUsage || 0)),
      0,
    );
    return {
      cpuPct,
      memoryBytes: memoryBytes || process.memoryUsage().rss,
      processCount: metrics.length,
    };
  };

  const nvidiaGpuUsage = () => {
    if (monotonicNow() < gpuUsageCache.expiresAt) {
      return Promise.resolve(gpuSnapshot(gpuUsageCache.gpus));
    }
    if (gpuUsageRequest === null) {
      gpuUsageRequest = new Promise((resolve) => {
        execFile(
          'nvidia-smi',
          [`--query-gpu=${GPU_QUERY}`, '--format=csv,noheader,nounits'],
          { timeout: 2000, windowsHide: true },
          (error, stdout) => {
            const gpus = error
              ? []
              : String(stdout || '')
                  .split(/\r?\n/)
                  .map((line) => line.trim())
                  .filter(Boolean)
                  .map(gpuUsageFromLine);
            gpuUsageCache = {
              expiresAt: monotonicNow() + GPU_USAGE_CACHE_MS,
              gpus,
            };
            resolve(gpus);
          },
        );
      }).finally(() => {
        gpuUsageRequest = null;
      });
    }
    return gpuUsageRequest.then(gpuSnapshot);
  };

  const getResourceUsage = async () => {
    const electron = electronResourceUsage();
    let backendUsage = {
      pid: backend.process?.pid || 0,
      cpuPct: 0,
      memoryBytes: 0,
    };
    try {
      backendUsage =
        (await backend.request('get_resource_usage', { includeGpu: false })) ||
        backendUsage;
    } catch (error) {
      logger.warn('[resources] failed to read backend metrics', error);
    }
    const backendMemoryBytes = Math.max(
      0,
      Number(backendUsage.memoryBytes || 0),
    );
    const electronMemoryBytes = Math.max(0, Number(electron.memoryBytes || 0));
    const backendCpuPct = Math.max(0, Number(backendUsage.cpuPct || 0));
    const electronCpuPct = Math.max(0, Number(electron.cpuPct || 0));
    return {
      ts: Date.now(),
      app: {
        cpuPct: electronCpuPct + backendCpuPct,
        memoryBytes: electronMemoryBytes + backendMemoryBytes,
        electronCpuPct,
        electronMemoryBytes,
        backendCpuPct,
        backendMemoryBytes,
        backendPid: Number(backendUsage.pid || backend.process?.pid || 0),
        electronProcessCount: Number(electron.processCount || 0),
      },
      system: {
        totalMemoryBytes: os.totalmem(),
        freeMemoryBytes: os.freemem(),
        logicalCores: os.cpus().length,
      },
      gpus: await nvidiaGpuUsage(),
    };
  };

  return { electronResourceUsage, nvidiaGpuUsage, getResourceUsage };
};

module.exports = { createResourceMonitor };
