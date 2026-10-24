import { formatBytes, formatNumber } from '../../shared/lib/format';
import { StatLine } from '../../shared/ui/StatLine';

const hasFiniteValue = (object, key) => {
  if (!Object.hasOwn(object || {}, key)) {
    return false;
  }
  const value = object[key];
  return (
    value !== null &&
    value !== undefined &&
    value !== '' &&
    Number.isFinite(Number(value))
  );
};

const readFinite = (source, key) => {
  if (!hasFiniteValue(source, key)) {
    return null;
  }
  return Number(source[key]);
};

const usageMetrics = (usage) => {
  const app = usage?.app || {};
  const system = usage?.system || {};
  const gpu = Array.isArray(usage?.gpus) ? usage.gpus[0] : null;
  const totalMemory = readFinite(system, 'totalMemoryBytes');
  const freeMemory = readFinite(system, 'freeMemoryBytes');
  const appMemoryRaw = readFinite(app, 'memoryBytes');
  return {
    totalMemory,
    freeMemory,
    usedMemory:
      totalMemory !== null && freeMemory !== null
        ? totalMemory - freeMemory
        : null,
    logicalCores: readFinite(system, 'logicalCores'),
    appCpu: readFinite(app, 'cpuPct'),
    appMemory: appMemoryRaw !== null && appMemoryRaw > 0 ? appMemoryRaw : null,
    gpuLoad: readFinite(gpu, 'gpuUtilizationPct'),
    gpuUsedMemory: readFinite(gpu, 'memoryUsedMiB'),
    gpuTotalMemory: readFinite(gpu, 'memoryTotalMiB'),
  };
};

const formatCpuValue = (metrics) => {
  if (metrics.appCpu === null) {
    return '-';
  }
  if (metrics.logicalCores !== null) {
    return `${formatNumber(metrics.appCpu)}% / ${metrics.logicalCores} cores`;
  }
  return `${formatNumber(metrics.appCpu)}%`;
};

const bytesOrDash = (value) => (value !== null ? formatBytes(value) : '-');

const joinTriplet = (first, second, third) =>
  [first, second, third].join(' / ');

const formatRamValue = (metrics) => {
  if (metrics.appMemory === null && metrics.usedMemory === null) {
    return '-';
  }
  return joinTriplet(
    bytesOrDash(metrics.appMemory),
    bytesOrDash(metrics.usedMemory),
    bytesOrDash(metrics.totalMemory),
  );
};

const formatGpuValue = (metrics) => {
  if (metrics.gpuLoad === null && metrics.gpuUsedMemory === null) {
    return '-';
  }
  return joinTriplet(
    metrics.gpuLoad !== null ? `${metrics.gpuLoad}%` : '-',
    metrics.gpuUsedMemory !== null ? `${metrics.gpuUsedMemory} MiB` : '-',
    metrics.gpuTotalMemory !== null ? `${metrics.gpuTotalMemory} MiB` : '-',
  );
};

export const ResourceUsage = ({ usage }) => {
  const metrics = usageMetrics(usage);
  const cpuValue = formatCpuValue(metrics);
  const ramValue = formatRamValue(metrics);
  const gpuValue = formatGpuValue(metrics);

  return (
    <div className="resource-strip">
      <div className="resource-strip-head">Resources</div>
      <StatLine label="CPU" value={cpuValue} accent="blue" />
      <StatLine label="RAM" value={ramValue} />
      <StatLine label="GPU" value={gpuValue} />
      {usage?.unavailable ? (
        <small className="resource-note">
          {usage.reason || 'Resource usage unavailable'}
        </small>
      ) : null}
    </div>
  );
};
