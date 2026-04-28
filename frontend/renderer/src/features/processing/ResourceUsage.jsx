import { formatBytes, formatNumber } from "../../shared/lib/format";
import { StatLine } from "../../shared/ui/StatLine";

export function ResourceUsage({ usage }) {
  const app = usage?.app || {};
  const system = usage?.system || {};
  const gpu = Array.isArray(usage?.gpus) ? usage.gpus[0] : null;
  const totalMemory = hasFiniteValue(system, "totalMemoryBytes") ? Number(system.totalMemoryBytes) : null;
  const freeMemory = hasFiniteValue(system, "freeMemoryBytes") ? Number(system.freeMemoryBytes) : null;
  const usedMemory = totalMemory !== null && freeMemory !== null ? totalMemory - freeMemory : null;
  const logicalCores = hasFiniteValue(system, "logicalCores") ? Number(system.logicalCores) : null;
  const appCpu = hasFiniteValue(app, "cpuPct") ? Number(app.cpuPct) : null;
  const appMemory = hasFiniteValue(app, "memoryBytes") && Number(app.memoryBytes) > 0 ? Number(app.memoryBytes) : null;
  const gpuLoad = gpu && hasFiniteValue(gpu, "gpuUtilizationPct") ? Number(gpu.gpuUtilizationPct) : null;
  const gpuUsedMemory = gpu && hasFiniteValue(gpu, "memoryUsedMiB") ? Number(gpu.memoryUsedMiB) : null;
  const gpuTotalMemory = gpu && hasFiniteValue(gpu, "memoryTotalMiB") ? Number(gpu.memoryTotalMiB) : null;

  const cpuValue =
    appCpu === null
      ? "-"
      : logicalCores !== null
        ? `${formatNumber(appCpu)}% / ${logicalCores} cores`
        : `${formatNumber(appCpu)}%`;

  const ramValue =
    appMemory === null && usedMemory === null
      ? "-"
      : [
          appMemory !== null ? formatBytes(appMemory) : "-",
          usedMemory !== null ? formatBytes(usedMemory) : "-",
          totalMemory !== null ? formatBytes(totalMemory) : "-"
        ].join(" / ");

  const gpuValue =
    gpuLoad === null && gpuUsedMemory === null
      ? "-"
      : [
          gpuLoad !== null ? `${gpuLoad}%` : "-",
          gpuUsedMemory !== null ? `${gpuUsedMemory} MiB` : "-",
          gpuTotalMemory !== null ? `${gpuTotalMemory} MiB` : "-"
        ].join(" / ");

  return (
    <div className="resource-strip">
      <div className="resource-strip-head">Resources</div>
      <StatLine label="CPU" value={cpuValue} accent="blue" />
      <StatLine label="RAM" value={ramValue} />
      <StatLine label="GPU" value={gpuValue} />
      {usage?.unavailable ? <small className="resource-note">{usage.reason || "Resource usage unavailable"}</small> : null}
    </div>
  );
}

function hasFiniteValue(object, key) {
  if (!Object.prototype.hasOwnProperty.call(object || {}, key)) {
    return false;
  }
  return Number.isFinite(Number(object[key]));
}

function readGpuFreeMemory(gpu) {
  if (hasFiniteValue(gpu, "memoryFreeMiB")) {
    return Math.max(0, Number(gpu.memoryFreeMiB));
  }
  if (hasFiniteValue(gpu, "memoryTotalMiB") && hasFiniteValue(gpu, "memoryUsedMiB")) {
    return Math.max(0, Number(gpu.memoryTotalMiB) - Number(gpu.memoryUsedMiB));
  }
  return null;
}
