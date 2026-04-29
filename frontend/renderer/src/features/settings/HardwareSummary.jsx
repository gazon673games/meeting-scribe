import { Cpu, HardDrive, Microchip } from "lucide-react";

import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";

export function HardwareSummary({ hardware }) {
  const cpu = hardware?.cpu || {};
  const memory = hardware?.memory || {};
  const gpus = Array.isArray(hardware?.gpus) ? hardware.gpus : [];

  return (
    <CollapsibleSection title="Current Hardware" defaultOpen={true}>
      <div className="hardware-grid">
        <HardwareTile icon={Cpu} label="CPU" value={cpu.name || "CPU"} meta={`${Number(cpu.logicalCores || 0) || "-"} logical cores`} />
        <HardwareTile icon={HardDrive} label="Memory" value={formatBytes(memory.totalBytes)} meta="Total RAM" />
        {gpus.length ? (
          gpus.map((gpu, index) => (
            <HardwareTile
              key={`${gpu.name}-${index}`}
              icon={Microchip}
              label="GPU"
              value={gpu.name || "NVIDIA GPU"}
              meta={`${gpu.memoryUsedMiB || 0}/${gpu.memoryTotalMiB || 0} MiB, ${gpu.gpuUtilizationPct || 0}%`}
            />
          ))
        ) : (
          <HardwareTile icon={Microchip} label="GPU" value="No NVIDIA GPU data" meta="CUDA may be unavailable" />
        )}
      </div>
    </CollapsibleSection>
  );
}

function HardwareTile({ icon: Icon, label, value, meta }) {
  return (
    <div className="hardware-tile">
      <Icon size={16} />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{meta}</small>
      </div>
    </div>
  );
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "-";
  }
  const gib = bytes / 1024 / 1024 / 1024;
  return `${gib.toFixed(gib >= 10 ? 0 : 1)} GB`;
}
