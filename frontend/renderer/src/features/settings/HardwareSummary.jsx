import { Cpu, HardDrive, Microchip } from 'lucide-react';

import { CollapsibleSection } from '../../shared/ui/CollapsibleSection';

const HardwareTile = ({ icon: Icon, label, value, meta }) => (
  <div className="hardware-tile">
    <Icon size={16} />
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{meta}</small>
    </div>
  </div>
);

const formatBytes = (value) => {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '-';
  }
  const gib = bytes / 1024 / 1024 / 1024;
  return `${gib.toFixed(gib >= 10 ? 0 : 1)} GB`;
};

const gpuMemorySummary = (gpu) => {
  const memory = `${gpu.memoryUsedMiB || 0}/${gpu.memoryTotalMiB || 0}`;
  return `${memory} MiB, ${gpu.gpuUtilizationPct || 0}%`;
};

export const HardwareSummary = ({ hardware }) => {
  const cpu = hardware?.cpu || {};
  const memory = hardware?.memory || {};
  const gpus = Array.isArray(hardware?.gpus) ? hardware.gpus : [];

  return (
    <CollapsibleSection
      title="System information"
      defaultOpen={false}
      summary={cpu.name || 'CPU, memory and GPU'}
    >
      <div className="hardware-grid">
        <HardwareTile
          icon={Cpu}
          label="CPU"
          value={cpu.name || 'CPU'}
          meta={`${Number(cpu.logicalCores || 0) || '-'} logical cores`}
        />
        <HardwareTile
          icon={HardDrive}
          label="Memory"
          value={formatBytes(memory.totalBytes)}
          meta="Total RAM"
        />
        {gpus.length ? (
          gpus.map((gpu, index) => (
            <HardwareTile
              key={`${gpu.name}-${index}`}
              icon={Microchip}
              label="GPU"
              value={gpu.name || 'NVIDIA GPU'}
              meta={gpuMemorySummary(gpu)}
            />
          ))
        ) : (
          <HardwareTile
            icon={Microchip}
            label="GPU"
            value="No NVIDIA GPU data"
            meta="CUDA may be unavailable"
          />
        )}
      </div>
    </CollapsibleSection>
  );
};
