import { formatBytes } from '../../shared/lib/format';

const positiveNumber = (value) => {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, number) : 0;
};

const hasNoProgress = (downloadedBytes, speedBps) =>
  downloadedBytes <= 0 && speedBps <= 0;

const downloadedLabel = (downloadedBytes) =>
  downloadedBytes > 0 ? formatBytes(downloadedBytes) : '0 B';

const totalLabel = (model, includeTotal) => {
  const totalBytes = positiveNumber(model.totalBytes);
  return includeTotal && totalBytes > 0 ? ` / ${formatBytes(totalBytes)}` : '';
};

const speedLabel = (speedBps) =>
  speedBps > 0 ? ` - ${formatBytes(speedBps)}/s` : '';

export const formatDownloadProgress = (
  model,
  { fallback = '', includeTotal = true } = {},
) => {
  const downloadedBytes = positiveNumber(model.downloadedBytes);
  const speedBps = positiveNumber(model.speedBps);
  if (hasNoProgress(downloadedBytes, speedBps) && fallback) {
    return fallback;
  }
  const downloaded = downloadedLabel(downloadedBytes);
  const total = totalLabel(model, includeTotal);
  const speed = speedLabel(speedBps);
  return `${downloaded}${total}${speed}`;
};
