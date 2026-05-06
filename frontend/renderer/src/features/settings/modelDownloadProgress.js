import { formatBytes } from "../../shared/lib/format";

export function formatDownloadProgress(model, { fallback = "", includeTotal = true } = {}) {
  const downloadedBytes = positiveNumber(model.downloadedBytes);
  const speedBps = positiveNumber(model.speedBps);
  if (hasNoProgress(downloadedBytes, speedBps) && fallback) {
    return fallback;
  }
  return `${downloadedLabel(downloadedBytes)}${totalLabel(model, includeTotal)}${speedLabel(speedBps)}`;
}

function positiveNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, number) : 0;
}

function hasNoProgress(downloadedBytes, speedBps) {
  return downloadedBytes <= 0 && speedBps <= 0;
}

function downloadedLabel(downloadedBytes) {
  return downloadedBytes > 0 ? formatBytes(downloadedBytes) : "0 B";
}

function totalLabel(model, includeTotal) {
  const totalBytes = positiveNumber(model.totalBytes);
  return includeTotal && totalBytes > 0 ? ` / ${formatBytes(totalBytes)}` : "";
}

function speedLabel(speedBps) {
  return speedBps > 0 ? ` - ${formatBytes(speedBps)}/s` : "";
}
