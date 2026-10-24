export const displayDeviceLabel = (label) =>
  String(label || '')
    .replace(/^\[\d+\]\s*/, '')
    .replace(/\s*\(in=[^)]*\)\s*$/i, '')
    .trim();
