export function displayDeviceLabel(label) {
  return String(label || "")
    .replace(/^\[\d+\]\s*/, "")
    .replace(/\s*\(in=[^)]*\)\s*$/i, "")
    .trim();
}
