export function formatDelay(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) {
    return "0";
  }
  return Math.abs(number - Math.round(number)) < 1e-6 ? String(Math.round(number)) : number.toFixed(2);
}

export function formatNumber(value) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toFixed(2) : "0.00";
}

export function formatTime(ts) {
  const date = new Date(Number(ts || 0) * 1000);
  if (Number.isNaN(date.getTime())) {
    return "--:--";
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
