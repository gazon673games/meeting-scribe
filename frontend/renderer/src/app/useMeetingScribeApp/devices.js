import { DEVICE_CACHE_KEY } from "./constants";

export function normalizeDevicesResult(result) {
  return {
    loopback: Array.isArray(result?.loopback) ? result.loopback : [],
    input: Array.isArray(result?.input) ? result.input : [],
    errors: Array.isArray(result?.errors) ? result.errors : []
  };
}

export function readCachedDevices() {
  try {
    const raw = window.localStorage?.getItem(DEVICE_CACHE_KEY);
    return normalizeDevicesResult(raw ? JSON.parse(raw) : null);
  } catch {
    return normalizeDevicesResult(null);
  }
}

export function writeCachedDevices(devices) {
  try {
    window.localStorage?.setItem(DEVICE_CACHE_KEY, JSON.stringify(normalizeDevicesResult(devices)));
  } catch {
    // Device cache is only a startup convenience.
  }
}
