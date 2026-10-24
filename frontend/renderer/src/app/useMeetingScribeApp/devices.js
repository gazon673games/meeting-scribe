import { DEVICE_CACHE_KEY } from './constants';

export const normalizeDevicesResult = (result) => ({
  loopback: Array.isArray(result?.loopback) ? result.loopback : [],
  input: Array.isArray(result?.input) ? result.input : [],
  errors: Array.isArray(result?.errors) ? result.errors : [],
});

export const readCachedDevices = () => {
  try {
    const raw = window.localStorage?.getItem(DEVICE_CACHE_KEY);
    return normalizeDevicesResult(raw ? JSON.parse(raw) : null);
  } catch (error) {
    if (
      !(error instanceof SyntaxError) &&
      !(error instanceof window.DOMException)
    ) {
      throw error;
    }
    console.warn('[devices] failed to read device cache', error);
    return normalizeDevicesResult(null);
  }
};

export const writeCachedDevices = (devices) => {
  try {
    window.localStorage?.setItem(
      DEVICE_CACHE_KEY,
      JSON.stringify(normalizeDevicesResult(devices)),
    );
  } catch (error) {
    if (!(error instanceof window.DOMException)) throw error;
    console.warn('[devices] failed to write device cache', error);
  }
};
