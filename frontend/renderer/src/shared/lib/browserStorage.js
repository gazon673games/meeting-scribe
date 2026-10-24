const STORAGE_FAILURES = new Set([
  'SecurityError',
  'QuotaExceededError',
  'NS_ERROR_DOM_QUOTA_REACHED',
]);

const accessStorage = (operation, fallback) => {
  try {
    const storage = globalThis.window?.localStorage;
    return storage ? operation(storage) : fallback;
  } catch (error) {
    if (!STORAGE_FAILURES.has(error?.name)) throw error;
    return fallback;
  }
};

export const readStorage = (key) =>
  accessStorage((storage) => storage.getItem(key), null);

export const writeStorage = (key, value) =>
  accessStorage((storage) => {
    storage.setItem(key, value);
    return true;
  }, false);

export const removeStorage = (key) =>
  accessStorage((storage) => {
    storage.removeItem(key);
    return true;
  }, false);

export const readJsonStorage = (key, fallback) => {
  const raw = readStorage(key);
  if (raw === null) return fallback;
  try {
    return JSON.parse(raw);
  } catch (error) {
    if (!(error instanceof SyntaxError)) throw error;
    return fallback;
  }
};
