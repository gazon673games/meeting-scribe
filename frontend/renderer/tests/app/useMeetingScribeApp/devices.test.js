/* @vitest-environment jsdom */
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import { DEVICE_CACHE_KEY } from '../../../src/app/useMeetingScribeApp/constants';
import {
  readCachedDevices,
  writeCachedDevices,
} from '../../../src/app/useMeetingScribeApp/devices';

beforeEach(() => {
  window.localStorage.clear();
  vi.spyOn(console, 'warn').mockImplementation(() => undefined);
});

afterEach(() => vi.restoreAllMocks());

test('logs corrupt cached JSON and returns an empty device snapshot', () => {
  window.localStorage.setItem(DEVICE_CACHE_KEY, '{');
  expect(readCachedDevices()).toEqual({ loopback: [], input: [], errors: [] });
  expect(console.warn).toHaveBeenCalledWith(
    '[devices] failed to read device cache',
    expect.any(SyntaxError),
  );
});

test('handles denied storage access and exhausted quota with diagnostics', () => {
  const security = new window.DOMException('denied', 'SecurityError');
  const quota = new window.DOMException('full', 'QuotaExceededError');
  vi.spyOn(window.localStorage, 'getItem').mockImplementation(() => {
    throw security;
  });
  vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
    throw quota;
  });

  expect(readCachedDevices()).toEqual({ loopback: [], input: [], errors: [] });
  expect(() => writeCachedDevices({})).not.toThrow();
  expect(console.warn).toHaveBeenCalledWith(
    '[devices] failed to read device cache',
    security,
  );
  expect(console.warn).toHaveBeenCalledWith(
    '[devices] failed to write device cache',
    quota,
  );
});

test.each(['getItem', 'setItem'])(
  'propagates programming errors from %s',
  (method) => {
    vi.spyOn(window.localStorage, method).mockImplementation(() => {
      throw new TypeError('bug');
    });
    const operation =
      method === 'getItem' ? readCachedDevices : () => writeCachedDevices({});
    expect(operation).toThrow('bug');
  },
);
