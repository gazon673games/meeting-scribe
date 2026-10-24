/* @vitest-environment jsdom */
import { afterEach, describe, expect, test, vi } from 'vitest';

import {
  readJsonStorage,
  readStorage,
  removeStorage,
  writeStorage,
} from '../../../src/shared/lib/browserStorage';

afterEach(() => vi.restoreAllMocks());

describe('browser storage boundary', () => {
  test('recovers unavailable storage for reads, writes and resets', () => {
    vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError');
    });
    expect(readStorage('key')).toBeNull();
    expect(writeStorage('key', 'value')).toBe(false);
    expect(removeStorage('key')).toBe(false);
  });

  test('preserves zero JSON values and recovers malformed stored input', () => {
    writeStorage('storage-zero', '0');
    expect(readJsonStorage('storage-zero', 10)).toBe(0);
    writeStorage('storage-zero', '{');
    expect(readJsonStorage('storage-zero', { fallback: true })).toEqual({
      fallback: true,
    });
    removeStorage('storage-zero');
  });

  test('does not hide programming errors during persistence', () => {
    vi.spyOn(window, 'localStorage', 'get').mockReturnValue({
      setItem: () => {
        throw new TypeError('programming error');
      },
    });
    expect(() => writeStorage('key', 'value')).toThrow('programming error');
  });
});
