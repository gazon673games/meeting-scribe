/* @vitest-environment jsdom */
import { act, renderHook } from '@testing-library/react';
import { describe, expect, test, vi } from 'vitest';

import { useModelList } from '../../../../src/features/settings/modelSections/useModelList';

const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((resolveValue, rejectValue) => {
    resolve = resolveValue;
    reject = rejectValue;
  });
  return { promise, resolve, reject };
};

describe('model catalog ownership', () => {
  test('ignores an old directory result and rejects calls after unmount', async () => {
    const first = deferred();
    const second = deferred();
    const loadFirst = vi.fn(() => first.promise);
    const loadSecond = vi.fn(() => second.promise);
    const { result, rerender, unmount } = renderHook(
      ({ loadFn }) => useModelList({ open: true, loadFn }),
      { initialProps: { loadFn: loadFirst } },
    );
    await act(async () => {});
    rerender({ loadFn: loadSecond });
    await act(async () => second.resolve({ models: [{ name: 'new' }] }));
    await act(async () => first.resolve({ models: [{ name: 'old' }] }));
    expect(result.current.models).toEqual([{ name: 'new' }]);
    const load = result.current.load;
    unmount();
    await load();
    expect(loadSecond).toHaveBeenCalledTimes(1);
  });

  test('coalesces active loads and clears the error on a successful retry', async () => {
    const first = deferred();
    const second = deferred();
    const loadFn = vi
      .fn()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const { result } = renderHook(() =>
      useModelList({ open: true, loadFn, refreshToken: 1 }),
    );
    await act(async () => {});
    let original;
    act(() => {
      original = result.current.load();
      expect(result.current.load()).toBe(original);
    });
    expect(loadFn).toHaveBeenCalledTimes(1);
    await act(async () => first.reject(new Error('failed')));
    expect(result.current.error).toBe('failed');
    act(() => {
      result.current.load();
    });
    await act(async () => second.resolve({ models: [] }));
    expect(result.current.error).toBe('');
    expect(result.current.models).toEqual([]);
    expect(loadFn).toHaveBeenCalledTimes(2);
  });
});
