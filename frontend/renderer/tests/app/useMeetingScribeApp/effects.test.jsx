/* @vitest-environment jsdom */
import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const client = vi.hoisted(() => ({ request: vi.fn(), resourceUsage: vi.fn() }));
vi.mock('../../../src/shared/api/meetingScribeClient', () => ({
  meetingScribeClient: client,
}));

import {
  useResourceUsagePolling,
  useRuntimeStatePolling,
} from '../../../src/app/useMeetingScribeApp/effects';

const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
};

beforeEach(() => {
  vi.useFakeTimers();
  vi.resetAllMocks();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

test('keeps at most one runtime poll pending and resumes after completion', async () => {
  const pending = deferred();
  client.request.mockReturnValueOnce(pending.promise).mockResolvedValue({});
  const setState = vi.fn();
  renderHook(() =>
    useRuntimeStatePolling({
      isRunning: true,
      setState,
      setError: vi.fn(),
    }),
  );

  await act(() => vi.advanceTimersByTimeAsync(1800));
  expect(client.request).toHaveBeenCalledOnce();
  await act(async () => {
    pending.resolve({});
  });
  expect(setState).toHaveBeenCalledOnce();
  await act(() => vi.advanceTimersByTimeAsync(600));
  expect(client.request).toHaveBeenCalledTimes(2);
});

test.each(['resolve', 'reject'])(
  'ignores a late runtime %s after cleanup',
  async (settle) => {
    const pending = deferred();
    client.request.mockReturnValue(pending.promise);
    const setState = vi.fn();
    const setError = vi.fn();
    const view = renderHook(() =>
      useRuntimeStatePolling({
        isRunning: true,
        setState,
        setError,
      }),
    );
    await act(() => vi.advanceTimersByTimeAsync(600));
    view.unmount();

    await act(async () => {
      pending[settle](new Error('late'));
    });

    expect(setState).not.toHaveBeenCalled();
    expect(setError).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  },
);

test('keeps one resource request pending and ignores completion after cleanup', async () => {
  const pending = deferred();
  client.resourceUsage.mockReturnValue(pending.promise);
  const setResourceUsage = vi.fn();
  const view = renderHook(() =>
    useResourceUsagePolling({
      fastPolling: true,
      setResourceUsage,
    }),
  );
  await act(() => vi.advanceTimersByTimeAsync(6000));
  expect(client.resourceUsage).toHaveBeenCalledOnce();
  view.unmount();

  await act(async () => {
    pending.resolve({ app: {} });
  });

  expect(setResourceUsage).not.toHaveBeenCalled();
  expect(vi.getTimerCount()).toBe(0);
});
