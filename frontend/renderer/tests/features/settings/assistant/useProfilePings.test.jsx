/* @vitest-environment jsdom */
import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';

import { useProfilePings } from '../../../../src/features/settings/assistant/useProfilePings';

const client = vi.hoisted(() => ({
  request: vi.fn(),
  onBackendEvent: vi.fn(),
}));
vi.mock('../../../../src/shared/api/meetingScribeClient', () => ({
  meetingScribeClient: client,
}));
const profile = {
  id: 'draft',
  provider: 'openai_local',
  model: 'unsaved-model',
};
let receive;
let unsubscribe;

beforeEach(() => {
  client.request.mockReset().mockResolvedValue({ pending: true });
  unsubscribe = vi.fn();
  client.onBackendEvent.mockImplementation((callback) => {
    receive = callback;
    return unsubscribe;
  });
});

test('keeps a pending ping busy until its correlated completion event', async () => {
  const { result } = renderHook(() => useProfilePings([profile]));
  await act(() => result.current.pingProfile(profile));
  const params = client.request.mock.calls[0][1];
  expect(params.profile).toEqual(profile);
  expect(result.current.pingStates.get('draft').busy).toBe(true);
  act(() =>
    receive({
      type: 'assistant_ping_result',
      profileId: 'draft',
      requestId: 'older',
      ok: true,
    }),
  );
  expect(result.current.pingStates.get('draft').busy).toBe(true);
  act(() =>
    receive({
      type: 'assistant_ping_result',
      profileId: 'draft',
      requestId: params.requestId,
      ok: false,
      message: 'offline',
    }),
  );
  expect(result.current.pingStates.get('draft')).toMatchObject({
    busy: false,
    ok: false,
    message: 'offline',
  });
});

test('invalidates a ping when its draft changes and unsubscribes on unmount', async () => {
  const { result, rerender, unmount } = renderHook(
    ({ profiles }) => useProfilePings(profiles),
    {
      initialProps: { profiles: [profile] },
    },
  );
  await act(() => result.current.pingProfile(profile));
  const params = client.request.mock.calls[0][1];
  rerender({ profiles: [{ ...profile, model: 'replacement' }] });
  act(() =>
    receive({
      type: 'assistant_ping_result',
      profileId: 'draft',
      requestId: params.requestId,
      ok: true,
    }),
  );
  expect(result.current.pingStates.get('draft')).toBeUndefined();
  unmount();
  expect(unsubscribe).toHaveBeenCalledOnce();
});

test.each(['constructor', '__proto__'])(
  'keeps a stable ping record for profile ID %s',
  async (id) => {
    const selected = { ...profile, id };
    const { result } = renderHook(() => useProfilePings([selected]));
    await act(() => result.current.pingProfile(selected));
    const pending = result.current.pingStates.get(id);
    const params = client.request.mock.lastCall[1];
    act(() =>
      receive({
        type: 'assistant_ping_result',
        profileId: id,
        requestId: params.requestId,
        ok: true,
        ts: 100,
      }),
    );
    const completed = result.current.pingStates.get(id);
    expect(Object.keys(completed)).toEqual(Object.keys(pending));
    expect(completed).toMatchObject({ busy: false, ok: true, ts: 100 });
  },
);
