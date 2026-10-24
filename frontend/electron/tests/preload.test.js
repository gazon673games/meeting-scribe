import { createRequire } from 'node:module';

import { describe, expect, test, vi } from 'vitest';

const require = createRequire(import.meta.url);
const { exposeMeetingScribeApi } = require('../preload-api.cjs');

describe('electron preload entry', () => {
  test('exposes the API through the isolated context bridge', async () => {
    const ipcRenderer = { invoke: vi.fn() };
    const contextBridge = { exposeInMainWorld: vi.fn() };
    const api = exposeMeetingScribeApi({ contextBridge, ipcRenderer });

    expect(contextBridge.exposeInMainWorld).toHaveBeenCalledWith(
      'meetingScribe',
      api,
    );
    await api.request('ping', { value: 1 });
    expect(ipcRenderer.invoke).toHaveBeenCalledWith('backend:request', 'ping', {
      value: 1,
    });
  });
});
