import { EventEmitter } from 'node:events';
import { createRequire } from 'node:module';

import { afterEach, expect, test, vi } from 'vitest';

const require = createRequire(import.meta.url);
const http = require('node:http');
const https = require('node:https');
const { waitForHttp } = require('../dev-runner.cjs');

const makeRequest = () =>
  Object.assign(new EventEmitter(), {
    setTimeout: vi.fn(),
    destroy: vi.fn(),
  });

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

test.each(['http', 'https'])(
  'uses the %s client for renderer readiness',
  async (protocol) => {
    const client = protocol === 'https' ? https : http;
    const request = makeRequest();
    const get = vi.spyOn(client, 'get').mockReturnValue(request);
    const url = `${protocol}://localhost:5173/`;
    const ready = waitForHttp(url);
    const response = { resume: vi.fn() };

    expect(get).toHaveBeenCalledWith(url, expect.any(Function));
    get.mock.calls[0][1](response);

    await expect(ready).resolves.toBeUndefined();
    expect(response.resume).toHaveBeenCalledOnce();
  },
);

test('does not retry a request error after receiving the response', async () => {
  vi.useFakeTimers();
  const request = makeRequest();
  const get = vi.spyOn(http, 'get').mockReturnValue(request);
  const ready = waitForHttp('http://localhost:5173/');
  get.mock.calls[0][1]({ resume: vi.fn() });
  await ready;

  request.emit('error', new Error('connection ended after response'));

  expect(vi.getTimerCount()).toBe(0);
  expect(get).toHaveBeenCalledOnce();
});
