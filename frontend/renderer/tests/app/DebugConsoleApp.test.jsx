/* @vitest-environment jsdom */
import '@testing-library/jest-dom/vitest';

import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

const client = vi.hoisted(() => ({
  callback: null,
  onBackendEvent: vi.fn(),
  recentBackendEvents: vi.fn(),
  resourceUsage: vi.fn(),
  status: vi.fn(),
}));

vi.mock('../../src/shared/api/meetingScribeClient', () => ({
  meetingScribeClient: client,
}));

import { DebugConsoleApp } from '../../src/app/DebugConsoleApp';

describe('DebugConsoleApp', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    client.callback = null;
    client.onBackendEvent.mockImplementation((callback) => {
      client.callback = callback;
      return vi.fn();
    });
    client.recentBackendEvents.mockResolvedValue([
      { type: 'asr_started', ts: 3, model: 'large-v3' },
      {
        type: 'diar_debug',
        ts: 2,
        stream: 'mic',
        speaker: 'S1',
        best_sim: 0.91,
        n_speakers_window: 2,
      },
      {
        type: 'transcript_speaker_update',
        ts: 1,
        stream: 'mic',
        speaker: 'S1',
        t_start: 0,
        t_end: 1,
      },
      { type: 'backend_stderr', ts: 4, text: 'Traceback: failed' },
    ]);
    client.status.mockResolvedValue({
      ready: true,
      running: true,
      lastError: '',
    });
    client.resourceUsage.mockResolvedValue({
      app: { backendPid: 123, memoryBytes: 1024 * 1024 },
      system: {},
      gpus: [
        {
          name: 'NVIDIA',
          gpuUtilizationPct: 91,
          memoryUsedMiB: 512,
          memoryTotalMiB: 4096,
        },
      ],
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  test('retains newer live events when a full history arrives later', async () => {
    let resolveHistory;
    client.recentBackendEvents.mockReturnValue(
      new Promise((resolve) => {
        resolveHistory = resolve;
      }),
    );
    render(<DebugConsoleApp />);
    act(() => client.callback({ type: 'marker', ts: 601, text: 'live event' }));
    await act(async () => {
      resolveHistory(
        Array.from({ length: 600 }, (_, index) => ({
          type: 'marker',
          ts: index + 1,
          text: `history #${index}`,
        })),
      );
    });
    expect(screen.getByText('600 events')).toBeInTheDocument();
    expect(screen.getByText(/text=live event/)).toBeInTheDocument();
    expect(screen.queryByText(/text=history #0$/)).not.toBeInTheDocument();
  });

  test('waits for the current runtime poll before starting another', async () => {
    vi.useFakeTimers();
    let resolveStatus;
    client.status.mockReturnValue(
      new Promise((resolve) => {
        resolveStatus = resolve;
      }),
    );
    const view = render(<DebugConsoleApp />);
    await act(() => vi.advanceTimersByTimeAsync(6000));
    expect(client.status).toHaveBeenCalledTimes(1);
    expect(client.resourceUsage).toHaveBeenCalledTimes(1);
    await act(async () => resolveStatus({ ready: true, running: true }));
    await act(() => vi.advanceTimersByTimeAsync(2000));
    expect(client.status).toHaveBeenCalledTimes(2);
    view.unmount();
  });

  test('treats inherited object keys as ordinary unknown events', async () => {
    const types = ['constructor', 'toString', '__proto__'];
    client.recentBackendEvents.mockResolvedValue(
      types.map((type, index) => ({
        type,
        ts: index + 1,
        message: `Unknown event ${type}`,
      })),
    );
    render(<DebugConsoleApp />);
    expect(await screen.findByText('3 events')).toBeInTheDocument();
    for (const type of types) {
      expect(
        screen.getByText(`Unknown event ${type}`, { exact: false }),
      ).toBeInTheDocument();
    }
  });

  test('renders runtime cards, recent events, live events, and clear action', async () => {
    const user = userEvent.setup();
    render(<DebugConsoleApp />);

    expect(
      screen.getByRole('heading', { name: 'ASR / Speaker ID Console' }),
    ).toBeInTheDocument();
    expect(await screen.findByText('4 events')).toBeInTheDocument();
    expect(screen.getByText('large-v3')).toBeInTheDocument();
    expect(screen.getByText(/NVIDIA memory 512\/4096 MiB/)).toBeInTheDocument();

    client.callback?.({
      type: 'error',
      ts: 5,
      where: 'asr.decode',
      error: 'decode failed',
    });
    expect(
      await screen.findByText('asr.decode: decode failed'),
    ).toBeInTheDocument();

    await user.click(screen.getByTitle('Clear visible logs'));
    await waitFor(() =>
      expect(screen.getByText('No backend events yet.')).toBeInTheDocument(),
    );
  });
});
