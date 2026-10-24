/* @vitest-environment jsdom */
import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';

import { applyAsrMetrics } from '../../../src/app/useMeetingScribeApp/sessionState';
import { ProcessingStats } from '../../../src/features/processing/ProcessingStats';

test('shows the measured device, queue wait and inference speed from backend metrics', () => {
  const state = applyAsrMetrics(
    { session: {} },
    {
      device: 'cpu',
      avg_latency_s: 0.16,
      queue_wait_s: 0.05,
      rtf: 0.28,
      seg_qsize: 2,
    },
  );
  expect(state.session.asrMetrics.queueSize).toBe(2);
  render(
    <ProcessingStats
      asrMetrics={state.session.asrMetrics}
      draft={{ device: 'cuda' }}
      session={{}}
      summary={{ model: 'Nemotron' }}
    />,
  );
  expect(screen.getByText('cpu')).toBeInTheDocument();
  expect(screen.getByText('0.05s')).toBeInTheDocument();
  expect(screen.getByText('0.28')).toBeInTheDocument();
});
