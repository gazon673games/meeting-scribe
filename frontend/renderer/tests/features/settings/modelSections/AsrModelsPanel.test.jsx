/* @vitest-environment jsdom */
import '@testing-library/jest-dom/vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';

import { AsrModelsPanel } from '../../../../src/features/settings/modelSections/AsrModelsPanel';

const request = vi.hoisted(() => vi.fn());
vi.mock('../../../../src/shared/api/meetingScribeClient', () => ({
  meetingScribeClient: { request },
}));

test('ignores metadata from a model directory that has been replaced', async () => {
  let resolveMetadata;
  request.mockImplementation((method) =>
    method === 'list_models'
      ? Promise.resolve({
          models: [{ name: 'model', label: 'Model', cached: true }],
        })
      : new Promise((resolve) => {
          resolveMetadata = resolve;
        }),
  );
  const props = {
    draft: {},
    open: true,
    onChange: vi.fn(),
    asrDir: { modelsDir: 'old' },
  };
  const { rerender } = render(<AsrModelsPanel {...props} />);
  fireEvent.click(screen.getByRole('button', { name: 'Transcription' }));
  fireEvent.click(await screen.findByTitle('Show metadata'));
  rerender(<AsrModelsPanel {...props} asrDir={{ modelsDir: 'new' }} />);
  await act(async () => resolveMetadata({ presentFiles: ['STALE.bin'] }));
  expect(screen.queryByText('STALE.bin')).not.toBeInTheDocument();
  expect(await screen.findByText('Model')).toBeInTheDocument();
});

test.each(['constructor', '__proto__', 'toString'])(
  'opens and closes metadata for a model named %s',
  async (name) => {
    request.mockImplementation((method) =>
      Promise.resolve(
        method === 'list_models'
          ? { models: [{ name, label: 'Model', cached: true }] }
          : { presentFiles: ['weights.bin'] },
      ),
    );
    render(
      <AsrModelsPanel
        draft={{}}
        open={true}
        onChange={vi.fn()}
        asrDir={{ modelsDir: 'models' }}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Transcription' }));
    fireEvent.click(await screen.findByTitle('Show metadata'));
    expect(await screen.findByText('weights.bin')).toBeInTheDocument();
    fireEvent.click(screen.getByTitle('Hide metadata'));
    expect(screen.queryByText('weights.bin')).not.toBeInTheDocument();
    expect(screen.getByTitle('Show metadata')).toBeInTheDocument();
  },
);

test('keeps the reopened metadata request when an earlier request completes', async () => {
  const completions = [];
  request.mockImplementation((method) =>
    method === 'list_models'
      ? Promise.resolve({ models: [{ name: 'model', cached: true }] })
      : new Promise((resolve) => {
          completions.push(resolve);
        }),
  );
  render(
    <AsrModelsPanel
      draft={{}}
      open={true}
      onChange={vi.fn()}
      asrDir={{ modelsDir: 'models' }}
    />,
  );
  fireEvent.click(screen.getByRole('button', { name: 'Transcription' }));
  fireEvent.click(await screen.findByTitle('Show metadata'));
  fireEvent.click(screen.getByTitle('Hide metadata'));
  fireEvent.click(screen.getByTitle('Show metadata'));
  await act(async () => completions[1]({ presentFiles: ['current.bin'] }));
  await act(async () => completions[0]({ presentFiles: ['stale.bin'] }));
  expect(screen.getByText('current.bin')).toBeInTheDocument();
  expect(screen.queryByText('stale.bin')).not.toBeInTheDocument();
});
