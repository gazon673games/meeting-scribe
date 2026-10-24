/* @vitest-environment jsdom */
import { render, waitFor } from '@testing-library/react';
import { expect, test, vi } from 'vitest';

import { DiarModelsPanel } from '../../../../src/features/settings/modelSections/DiarModelsPanel';

const request = vi.hoisted(() => vi.fn());
vi.mock('../../../../src/shared/api/meetingScribeClient', () => ({
  meetingScribeClient: { request },
}));

test('does not auto-select a cached speaker model while runtime settings are locked', async () => {
  request.mockResolvedValue({
    models: [
      {
        name: 'speaker',
        path: '/speaker.onnx',
        cached: true,
        compatible: true,
      },
    ],
  });
  const onChange = vi.fn();
  const props = {
    draft: { diarizationEnabled: true, diarizationBackend: 'online' },
    open: true,
    locked: true,
    onChange,
    diarDir: {},
  };
  const { rerender } = render(<DiarModelsPanel {...props} />);
  await waitFor(() => expect(request).toHaveBeenCalled());
  expect(onChange).not.toHaveBeenCalled();
  rerender(<DiarModelsPanel {...props} locked={false} />);
  await waitFor(() =>
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        diarSherpaEmbeddingModelPath: '/speaker.onnx',
      }),
    ),
  );
});
