/* @vitest-environment jsdom */
import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';

import { LlmModelsPanel } from '../../../../src/features/settings/modelSections/LlmModelsPanel';

const request = vi.hoisted(() => vi.fn());
vi.mock('../../../../src/shared/api/meetingScribeClient', () => ({
  meetingScribeClient: { request },
}));

test('creates distinct local profiles and reuses the selected local profile', async () => {
  request.mockResolvedValue({
    models: [{ name: 'model.gguf', modelAlias: 'local-model' }],
  });
  const onChange = vi.fn();
  const props = {
    draft: {
      assistantSelectedProfileId: 'codex_1',
      assistantProfiles: [
        { id: 'codex_1', provider: 'codex' },
        { id: 'openai_local_3', provider: 'openai_local', model: 'old-model' },
      ],
    },
    onChange,
    open: true,
    llmDir: { modelsDir: 'models' },
  };
  const { rerender } = render(<LlmModelsPanel {...props} />);
  fireEvent.click(screen.getByRole('button', { name: 'Language Models' }));
  const useModel = await screen.findByTitle(
    'Use local-model in selected assistant profile',
  );
  fireEvent.click(useModel);
  const first = onChange.mock.lastCall[0];
  expect(first.assistantSelectedProfileId).toBe('openai_local_4');
  expect(first.assistantProfiles.at(-1)).toMatchObject({
    provider: 'openai_local',
    model: 'local-model',
    offline: true,
  });

  rerender(
    <LlmModelsPanel
      {...props}
      draft={{
        ...first,
        assistantSelectedProfileId: 'codex_1',
      }}
    />,
  );
  fireEvent.click(useModel);
  const second = onChange.mock.lastCall[0];
  const ids = second.assistantProfiles.map((profile) => profile.id);
  expect(new Set(ids).size).toBe(ids.length);
  expect(second.assistantSelectedProfileId).not.toBe(
    first.assistantSelectedProfileId,
  );

  rerender(<LlmModelsPanel {...props} draft={second} />);
  fireEvent.click(useModel);
  const updated = onChange.mock.lastCall[0];
  expect(updated.assistantSelectedProfileId).toBe(
    second.assistantSelectedProfileId,
  );
  expect(updated.assistantProfiles).toHaveLength(ids.length);
});
