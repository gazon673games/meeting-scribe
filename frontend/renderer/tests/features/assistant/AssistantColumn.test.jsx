/* @vitest-environment jsdom */
import '@testing-library/jest-dom/vitest';

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, test, vi } from 'vitest';

import { AssistantColumn } from '../../../src/features/assistant/AssistantColumn';

const renderAssistant = (overrides = {}) => {
  const props = {
    assistant: { enabled: true, busy: false, selectedProfileId: 'fast' },
    assistantPing: {},
    contextReady: true,
    disabled: false,
    localLlmStatus: {},
    profiles: [{ id: 'fast', label: 'Fast', providerId: 'codex' }],
    onAuthorize: vi.fn(),
    onInvoke: vi.fn(),
    onPing: vi.fn(),
    onStartLocalModel: vi.fn(),
    onStopLocalModel: vi.fn(),
    ...overrides,
  };
  render(<AssistantColumn {...props} />);
  return props;
};

describe('AssistantColumn', () => {
  test('reconciles a removed selected profile before invoking an action', async () => {
    const props = {
      assistant: { enabled: true, busy: false, selectedProfileId: 'removed' },
      profiles: [{ id: 'removed', label: 'Old', providerId: 'codex' }],
      contextReady: true,
      onInvoke: vi.fn(),
    };
    const { rerender } = render(<AssistantColumn {...props} />);
    rerender(
      <AssistantColumn
        {...props}
        profiles={[{ id: 'current', label: 'Current', providerId: 'codex' }]}
      />,
    );
    await userEvent
      .setup()
      .click(screen.getByRole('button', { name: 'Summarize' }));
    expect(props.onInvoke).toHaveBeenCalledWith(
      expect.objectContaining({ profileId: 'current' }),
    );
  });

  test('keeps Ollama lifecycle external and offers a connection check', () => {
    renderAssistant({
      profiles: [{ id: 'fast', label: 'Ollama', providerId: 'ollama' }],
    });

    expect(screen.queryByRole('button', { name: /start model/i })).toBeNull();
    expect(screen.getByRole('button', { name: /^ping$/i })).toBeEnabled();
  });

  test('invokes quick actions with the selected profile', async () => {
    const user = userEvent.setup();
    const props = renderAssistant();

    await user.click(screen.getByRole('button', { name: /summarize/i }));

    expect(props.onInvoke).toHaveBeenCalledWith({
      action: 'summary',
      profileId: 'fast',
    });
    expect(screen.getAllByText('Summarize').length).toBeGreaterThan(1);
  });

  test('submits custom prompts and clears the input', async () => {
    const user = userEvent.setup();
    const props = renderAssistant();
    const input = screen.getByPlaceholderText('Ask the assistant anything...');

    await user.type(input, 'What changed?');
    await user.click(screen.getByRole('button', { name: /send/i }));

    expect(props.onInvoke).toHaveBeenCalledWith({
      requestText: 'What changed?',
      profileId: 'fast',
      sourceLabel: 'you',
    });
    expect(input).toHaveValue('');
  });
});
