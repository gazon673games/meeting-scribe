/* @vitest-environment jsdom */
import '@testing-library/jest-dom/vitest';

import { render, screen } from '@testing-library/react';
import { describe, expect, test } from 'vitest';

import { TranscriptList } from '../../../src/features/transcript/TranscriptList';

describe('TranscriptList', () => {
  test('scrolls growing streaming text while preserving speaker colors after history truncation', () => {
    const first = { id: 'first', ts: 1, speaker: 'First', text: 'old' };
    const streaming = {
      id: 'streaming',
      ts: 2,
      speaker: 'Second',
      text: 'hello',
    };
    const { container, rerender } = render(
      <TranscriptList lines={[first, streaming]} running />,
    );
    const row = screen.getByText('Second').closest('.transcript-line');
    const color = row.className;
    const list = container.querySelector('.transcript-list');
    Object.defineProperty(list, 'scrollHeight', {
      configurable: true,
      value: 300,
    });
    rerender(
      <TranscriptList
        lines={[{ ...streaming, text: 'hello and more words' }]}
        running
      />,
    );
    expect(list.scrollTop).toBe(300);
    expect(
      screen.getByText('Second').closest('.transcript-line').className,
    ).toBe(color);
  });

  test('renders transcript lines with speaker labels', () => {
    render(
      <TranscriptList
        lines={[
          {
            id: '1',
            ts: 1,
            stream: 'mic',
            speaker: 'Me',
            text: 'hello from mic',
          },
          {
            id: '2',
            ts: 2,
            stream: 'desktop',
            speaker: 'Remote',
            text: 'hello from desktop',
          },
        ]}
        running={false}
      />,
    );

    expect(screen.getByText('Me')).toBeInTheDocument();
    expect(screen.getByText('Remote')).toBeInTheDocument();
    expect(screen.getByText('hello from mic')).toBeInTheDocument();
    expect(screen.getByText('hello from desktop')).toBeInTheDocument();
  });

  test('shows empty and typing states', () => {
    const { rerender } = render(<TranscriptList lines={[]} running={false} />);

    expect(screen.getByText('Transcript stream')).toBeInTheDocument();

    rerender(<TranscriptList lines={[]} running />);

    expect(screen.queryByText('Transcript stream')).not.toBeInTheDocument();
    expect(screen.getByText('ASR')).toBeInTheDocument();
  });
});
