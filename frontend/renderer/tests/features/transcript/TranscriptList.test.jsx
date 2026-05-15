/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { TranscriptList } from "../../../src/features/transcript/TranscriptList";

describe("TranscriptList", () => {
  test("renders transcript lines with speaker labels", () => {
    render(
      <TranscriptList
        lines={[
          { id: "1", ts: 1, stream: "mic", speaker: "Me", text: "hello from mic" },
          { id: "2", ts: 2, stream: "desktop", speaker: "Remote", text: "hello from desktop" }
        ]}
        running={false}
      />
    );

    expect(screen.getByText("Me")).toBeInTheDocument();
    expect(screen.getByText("Remote")).toBeInTheDocument();
    expect(screen.getByText("hello from mic")).toBeInTheDocument();
    expect(screen.getByText("hello from desktop")).toBeInTheDocument();
  });

  test("shows empty and typing states", () => {
    const { rerender } = render(<TranscriptList lines={[]} running={false} />);

    expect(screen.getByText("Transcript stream")).toBeInTheDocument();

    rerender(<TranscriptList lines={[]} running />);

    expect(screen.queryByText("Transcript stream")).not.toBeInTheDocument();
    expect(screen.getByText("ASR")).toBeInTheDocument();
  });
});
