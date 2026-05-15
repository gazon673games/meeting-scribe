/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";

import { AssistantColumn } from "./AssistantColumn";

function renderAssistant(overrides = {}) {
  const props = {
    assistant: { enabled: true, busy: false, selectedProfileId: "fast" },
    assistantPing: {},
    contextReady: true,
    disabled: false,
    localLlmStatus: {},
    profiles: [{ id: "fast", label: "Fast", providerId: "codex" }],
    onAuthorize: vi.fn(),
    onInvoke: vi.fn(),
    onPing: vi.fn(),
    onStartLocalModel: vi.fn(),
    onStopLocalModel: vi.fn(),
    ...overrides
  };
  render(<AssistantColumn {...props} />);
  return props;
}

describe("AssistantColumn", () => {
  test("invokes quick actions with the selected profile", async () => {
    const user = userEvent.setup();
    const props = renderAssistant();

    await user.click(screen.getByRole("button", { name: /summarize/i }));

    expect(props.onInvoke).toHaveBeenCalledWith({ action: "summary", profileId: "fast" });
    expect(screen.getAllByText("Summarize").length).toBeGreaterThan(1);
  });

  test("submits custom prompts and clears the input", async () => {
    const user = userEvent.setup();
    const props = renderAssistant();
    const input = screen.getByPlaceholderText("Ask the assistant anything...");

    await user.type(input, "What changed?");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(props.onInvoke).toHaveBeenCalledWith({
      requestText: "What changed?",
      profileId: "fast",
      sourceLabel: "you"
    });
    expect(input).toHaveValue("");
  });
});
