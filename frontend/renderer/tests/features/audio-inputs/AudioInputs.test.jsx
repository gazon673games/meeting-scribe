/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

const client = vi.hoisted(() => ({
  request: vi.fn()
}));

vi.mock("../../../src/shared/api/meetingScribeClient", () => ({
  meetingScribeClient: client
}));

import { AudioInputs } from "../../../src/features/audio-inputs/AudioInputs";

function renderInputs(overrides = {}) {
  const props = {
    capabilities: { perProcessAudio: true },
    devices: {
      input: [{ id: "mic-1", label: "Built-in Mic" }],
      loopback: [{ id: "loop-1", label: "Speakers" }],
      errors: []
    },
    disabled: false,
    perProcessAudio: true,
    sources: [
      { kind: "input", name: "mic", label: "Built-in Mic", enabled: true, level: 35 },
      { kind: "loopback", name: "system", label: "Speakers", enabled: false, level: 0 },
      { kind: "process", name: "browser", label: "Browser", enabled: true, level: 62 },
      { kind: "virtual", name: "virtual", label: "Virtual Cable", enabled: true, level: 10 }
    ],
    onAdd: vi.fn(),
    onRemove: vi.fn(),
    onToggle: vi.fn(),
    ...overrides
  };
  const view = render(<AudioInputs {...props} />);
  return { ...props, container: view.container };
}

describe("AudioInputs", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 1200 });
    Object.defineProperty(window, "innerHeight", { configurable: true, value: 800 });
    client.request.mockResolvedValue({
      groups: [{ id: "apps", label: "Apps", sessions: [{ id: "proc-1", label: "Browser", streams: 1 }] }]
    });
  });

  test("renders primary, extra, and picker-backed sources", async () => {
    const user = userEvent.setup();
    const props = renderInputs({ devices: { input: [{ id: "mic-1", label: "Built-in Mic" }], loopback: [{ id: "loop-1", label: "Speakers" }], errors: ["WASAPI unavailable"] } });

    expect(screen.getByRole("heading", { name: "Audio Inputs" })).toBeInTheDocument();
    expect(screen.getAllByText("Virtual Cable").length).toBeGreaterThan(0);
    expect(screen.getByText("WASAPI unavailable")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Speakers/i }));
    expect(client.request).toHaveBeenCalledWith("list_process_sessions");
    expect((await screen.findAllByText("Browser")).length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: /^Browser$/ }));
    expect(props.onRemove).toHaveBeenCalledWith(expect.objectContaining({ name: "system" }));
    expect(props.onAdd).toHaveBeenCalledWith(expect.objectContaining({ id: "proc-1" }));
  });

  test("loads process catalog through Add Source and caches repeated opens", async () => {
    const user = userEvent.setup();
    const props = renderInputs({ sources: [] });

    await user.click(props.container.querySelector(".add-source summary"));
    await waitFor(() => expect(client.request).toHaveBeenCalledTimes(1));
    const addSelect = props.container.querySelector(".add-source select");
    await waitFor(() => expect(addSelect.querySelector('optgroup[label="Applications / Apps"]')).not.toBeNull());

    await user.selectOptions(addSelect, "proc-1");
    await user.click(screen.getByRole("button", { name: /^Add$/i }));
    expect(props.onAdd).toHaveBeenCalledWith(expect.objectContaining({ id: "proc-1", label: "Browser" }));

    await user.click(props.container.querySelector(".add-source summary"));
    await user.click(props.container.querySelector(".add-source summary"));
    expect(client.request).toHaveBeenCalledTimes(1);
  });

  test("falls back to regular system devices when per-process audio is disabled", async () => {
    const user = userEvent.setup();
    const props = renderInputs({
      perProcessAudio: false,
      sources: [{ kind: "loopback", name: "system", label: "Speakers", enabled: false, level: 0 }]
    });

    await user.click(within(screen.getByText("System Audio").closest(".source-card")).getByRole("button"));
    expect(props.onToggle).toHaveBeenCalledWith(expect.objectContaining({ name: "system" }));
    expect(client.request).not.toHaveBeenCalled();
  });

  test("keeps source switches available while device selection is locked during capture", async () => {
    const user = userEvent.setup();
    const props = renderInputs({ sourceSelectionLocked: true });
    const micCard = screen.getByText("Microphone").closest(".source-card");
    const virtualRemove = screen.getByRole("button", { name: /Remove Virtual Cable/i });
    const virtualCard = virtualRemove.closest(".source-card");

    expect(within(micCard).getByRole("combobox")).toBeDisabled();
    expect(virtualRemove).toBeDisabled();

    await user.click(micCard.querySelector(".switch-button"));
    await user.click(virtualCard.querySelector(".switch-button"));

    expect(props.onToggle).toHaveBeenNthCalledWith(1, expect.objectContaining({ name: "mic" }));
    expect(props.onToggle).toHaveBeenNthCalledWith(2, expect.objectContaining({ name: "virtual" }));
  });
});
