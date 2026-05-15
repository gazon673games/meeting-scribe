/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

const client = vi.hoisted(() => ({
  request: vi.fn()
}));

vi.mock("../../../src/shared/api/meetingScribeClient", () => ({
  meetingScribeClient: client
}));

import { SystemAudioPicker, normalizeProcessGroups } from "../../../src/features/audio-inputs/SystemAudioPicker";

describe("SystemAudioPicker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 1024 });
    Object.defineProperty(window, "innerHeight", { configurable: true, value: 768 });
    client.request.mockResolvedValue({
      groups: [
        {
          id: "speakers",
          label: "Speakers",
          sessions: [{ id: "proc:1", label: "Browser", streams: 2 }]
        }
      ]
    });
  });

  test("normalizes grouped and flat process catalog results", () => {
    expect(normalizeProcessGroups({ sessions: [{ id: "app", label: "App" }, { label: "missing" }] })).toEqual([
      {
        id: "active-apps",
        label: "Active applications",
        sessions: [
          expect.objectContaining({
            id: "app",
            fullLabel: "Active applications / App",
            groupLabel: "Active applications"
          })
        ]
      }
    ]);
    expect(normalizeProcessGroups({ groups: [{ id: "empty", sessions: [] }] })).toEqual([]);
  });

  test("loads applications, selects devices, refreshes, and closes on escape", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <SystemAudioPicker
        devices={[{ id: "loop", label: "Default Speakers" }]}
        selected={{ id: "loop", label: "Default Speakers" }}
        selectedId="loop"
        onSelect={onSelect}
      />
    );

    await user.click(screen.getByRole("button", { name: /Default Speakers/i }));
    expect(await screen.findByText("Browser")).toBeInTheDocument();
    expect(client.request).toHaveBeenCalledWith("list_process_sessions");

    await user.click(screen.getByRole("button", { name: /Browser2/ }));
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ id: "proc:1", label: "Browser" }));

    await user.click(screen.getByRole("button", { name: /Default Speakers/i }));
    await user.click(screen.getByTitle("Refresh app audio sources"));
    await waitFor(() => expect(client.request).toHaveBeenCalledTimes(2));

    await user.keyboard("{Escape}");
    expect(screen.queryByText("Applications")).not.toBeInTheDocument();
  });

  test("shows external loading and error states without making its own request", async () => {
    const user = userEvent.setup();
    render(
      <SystemAudioPicker
        loading
        error="PulseAudio unavailable"
        devices={[]}
        catalog={{ groups: [], sessions: [] }}
        selected={null}
      />
    );

    await user.click(screen.getByRole("button", { name: /No devices found/i }));
    expect(screen.getByText("PulseAudio unavailable")).toBeInTheDocument();
    expect(screen.getByText("No applications with active audio found.")).toBeInTheDocument();
    expect(client.request).not.toHaveBeenCalled();
  });
});
