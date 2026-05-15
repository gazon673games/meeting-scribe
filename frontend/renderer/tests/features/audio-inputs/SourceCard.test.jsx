/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";

import { SourceCard } from "../../../src/features/audio-inputs/SourceCard";

describe("SourceCard", () => {
  test("adds a selected device when no source is active", async () => {
    const user = userEvent.setup();
    const onAdd = vi.fn();
    const { container } = render(
      <SourceCard
        devices={[
          { id: "mic-1", label: "Built-in Mic" },
          { id: "mic-2", label: "USB Mic" }
        ]}
        title="Microphone"
        onAdd={onAdd}
        onRemove={vi.fn()}
        onToggle={vi.fn()}
      />
    );

    await user.selectOptions(container.querySelector("select"), "mic-2");
    expect(onAdd).toHaveBeenCalledWith(expect.objectContaining({ id: "mic-2" }));

    await user.click(container.querySelector(".switch-button"));
    expect(onAdd).toHaveBeenCalledWith(expect.objectContaining({ id: "mic-2" }));
  });

  test("replaces, toggles, and removes an existing source", async () => {
    const user = userEvent.setup();
    const props = {
      devices: [
        { id: "loop-1", label: "Speakers" },
        { id: "loop-2", label: "Headphones", fullLabel: "System / Headphones" }
      ],
      removable: true,
      source: { name: "loop", label: "Speakers", enabled: true, level: 92 },
      title: "System Audio",
      onAdd: vi.fn(),
      onRemove: vi.fn(),
      onToggle: vi.fn()
    };
    const { container } = render(<SourceCard {...props} />);

    expect(container.querySelector(".audio-meter > div")).toHaveClass("hot");
    await user.selectOptions(container.querySelector("select"), "loop-2");
    expect(props.onRemove).toHaveBeenCalledWith(props.source);
    expect(props.onAdd).toHaveBeenCalledWith(expect.objectContaining({ id: "loop-2" }));

    await user.click(container.querySelector(".switch-button"));
    expect(props.onToggle).toHaveBeenCalledWith(props.source);

    await user.click(screen.getByRole("button", { name: /Remove System Audio/i }));
    expect(props.onRemove).toHaveBeenCalledTimes(2);
  });

  test("renders static current source when the device picker has no alternatives", () => {
    render(
      <SourceCard
        devices={[]}
        source={{ name: "ghost", label: "Disconnected Mic", enabled: true, level: 20 }}
        title="Microphone"
        onAdd={vi.fn()}
        onRemove={vi.fn()}
        onToggle={vi.fn()}
      />
    );

    expect(within(screen.getByText("Disconnected Mic").closest(".source-static-value")).getByText("Disconnected Mic")).toBeInTheDocument();
  });
});
