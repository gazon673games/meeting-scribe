/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";

import { AssistantProfileSelect } from "../../../src/features/assistant/AssistantProfileSelect";
import { AssistantStats } from "../../../src/features/assistant/AssistantStats";

describe("assistant widgets", () => {
  test("selects assistant profile and shortens model paths", async () => {
    const user = userEvent.setup();
    const onProfileChange = vi.fn();
    render(
      <AssistantProfileSelect
        disabled={false}
        profileId="local"
        profiles={[
          { id: "codex", label: "Codex" },
          { id: "local", label: "Local", model: "C:/models/qwen.gguf" }
        ]}
        onProfileChange={onProfileChange}
      />
    );

    expect(screen.getByRole("option", { name: "Local - qwen" })).toBeInTheDocument();
    await user.selectOptions(screen.getByRole("combobox"), "codex");
    expect(onProfileChange).toHaveBeenCalledWith("codex");
  });

  test("shows assistant response timing and last action", () => {
    render(
      <AssistantStats
        assistant={{ busy: false, lastRequest: { sourceLabel: "Answer" } }}
        response={{ dtS: 2.345, provider: "ollama" }}
        selectedProfile={{ provider: "openai_local", model: "D:/llm/mistral.gguf" }}
      />
    );

    expect(screen.getByText("openai_local")).toBeInTheDocument();
    expect(screen.getByText("mistral")).toBeInTheDocument();
    expect(screen.getByText("2.35s")).toBeInTheDocument();
    expect(screen.getByText("Answer")).toBeInTheDocument();
  });
});
