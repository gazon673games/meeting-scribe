/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";

import { AssistantProfileDetails } from "../../../../src/features/settings/assistant/AssistantProfileDetails";
import { AssistantProfilesSidebar } from "../../../../src/features/settings/assistant/AssistantProfilesSidebar";

function renderDetails(profile, overrides = {}) {
  const props = {
    assistantEnabled: true,
    cachedGgufModels: [{ path: "models/qwen.gguf", label: "Qwen GGUF" }],
    cachedLocalModels: ["llama3", "mistral"],
    ping: null,
    profile,
    onDelete: vi.fn(),
    onPing: vi.fn(),
    onUpdate: vi.fn(),
    ...overrides
  };
  render(<AssistantProfileDetails {...props} />);
  return props;
}

describe("AssistantProfileDetails", () => {
  test("edits Codex profile fields and exposes ping/delete actions", async () => {
    const user = userEvent.setup();
    const props = renderDetails(
      {
        id: "codex_fast",
        label: "Codex Fast",
        provider: "codex",
        model: "gpt-5.3-codex",
        reasoning_effort: "low",
        codex_profile: "work",
        prompt: "Be short"
      },
      { ping: { ok: true, message: "ready" } }
    );

    expect(screen.getByText("Codex CLI")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^Ping$/i }));
    expect(props.onPing).toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText("Label"), { target: { value: "Main" } });
    expect(props.onUpdate).toHaveBeenCalledWith({ label: "Main" });

    await user.selectOptions(screen.getByLabelText("Reasoning"), "high");
    expect(props.onUpdate).toHaveBeenCalledWith({ reasoning_effort: "high" });

    fireEvent.change(screen.getByLabelText("Codex Profile"), { target: { value: "default" } });
    expect(props.onUpdate).toHaveBeenCalledWith({ codex_profile: "default" });

    fireEvent.change(screen.getByLabelText("Instructions"), { target: { value: "Be short now" } });
    expect(props.onUpdate).toHaveBeenCalledWith({ prompt: "Be short now" });

    await user.click(screen.getByTitle("Delete profile"));
    expect(props.onDelete).toHaveBeenCalled();
  });

  test("switches local runtimes and model picker modes", async () => {
    const user = userEvent.setup();
    const props = renderDetails({
      id: "local_api",
      label: "Local API",
      provider: "ollama",
      model: "llama3",
      base_url: "http://127.0.0.1:11434",
      temperature: "0.4",
      max_tokens: 2048,
      prompt: ""
    });

    await user.selectOptions(screen.getByLabelText("Model"), "mistral");
    expect(props.onUpdate).toHaveBeenCalledWith({ model: "mistral" });

    await user.selectOptions(screen.getByLabelText("Model"), "__custom__");
    await user.click(screen.getByTitle("Pick from list"));
    expect(props.onUpdate).toHaveBeenCalledWith({ model: "" });

    await user.selectOptions(screen.getByLabelText("Runtime"), "local");
    expect(props.onUpdate).toHaveBeenCalledWith({
      provider: "local",
      base_url: "",
      offline: true,
      gpu_layers: 0,
      context_size: 4096
    });

    fireEvent.change(screen.getByLabelText("Temperature"), { target: { value: "0.8" } });
    expect(props.onUpdate).toHaveBeenCalledWith({ temperature: "0.8" });

    fireEvent.change(screen.getByLabelText("Max Tokens"), { target: { value: "4096x" } });
    expect(props.onUpdate).toHaveBeenCalledWith({ max_tokens: "4096" });
  });
});

describe("AssistantProfilesSidebar", () => {
  test("selects, deletes, and adds assistant profiles", async () => {
    const user = userEvent.setup();
    const props = {
      profiles: [
        { id: "codex", label: "Codex", provider: "codex" },
        { id: "local", label: "GGUF", provider: "local", model: "C:/models/qwen.gguf" }
      ],
      selectedProfileId: "local",
      onSelect: vi.fn(),
      onDelete: vi.fn(),
      onAdd: vi.fn()
    };
    render(<AssistantProfilesSidebar {...props} />);

    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText(/qwen/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Codex CLI/i }));
    expect(props.onSelect).toHaveBeenCalledWith("codex");

    const localRow = screen.getByRole("button", { name: /Local GGUF/i }).closest(".assistant-profile-list-item");
    await user.click(within(localRow).getByTitle("Delete"));
    expect(props.onDelete).toHaveBeenCalledWith(1);

    await user.click(screen.getByRole("button", { name: /^Ollama$/i }));
    expect(props.onAdd).toHaveBeenCalledWith("ollama");
  });
});
