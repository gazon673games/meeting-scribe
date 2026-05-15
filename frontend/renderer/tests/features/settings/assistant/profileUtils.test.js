import { describe, expect, test } from "vitest";

import {
  defaultBaseUrl,
  modelBasename,
  newProfile,
  normalizeProvider,
  runtimeLabel,
  runtimePatch,
  uniqueProfileId
} from "../../../../src/features/settings/assistant/profileUtils";

describe("assistant profile utilities", () => {
  test("normalizes provider labels, defaults, and model display names", () => {
    expect(normalizeProvider("OLLAMA")).toBe("ollama");
    expect(normalizeProvider("unknown")).toBe("codex");
    expect(runtimeLabel("openai_local")).toBe("OpenAI-compatible");
    expect(defaultBaseUrl("openai_local")).toBe("http://127.0.0.1:1234/v1");
    expect(modelBasename("C:\\models\\Qwen.gguf")).toBe("Qwen");
  });

  test("creates local and codex profiles with provider-specific defaults", () => {
    expect(newProfile("codex_1", "codex")).toMatchObject({
      id: "codex_1",
      provider: "codex",
      reasoning_effort: "low",
      offline: false
    });
    expect(newProfile("local_1", "local")).toMatchObject({
      id: "local_1",
      provider: "local",
      gpu_layers: 0,
      context_size: 4096,
      offline: true
    });
  });

  test("builds runtime patches without overwriting custom endpoint values", () => {
    expect(runtimePatch({ provider: "ollama", base_url: "http://custom" }, "openai_local")).toEqual({
      provider: "openai_local",
      base_url: "http://custom",
      offline: true
    });
    expect(runtimePatch({ provider: "codex" }, "local")).toMatchObject({
      provider: "local",
      gpu_layers: 0,
      context_size: 4096
    });
  });

  test("allocates unique profile ids from existing profiles", () => {
    expect(uniqueProfileId([{ id: "ollama_2" }, { id: "ollama_3" }], "ollama")).toBe("ollama_4");
  });
});
