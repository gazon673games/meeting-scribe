/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { makeSettingsDraft } from "../../../src/entities/settings/model";
import { SettingsDialogButton } from "../../../src/features/settings/SettingsDialogButton";

const client = vi.hoisted(() => ({
  request: vi.fn()
}));

vi.mock("../../../src/shared/api/meetingScribeClient", () => ({
  meetingScribeClient: client
}));

function makeDraft() {
  return {
    ...makeSettingsDraft({
      ui: {
        screen_capture_protection: true,
        per_process_audio: true,
        model: "large-v3",
        profile: "Quality",
        theme: "dark"
      },
      asr: {
        device: "cuda",
        compute_type: "float16",
        diarization_enabled: true,
        diar_backend: "sherpa_onnx",
        diar_sherpa_embedding_model_path: "models/diar/speaker.onnx",
        diar_sherpa_provider: "cpu"
      },
      codex: {
        enabled: true,
        selected_profile: "codex_fast",
        profiles: [
          { id: "codex_fast", label: "Codex Fast", provider: "codex", model: "gpt-5.3-codex" },
          { id: "local_api", label: "Local API", provider: "openai_local", model: "qwen-local" }
        ]
      },
      models: { cache_dir: "models", use_proxy: true, proxy: "http://proxy.test:8080" }
    }),
    asrModelsSubdir: "asr",
    diarModelsSubdir: "diar",
    llmModelsSubdir: "llm"
  };
}

function renderSettings(overrides = {}) {
  const props = {
    capabilities: { perProcessAudio: true, screenCaptureProtection: true },
    dirty: true,
    draft: makeDraft(),
    hardware: {
      cpu: { name: "Ryzen", logicalCores: 16 },
      memory: { totalBytes: 32 * 1024 * 1024 * 1024 },
      gpus: [{ name: "RTX", memoryUsedMiB: 512, memoryTotalMiB: 8192, gpuUtilizationPct: 7 }]
    },
    locked: false,
    options: {
      asrDevices: ["cuda", "cpu"],
      asrModels: ["large-v3", "medium"],
      computeTypes: ["float16", "int8"],
      diarizationBackends: ["sherpa_onnx", "online"],
      diarizationProviders: ["cpu", "cuda"],
      overloadStrategies: ["drop_old", "block"]
    },
    saving: false,
    onAsrChange: vi.fn(),
    onChange: vi.fn(),
    onReloadApp: vi.fn(async () => ({ reloaded: true })),
    onSave: vi.fn(async () => true),
    ...overrides
  };
  render(<SettingsDialogButton {...props} />);
  return props;
}

describe("SettingsDialogButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    client.request.mockImplementation(async (method) => {
      if (method === "list_models") {
        return { models: [{ name: "large-v3", label: "Large V3", cached: true, compatible: true, bytes: 1000 }] };
      }
      if (method === "list_diarization_models") {
        return {
          models: [
            {
              name: "speaker.onnx",
              label: "Speaker ONNX",
              cached: true,
              compatible: true,
              backend: "sherpa_onnx",
              provider: "cpu",
              path: "models/diar/speaker.onnx"
            }
          ]
        };
      }
      if (method === "list_llm_models") {
        return {
          models: [
            { name: "qwen.gguf", label: "Qwen GGUF", cached: true, modelAlias: "qwen-local", path: "models/llm/qwen.gguf" }
          ]
        };
      }
      if (method === "model_metadata") {
        return { name: "large-v3", compatible: true, totalBytes: 2048, presentFiles: ["model.bin"] };
      }
      if (method === "ping_assistant_provider") {
        return { ok: true, message: "ready" };
      }
      return {};
    });
  });

  test("opens settings, edits sections, loads model lists, and saves before refresh", async () => {
    const user = userEvent.setup();
    const props = renderSettings();

    await user.click(screen.getByTitle("Settings"));
    expect(screen.getByRole("heading", { name: "Runtime & ASR" })).toBeInTheDocument();
    expect(screen.getByText("Ryzen")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Screen Share Protection/i }));
    expect(props.onChange).toHaveBeenCalledWith({ screenCaptureProtection: false });

    await user.click(screen.getByRole("button", { name: /^Proxy$/i }));
    await user.click(screen.getByRole("button", { name: /Enable Proxy/i }));
    expect(props.onChange).toHaveBeenCalledWith({ assistantProxyEnabled: true });

    await user.click(screen.getByRole("button", { name: /^Assistant$/i }));
    await user.click(screen.getByRole("button", { name: /^Ping$/i }));
    await waitFor(() => expect(client.request).toHaveBeenCalledWith("ping_assistant_provider", {
      providerId: "codex",
      profileId: "codex_fast"
    }));

    await user.click(screen.getByRole("button", { name: /ASR Runtime/i }));
    await user.selectOptions(screen.getByDisplayValue("cuda"), "cpu");
    expect(props.onChange).toHaveBeenCalledWith({ device: "cpu" });

    await user.click(screen.getByRole("button", { name: /^Speaker ID$/i }));
    await user.click(screen.getByRole("button", { name: /Identify Speakers/i }));
    expect(props.onChange).toHaveBeenCalledWith({ diarizationEnabled: false });

    await user.click(screen.getByRole("button", { name: /^Models$/i }));
    await waitFor(() => expect(client.request).toHaveBeenCalledWith("list_models", { modelsDir: "models/asr" }));
    await user.click(screen.getByRole("button", { name: /^Transcription$/i }));
    expect(await screen.findByText("Large V3")).toBeInTheDocument();

    const asrRow = screen.getByText("Large V3").closest(".model-row-shell");
    await user.click(within(asrRow).getByTitle("Show metadata"));
    expect(await screen.findByText("model.bin")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^Language Models$/i }));
    expect(await screen.findByText("Qwen GGUF")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Save & Refresh/i }));
    expect(props.onSave).toHaveBeenCalled();
    expect(props.onReloadApp).toHaveBeenCalled();
  });
});
