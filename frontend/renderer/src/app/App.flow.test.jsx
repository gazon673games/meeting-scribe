/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

const client = vi.hoisted(() => ({
  callbacks: [],
  onBackendEvent: vi.fn(),
  recentBackendEvents: vi.fn(),
  reloadApp: vi.fn(),
  request: vi.fn(),
  resourceUsage: vi.fn(),
  setContentProtection: vi.fn(),
  showDebugConsole: vi.fn(),
  status: vi.fn()
}));

vi.mock("../shared/api/meetingScribeClient", () => ({
  meetingScribeClient: client
}));

import { App } from "./App";

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function installResizeObserver() {
  globalThis.ResizeObserver = class {
    observe() {}
    disconnect() {}
  };
}

function initialState() {
  return {
    assistant: {
      enabled: true,
      busy: false,
      selectedProfileId: "fast",
      profiles: [{ id: "fast", label: "Fast", providerId: "codex" }]
    },
    capabilities: {
      assistant: true,
      perProcessAudio: false,
      screenCaptureProtection: true,
      sessionControl: true,
      sourceControl: true
    },
    configSummary: { language: "en" },
    hardware: {},
    options: {
      asrProfiles: ["Realtime"],
      languages: ["en"],
      profileDefaults: {}
    },
    session: {
      asrMetrics: { avgLatencyS: 0 },
      offlinePass: {},
      running: false,
      sources: [{ name: "mic", kind: "input", label: "Mic", enabled: true }],
      state: "idle",
      transcript: []
    }
  };
}

function config() {
  return {
    ui: {
      asr_enabled: true,
      lang: "en",
      model: "medium",
      profile: "Realtime",
      theme: "dark"
    },
    asr: {},
    codex: {
      enabled: true,
      selected_profile: "fast",
      profiles: [{ id: "fast", label: "Fast", provider: "codex" }]
    }
  };
}

function wireFakeBackend() {
  let state = initialState();
  const emit = (event) => client.callbacks.forEach((callback) => callback(event));

  client.callbacks = [];
  client.status.mockResolvedValue({ ready: true, running: true, lastError: "" });
  client.resourceUsage.mockResolvedValue({ app: {}, system: {}, gpus: [] });
  client.setContentProtection.mockResolvedValue({ enabled: false });
  client.onBackendEvent.mockImplementation((callback) => {
    client.callbacks.push(callback);
    return () => {
      client.callbacks = client.callbacks.filter((item) => item !== callback);
    };
  });
  client.request.mockImplementation(async (method, params = {}) => {
    if (method === "get_config") return config();
    if (method === "get_state" || method === "get_runtime_state") return clone(state);
    if (method === "list_devices") return { input: [], loopback: [], errors: [] };
    if (method === "start_session") {
      state = { ...state, session: { ...state.session, running: true, state: "recording" } };
      queueMicrotask(() => {
        const line = { id: "line-1", ts: 1, stream: "mic", speaker: "Me", text: "hello from fake backend" };
        state = {
          ...state,
          session: { ...state.session, transcript: [line] }
        };
        emit({ type: "transcript_line", ...line });
      });
      return clone(state.session);
    }
    if (method === "invoke_assistant") {
      state = {
        ...state,
        assistant: {
          ...state.assistant,
          lastResponse: {
            ok: true,
            text: `assistant handled ${params.action || params.requestText}`,
            ts: 10,
            provider: "codex"
          }
        }
      };
      return clone(state.assistant.lastResponse);
    }
    return {};
  });
}

describe("Meeting Scribe app flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    installResizeObserver();
    window.localStorage.clear();
    wireFakeBackend();
  });

  test("starts a session, receives transcript text, and invokes the assistant", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: /^start$/i }));

    expect(await screen.findByText("hello from fake backend")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /answer latest/i }));

    await waitFor(() => {
      expect(client.request).toHaveBeenCalledWith("invoke_assistant", {
        action: "answer",
        profileId: "fast"
      });
    });
    expect(await screen.findByText("assistant handled answer")).toBeInTheDocument();
  });
});
