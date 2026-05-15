/* @vitest-environment jsdom */
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { makeSettingsDraft } from "../../../src/entities/settings/model";

const client = vi.hoisted(() => ({
  reloadApp: vi.fn(),
  request: vi.fn(),
  showDebugConsole: vi.fn()
}));

vi.mock("../../../src/shared/api/meetingScribeClient", () => ({
  meetingScribeClient: client
}));

import { useAppActions } from "../../../src/app/useMeetingScribeApp/actions";

function makeHarness(overrides = {}) {
  const store = {
    assistantPing: {},
    config: { ui: {}, asr: {}, codex: {} },
    error: "",
    localLlmStatus: {},
    runtimeNotices: [{ key: "old" }],
    savingSettings: false,
    settingsDirty: false,
    settingsDraft: {
      ...makeSettingsDraft({ ui: { profile: "Realtime" }, asr: {}, codex: {} }),
      outputFile: "out.wav",
      language: "en",
      model: "medium"
    },
    state: { session: { sources: [{ name: "mic", enabled: true }], transcript: [] } },
    ...overrides
  };

  const update = (key) => vi.fn((value) => {
    store[key] = typeof value === "function" ? value(store[key]) : value;
  });
  const setters = {
    refresh: vi.fn(async () => undefined),
    setAssistantPing: update("assistantPing"),
    setConfig: update("config"),
    setError: update("error"),
    setLocalLlmStatus: update("localLlmStatus"),
    setRuntimeNotices: update("runtimeNotices"),
    setSavingSettings: update("savingSettings"),
    setSettingsDirty: update("settingsDirty"),
    setSettingsDraft: update("settingsDraft"),
    setState: update("state")
  };

  const render = (runState = false) =>
    renderHook(() =>
      useAppActions({
        appOptions: { streamingRequiredProfiles: ["Ultra Fast"], profileDefaults: {} },
        config: store.config,
        runState,
        settingsDraft: store.settingsDraft,
        ...setters
      })
    );
  return { render, setters, store };
}

describe("useAppActions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    client.reloadApp.mockResolvedValue({ reloaded: true });
    client.showDebugConsole.mockResolvedValue({ shown: true });
    client.request.mockResolvedValue({});
  });

  test("updates settings, notices, debug helpers, and login errors", async () => {
    const { render, setters, store } = makeHarness();
    const { result } = render();

    act(() => result.current.updateSettings({ profile: "Ultra Fast" }));
    expect(store.settingsDraft).toMatchObject({ profile: "Ultra Fast", streamingEnabled: true });
    expect(store.settingsDirty).toBe(true);

    act(() => result.current.dismissRuntimeNotice("old"));
    expect(store.runtimeNotices).toEqual([]);
    act(() => result.current.clearError());
    expect(store.error).toBe("");

    await act(async () => result.current.reloadApp());
    await act(async () => result.current.openDebugConsole());
    expect(client.reloadApp).toHaveBeenCalled();
    expect(client.showDebugConsole).toHaveBeenCalled();

    client.request.mockResolvedValueOnce({ errorCode: "auth", suggestion: "Login again" });
    await act(async () => result.current.startAssistantLogin("codex"));
    expect(client.request).toHaveBeenCalledWith("start_assistant_login", { providerId: "codex" });
    expect(setters.refresh).toHaveBeenCalled();
    expect(store.error).toBe("Login again");
  });

  test("saves settings and merges backend state snapshots", async () => {
    const { render, store } = makeHarness();
    client.request.mockImplementation(async (method) => {
      if (method === "save_config") return { config: { ui: { lang: "en" } } };
      if (method === "get_state") return { session: { running: true, transcript: [{ id: "a", text: "saved" }] } };
      return {};
    });
    const { result } = render();

    let saved = false;
    await act(async () => {
      saved = await result.current.saveSettings();
    });
    expect(saved).toBe(true);
    expect(store.settingsDirty).toBe(false);
    expect(store.config).toEqual({ ui: { lang: "en" } });
    expect(store.state.session.transcript[0]).toMatchObject({ text: "saved" });

    client.request.mockRejectedValueOnce(new Error("disk full"));
    await act(async () => {
      saved = await result.current.saveSettings();
    });
    expect(saved).toBe(false);
    expect(store.error).toBe("Error: disk full");
  });

  test("runs backend actions and maps direct state updates", async () => {
    const { render, setters, store } = makeHarness();
    client.request.mockImplementation(async (method) => {
      if (method === "start_session") return { running: true, warning: "degraded" };
      if (method === "add_source") return { name: "sys", enabled: true };
      if (method === "set_source_enabled") return { name: "mic", enabled: false };
      if (method === "remove_source") return { name: "sys" };
      return { refreshed: true };
    });
    const { result } = render();

    await act(async () => result.current.runBackendAction("start_session", { language: "en" }));
    expect(store.state.session.running).toBe(true);
    expect(store.error).toBe("degraded");

    await act(async () => result.current.runBackendAction("add_source", {}));
    await act(async () => result.current.runBackendAction("set_source_enabled", {}));
    await act(async () => result.current.runBackendAction("remove_source", {}));
    expect(store.state.session.sources).toEqual([{ name: "mic", enabled: false }]);

    await act(async () => result.current.runBackendAction("list_devices", {}));
    expect(setters.refresh).toHaveBeenCalled();

    await act(async () => result.current.startOrStop());
    expect(client.request).toHaveBeenCalledWith("start_session", expect.objectContaining({ language: "en" }));
  });

  test("handles assistant ping and local model lifecycle failures", async () => {
    const { render, store } = makeHarness();
    client.request.mockRejectedValueOnce(new Error("offline"));
    const { result } = render();

    act(() => result.current.pingAssistantProvider("codex", "fast"));
    expect(store.assistantPing).toMatchObject({ busy: true, providerId: "codex", profileId: "fast" });
    await waitFor(() => expect(store.assistantPing).toMatchObject({ busy: false, ok: false, message: "Error: offline" }));

    client.request.mockRejectedValueOnce(new Error("no llama"));
    act(() => result.current.startLocalModel("local"));
    expect(store.localLlmStatus.local).toEqual({ state: "starting", message: "" });
    await waitFor(() => expect(store.localLlmStatus.local).toEqual({ state: "error", message: "Failed to start" }));

    client.request.mockResolvedValueOnce({});
    await act(async () => result.current.stopLocalModel("local"));
    expect(store.localLlmStatus.local).toEqual({ state: "stopped", message: "" });
  });

  test("stops an active session with current draft params", async () => {
    const { render } = makeHarness();
    const { result } = render(true);

    await act(async () => result.current.startOrStop());
    expect(client.request).toHaveBeenCalledWith("stop_session", {
      runOfflinePass: false,
      outputFile: "out.wav",
      language: "en",
      model: "medium"
    });
  });
});
