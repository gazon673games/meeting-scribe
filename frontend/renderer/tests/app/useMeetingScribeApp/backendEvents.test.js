import { describe, expect, test, vi } from "vitest";

import { handleBackendEvent } from "../../../src/app/useMeetingScribeApp/backendEvents";

function makeHandlers(initialState = { session: { sources: [], transcript: [] } }) {
  const handlers = {
    assistantPing: {},
    backendStatus: {},
    error: "",
    events: [],
    localLlmStatus: {},
    runtimeNotices: [],
    state: initialState,
    refresh: vi.fn(),
    setAssistantPing: vi.fn((updater) => {
      handlers.assistantPing = typeof updater === "function" ? updater(handlers.assistantPing) : updater;
    }),
    setBackendStatus: vi.fn((updater) => {
      handlers.backendStatus = typeof updater === "function" ? updater(handlers.backendStatus) : updater;
    }),
    setError: vi.fn((value) => {
      handlers.error = value;
    }),
    setEvents: vi.fn((updater) => {
      handlers.events = typeof updater === "function" ? updater(handlers.events) : updater;
    }),
    setLocalLlmStatus: vi.fn((updater) => {
      handlers.localLlmStatus = typeof updater === "function" ? updater(handlers.localLlmStatus) : updater;
    }),
    setRuntimeNotices: vi.fn((updater) => {
      handlers.runtimeNotices = typeof updater === "function" ? updater(handlers.runtimeNotices) : updater;
    }),
    setState: vi.fn((updater) => {
      handlers.state = typeof updater === "function" ? updater(handlers.state) : updater;
    })
  };
  return handlers;
}

describe("backend event handling", () => {
  test("stores recent events and routes transcript, metrics, and runtime notices", () => {
    const handlers = makeHandlers();

    handleBackendEvent({ type: "transcript_line", id: "line", stream: "mic", text: "hello" }, handlers);
    handleBackendEvent({ type: "transcript_line_update", id: "line", speaker: "S1" }, handlers);
    handleBackendEvent({ type: "asr_metrics", avg_latency_s: 1.2, seg_dropped_total: 4 }, handlers);
    handleBackendEvent({ type: "source_error", source: "mic", error: "device gone" }, handlers);

    expect(handlers.events).toHaveLength(4);
    expect(handlers.state.session.transcript[0]).toMatchObject({ text: "hello", speaker: "S1" });
    expect(handlers.state.session.asrMetrics).toMatchObject({ avgLatencyS: 1.2, segDroppedTotal: 4 });
    expect(handlers.runtimeNotices[0]).toMatchObject({ title: "Audio Source", message: "mic: device gone" });
  });

  test("handles backend lifecycle, assistant, local model, and source events", () => {
    const handlers = makeHandlers({ session: { sources: [{ name: "mic", enabled: true }], transcript: [] } });

    handleBackendEvent({ type: "backend_ready" }, handlers);
    handleBackendEvent({ type: "assistant_ping_result", ok: true, providerId: "codex" }, handlers);
    handleBackendEvent({ type: "local_llm_status", profileId: "local", state: "ready", message: "loaded" }, handlers);
    handleBackendEvent({ type: "source_added", source: { name: "sys", enabled: true } }, handlers);
    handleBackendEvent({ type: "source_updated", source: { name: "mic", enabled: false } }, handlers);
    handleBackendEvent({ type: "source_removed", source: { name: "sys" } }, handlers);
    handleBackendEvent({ type: "session_started" }, handlers);
    handleBackendEvent({ type: "backend_exit", code: 1, signal: null }, handlers);

    expect(handlers.backendStatus).toMatchObject({ ready: false, running: false });
    expect(handlers.assistantPing).toMatchObject({ busy: false, ok: true, providerId: "codex" });
    expect(handlers.localLlmStatus.local).toEqual({ state: "ready", message: "loaded" });
    expect(handlers.state.session.sources).toEqual([{ name: "mic", enabled: false }]);
    expect(handlers.refresh).toHaveBeenCalledWith({ includeDevices: false, showLoading: false });
    expect(handlers.error).toBe("Python backend stopped (1, null)");
  });

  test("applies streaming words and final events", () => {
    const handlers = makeHandlers();

    handleBackendEvent({ type: "streaming_words", stream: "mic", confirmed: [{ text: "hello" }], tentative: [{ text: "there" }] }, handlers);
    expect(handlers.state.session.transcript[0]).toMatchObject({ tentative: true, confirmedText: "hello" });

    handleBackendEvent({ type: "streaming_final", stream: "mic", words: [{ text: "hello" }, { text: "there" }] }, handlers);
    expect(handlers.state.session.transcript).toEqual([expect.objectContaining({ text: "hello there", tentative: false })]);
  });
});
