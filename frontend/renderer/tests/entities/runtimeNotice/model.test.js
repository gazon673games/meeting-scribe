import { describe, expect, test, vi } from "vitest";

import { noticeFromBackendEvent, upsertRuntimeNotice } from "../../../src/entities/runtimeNotice/model";

describe("runtime notice model", () => {
  test("builds user-facing notices from actionable backend events", () => {
    expect(noticeFromBackendEvent(null)).toBeNull();
    expect(noticeFromBackendEvent({ type: "backend_stderr", text: "noise only" })).toBeNull();
    expect(noticeFromBackendEvent({ type: "source_error", source: "mic", error: "device missing" })).toMatchObject({
      title: "Audio Source",
      message: "mic: device missing",
      severity: "error"
    });
    expect(noticeFromBackendEvent({ type: "error", component: "diarization", error: "diarization failed" })).toMatchObject({
      title: "Speaker ID",
      severity: "error"
    });
    expect(noticeFromBackendEvent({ type: "backend_exit", code: 1, signal: null })).toMatchObject({
      title: "Backend",
      message: "Python backend stopped (1, null)"
    });
  });

  test("upserts notices by key and keeps the newest notices first", () => {
    vi.spyOn(Date, "now").mockReturnValue(123);
    const first = { key: "same", title: "Runtime", message: "A", count: 1 };
    const updated = upsertRuntimeNotice([first, { key: "old" }], { key: "same", title: "Runtime", message: "B" });

    expect(updated[0]).toMatchObject({ key: "same", message: "B", count: 2, ts: 123 });
    expect(upsertRuntimeNotice(updated, null)).toBe(updated);
    expect(upsertRuntimeNotice(updated, { key: "new", message: "C" }, 2)).toHaveLength(2);
  });
});
