import { describe, expect, test } from "vitest";

import { applyAsrMetrics, removeSessionSource, upsertSessionSource } from "../../../src/app/useMeetingScribeApp/sessionState";

describe("session state reducers", () => {
  test("updates ASR metrics while preserving session state", () => {
    const current = { session: { running: true, asrMetrics: { avgLatencyS: 1, lagS: 2 } } };

    expect(applyAsrMetrics(current, { seg_dropped_total: 3, p95_latency_s: 1.5 })).toMatchObject({
      session: {
        running: true,
        asrMetrics: {
          segDroppedTotal: 3,
          segSkippedTotal: 0,
          avgLatencyS: 1,
          p95LatencyS: 1.5,
          lagS: 2
        }
      }
    });
    expect(applyAsrMetrics(null, {})).toBeNull();
  });

  test("adds, updates, and removes session sources by name", () => {
    let current = { session: { sources: [{ name: "mic", enabled: true }] } };

    current = upsertSessionSource(current, { name: "sys", enabled: true });
    expect(current.session.sources.map((source) => source.name)).toEqual(["mic", "sys"]);

    current = upsertSessionSource(current, { name: "mic", enabled: false, label: "Microphone" });
    expect(current.session.sources[0]).toMatchObject({ name: "mic", enabled: false, label: "Microphone" });

    current = removeSessionSource(current, "sys");
    expect(current.session.sources.map((source) => source.name)).toEqual(["mic"]);
    expect(removeSessionSource(current, "")).toBe(current);
    expect(upsertSessionSource(current, {})).toBe(current);
  });
});
