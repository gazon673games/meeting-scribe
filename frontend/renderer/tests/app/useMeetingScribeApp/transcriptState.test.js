import { describe, expect, test, vi } from "vitest";

import {
  appendTranscriptLine,
  applyStreamingFinal,
  applyStreamingWords,
  mergeBackendStateSnapshot,
  updateTranscriptLine
} from "../../../src/app/useMeetingScribeApp/transcriptState";

function state(transcript = [], running = true) {
  return { session: { running, transcript } };
}

describe("transcript state reducers", () => {
  test("appends, updates, and deduplicates transcript lines", () => {
    vi.spyOn(Date, "now").mockReturnValue(10_000);
    const first = appendTranscriptLine(state(), {
      stream: "mic",
      t_start: 1,
      t_end: 2,
      text: "hello",
      speaker: "A"
    });

    expect(first.session.transcript[0]).toMatchObject({
      id: "mic:1000:2000",
      stream: "mic",
      speaker: "A",
      text: "hello"
    });
    expect(appendTranscriptLine(first, {
      id: first.session.transcript[0].id,
      stream: "mic",
      t_start: 1,
      t_end: 2,
      text: "hello",
      speaker: "A"
    })).toBe(first);

    const changed = appendTranscriptLine(first, { id: first.session.transcript[0].id, text: "updated", speaker: "B" });
    expect(changed.session.transcript).toHaveLength(1);
    expect(changed.session.transcript[0]).toMatchObject({ text: "updated", speaker: "B" });
    expect(appendTranscriptLine(changed, { text: "   " })).toBe(changed);
  });

  test("applies speaker updates by id or timing", () => {
    const current = state([{ id: "line-1", stream: "mic", t_start: 1, t_end: 2, speaker: "" }]);

    expect(updateTranscriptLine(current, { id: "line-1", speaker: "S1", speakerConfidence: 0.8 })).toMatchObject({
      session: { transcript: [{ speaker: "S1", speakerConfidence: 0.8 }] }
    });
    expect(updateTranscriptLine(current, { stream: "mic", t_start: 1, t_end: 2, speaker: "S2" })).toMatchObject({
      session: { transcript: [{ speaker: "S2" }] }
    });
    expect(updateTranscriptLine(current, { id: "missing", speaker: "S3" })).toBe(current);
  });

  test("keeps streaming words tentative until a final segment arrives", () => {
    let current = applyStreamingWords(state(), {
      stream: "mic",
      ts: 1,
      t_start: 0,
      t_end: 1,
      confirmed: [{ text: "hello" }],
      tentative: [{ text: "world" }]
    });
    current = applyStreamingWords(current, {
      stream: "mic",
      ts: 2,
      t_end: 2,
      confirmed: [{ text: "again" }],
      tentative: [{ text: "soon" }]
    });

    expect(current.session.transcript[0]).toMatchObject({
      id: "streaming-mic",
      confirmedText: "hello again",
      tentativeText: "soon",
      tentative: true
    });

    const final = applyStreamingFinal(current, {
      stream: "mic",
      ts: 3,
      t_start: 0,
      t_end: 2,
      words: [{ text: "hello" }, { text: "again" }]
    });
    expect(final.session.transcript).toEqual([
      expect.objectContaining({ id: "mic:3000:2000", text: "hello again", tentative: false })
    ]);
    expect(applyStreamingFinal(final, { stream: "mic", words: [] })).toMatchObject({ session: { transcript: final.session.transcript } });
  });

  test("merges running backend snapshots without losing local transcript lines", () => {
    const current = state([
      { id: "a", ts: 2, text: "local" },
      { id: "b", ts: 1, text: "old" },
      { id: "empty", ts: 3, text: "" }
    ]);
    const next = state([
      { id: "b", ts: 1, text: "updated" },
      { id: "c", ts: 4, text: "remote" }
    ]);

    expect(mergeBackendStateSnapshot(current, next).session.transcript.map((line) => line.text)).toEqual([
      "updated",
      "local",
      "remote"
    ]);
    expect(mergeBackendStateSnapshot(current, null)).toBe(current);
    expect(mergeBackendStateSnapshot(null, next)).toBe(next);
    expect(mergeBackendStateSnapshot(state([{ id: "x", text: "x" }], false), next)).toMatchObject(next);
  });
});
