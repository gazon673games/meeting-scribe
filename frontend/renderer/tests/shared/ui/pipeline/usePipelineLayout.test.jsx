/* @vitest-environment jsdom */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, test } from "vitest";

import { usePipelineLayout } from "../../../../src/shared/ui/pipeline/usePipelineLayout";

const columns = [
  { id: "audio", label: "Audio" },
  { id: "processing", label: "Processing" },
  { id: "assistant", label: "Assistant" }
];

describe("usePipelineLayout", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  test("normalizes stored layout and persists ordering changes", () => {
    window.localStorage.setItem("layout-test", JSON.stringify({ order: ["missing", "assistant"], hidden: ["ghost"] }));

    const { result } = renderHook(() => usePipelineLayout(columns, "layout-test"));
    expect(result.current.columns.map((column) => column.id)).toEqual(["assistant", "audio", "processing"]);
    expect(result.current.hiddenIds).toEqual([]);

    act(() => result.current.moveColumn("assistant", 1));
    expect(result.current.columns.map((column) => column.id)).toEqual(["audio", "assistant", "processing"]);

    act(() => result.current.moveColumnTo("processing", "audio", "before"));
    expect(result.current.columns.map((column) => column.id)).toEqual(["processing", "audio", "assistant"]);
    expect(JSON.parse(window.localStorage.getItem("layout-test")).order).toEqual(["processing", "audio", "assistant"]);
  });

  test("hides, shows, and resets columns while keeping one visible column", () => {
    const { result } = renderHook(() => usePipelineLayout(columns, "layout-test"));

    act(() => result.current.hideColumn("audio"));
    act(() => result.current.hideColumn("processing"));
    act(() => result.current.hideColumn("assistant"));
    expect(result.current.hiddenIds).toEqual(["audio", "processing"]);
    expect(result.current.visibleIds).toEqual(["assistant"]);

    act(() => result.current.showColumn("audio"));
    expect(result.current.visibleIds).toEqual(["audio", "assistant"]);

    act(() => result.current.resetLayout());
    expect(result.current.visibleIds).toEqual(["audio", "processing", "assistant"]);
    expect(result.current.resetRevision).toBe(1);
  });
});
