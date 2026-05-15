/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { PipelinePanel } from "../../../../src/shared/ui/pipeline/PipelinePanel";
import { ResizablePipeline } from "../../../../src/shared/ui/pipeline/ResizablePipeline";

function installResizeObserver(width = 900) {
  globalThis.ResizeObserver = class {
    constructor(callback) {
      this.callback = callback;
    }
    observe(element) {
      Object.defineProperty(element, "clientWidth", { configurable: true, value: width });
      this.callback();
    }
    disconnect() {}
  };
}

function columns() {
  return [
    { id: "audio", label: "Audio", minWidth: 160, defaultWidth: 220, element: <Column title="Audio">A</Column> },
    { id: "processing", label: "Processing", minWidth: 180, defaultWidth: 260, flex: true, element: <Column title="Processing">B</Column> },
    { id: "assistant", label: "Assistant", minWidth: 160, defaultWidth: 220, element: <Column title="Assistant">C</Column> }
  ];
}

function Column({ children, headerProps, layoutControls, title }) {
  return (
    <PipelinePanel title={title} headerControls={layoutControls} headerProps={headerProps}>
      {children}
    </PipelinePanel>
  );
}

describe("ResizablePipeline", () => {
  beforeEach(() => {
    window.localStorage.clear();
    installResizeObserver();
  });

  test("renders resizable columns, persists keyboard resize, and hides columns", () => {
    const onHideColumn = vi.fn();
    render(<ResizablePipeline columns={columns()} onHideColumn={onHideColumn} storageKey="width-test" />);

    expect(screen.getByRole("heading", { name: "Audio" })).toBeInTheDocument();
    const resizer = screen.getByRole("button", { name: /Resize Audio and Processing/i });
    fireEvent.keyDown(resizer, { key: "ArrowRight" });

    expect(JSON.parse(window.localStorage.getItem("width-test")).audio).toBeGreaterThan(220);
    fireEvent.click(screen.getByRole("button", { name: /Hide Audio/i }));
    expect(onHideColumn).toHaveBeenCalledWith("audio");
  });

  test("supports drag-and-drop reordering and reset signals", () => {
    const onReorderColumn = vi.fn();
    const { rerender } = render(
      <ResizablePipeline columns={columns()} onReorderColumn={onReorderColumn} resetSignal={0} storageKey="width-test" />
    );
    const audioHead = screen.getByRole("heading", { name: "Audio" }).closest("header");
    const assistantHead = screen.getByRole("heading", { name: "Assistant" }).closest("header");

    assistantHead.getBoundingClientRect = () => ({ left: 100, width: 100 });
    fireEvent.dragStart(audioHead, { dataTransfer: dataTransfer() });
    fireEvent.dragOver(assistantHead, { clientX: 180, dataTransfer: dataTransfer() });
    fireEvent.drop(assistantHead, { clientX: 180, dataTransfer: dataTransfer("audio") });

    expect(onReorderColumn).toHaveBeenCalledWith("audio", "assistant", "before");

    fireEvent.keyDown(screen.getByRole("button", { name: /Resize Audio and Processing/i }), { key: "ArrowRight" });
    expect(window.localStorage.getItem("width-test")).toBeTruthy();
    rerender(<ResizablePipeline columns={columns()} onReorderColumn={onReorderColumn} resetSignal={1} storageKey="width-test" />);
    expect(window.localStorage.getItem("width-test")).toBeNull();
  });
});

function dataTransfer(sourceId = "") {
  const store = new Map();
  if (sourceId) store.set("text/plain", sourceId);
  return {
    effectAllowed: "",
    dropEffect: "",
    getData: vi.fn((key) => store.get(key) || ""),
    setData: vi.fn((key, value) => store.set(key, value))
  };
}
