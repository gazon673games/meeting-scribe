/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";

import { AsrModelRow, asrStatusClass, asrStatusLabel } from "../../../src/features/settings/AsrModelRow";
import { DiarModelRow } from "../../../src/features/settings/DiarModelRow";
import { LlmModelRow } from "../../../src/features/settings/LlmModelRow";

describe("model row components", () => {
  test("formats ASR row states and actions", async () => {
    const user = userEvent.setup();
    expect(asrStatusLabel({ cached: true, compatible: false })).toBe("incompatible");
    expect(asrStatusLabel({ status: "missing_files", missing: ["config.json"] })).toBe("missing config.json");
    expect(asrStatusClass({ downloadError: "bad" }, false, false)).toBe("model-status-error");

    const onChange = vi.fn();
    const onDelete = vi.fn();
    const onToggleMeta = vi.fn();
    render(
      <AsrModelRow
        model={{ name: "large-v3", label: "Large", cached: true, compatible: true, deletable: true, path: "models/large" }}
        isDownloading={false}
        isSelected={false}
        isExternal
        metaEntry={{
          loading: false,
          error: "",
          metadata: { name: "large-v3", compatible: true, totalBytes: 2048, weightFiles: [{ name: "model.bin", bytes: 2048 }] }
        }}
        onChange={onChange}
        onDelete={onDelete}
        onDownload={vi.fn()}
        onRemoveEntry={vi.fn()}
        onToggleMeta={onToggleMeta}
      />
    );

    expect(screen.getByText("model.bin (2 KB)")).toBeInTheDocument();
    await user.click(screen.getByTitle("Use large-v3"));
    expect(onChange).toHaveBeenCalledWith({ model: "large-v3" });
    await user.click(screen.getByTitle("Delete large-v3"));
    expect(onDelete).toHaveBeenCalledWith("large-v3");
    await user.click(screen.getByTitle("Hide metadata"));
    expect(onToggleMeta).toHaveBeenCalledWith(expect.objectContaining({ name: "large-v3" }));
  });

  test("renders diarization row download, use, delete, and details states", async () => {
    const user = userEvent.setup();
    const props = {
      model: { name: "speaker", label: "Speaker", cached: false, compatible: false, bytes: 4096, path: "speaker.onnx" },
      selected: false,
      isDownloading: false,
      expanded: true,
      deletable: true,
      showStatus: true,
      onUse: vi.fn(),
      onDownload: vi.fn(),
      onDelete: vi.fn(),
      onToggleExpand: vi.fn()
    };
    const { rerender } = render(<DiarModelRow {...props} />);

    expect(screen.getAllByText("downloadable")).toHaveLength(2);
    expect(screen.getByText("speaker.onnx")).toBeInTheDocument();
    await user.click(screen.getByTitle("Download Speaker"));
    expect(props.onDownload).toHaveBeenCalledWith(props.model);
    await user.click(screen.getByTitle("Delete Speaker"));
    expect(props.onDelete).toHaveBeenCalledWith(props.model);

    rerender(<DiarModelRow {...props} model={{ ...props.model, cached: true }} selected />);
    await user.click(screen.getByTitle("Already selected"));
    expect(props.onUse).not.toHaveBeenCalled();
  });

  test("renders LLM row linked and downloadable states", async () => {
    const user = userEvent.setup();
    const onUse = vi.fn();
    const onDelete = vi.fn();
    const { rerender } = render(
      <LlmModelRow
        model={{ name: "qwen.gguf", label: "Qwen", cached: true, path: "models/qwen.gguf", bytes: 2048 }}
        alias="qwen"
        linked={false}
        isDownloading={false}
        onUse={onUse}
        onDelete={onDelete}
      />
    );

    await user.click(screen.getByTitle("Use qwen in selected assistant profile"));
    expect(onUse).toHaveBeenCalledWith(expect.objectContaining({ name: "qwen.gguf" }));
    await user.click(screen.getByTitle("Delete Qwen"));
    expect(onDelete).toHaveBeenCalledWith(expect.objectContaining({ path: "models/qwen.gguf" }));

    rerender(
      <LlmModelRow
        model={{ name: "bad.gguf", downloadError: "network", downloadMessage: "retrying" }}
        alias=""
        linked
        isDownloading
        onUse={onUse}
        onDelete={onDelete}
      />
    );
    const row = screen.getByText("bad.gguf").closest(".model-row");
    expect(within(row).getByText("0 B")).toBeInTheDocument();
    expect(within(row).getAllByRole("button")[0]).toBeDisabled();
  });
});
