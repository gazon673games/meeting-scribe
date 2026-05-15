/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";

import { ProcessingFeatureGrid } from "../../../src/features/processing/ProcessingFeatureGrid";
import { ProcessingStats } from "../../../src/features/processing/ProcessingStats";

describe("ProcessingFeatureGrid", () => {
  test("toggles processing features and locks streaming when required by profile", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <ProcessingFeatureGrid
        draft={{ asrMode: "mix", diarizationEnabled: false, profile: "Ultra Fast", streamingEnabled: false, wavEnabled: true }}
        locked={false}
        options={{ streamingLockedProfiles: ["Ultra Fast"] }}
        onChange={onChange}
      />
    );

    await user.click(screen.getByRole("button", { name: /Speaker Separation/i }));
    await user.click(screen.getByRole("button", { name: /Speaker ID/i }));
    await user.click(screen.getByRole("button", { name: /Record to File/i }));

    expect(onChange).toHaveBeenCalledWith({ asrMode: "split" });
    expect(onChange).toHaveBeenCalledWith({ diarizationEnabled: true });
    expect(onChange).toHaveBeenCalledWith({ wavEnabled: false });
    expect(screen.getByRole("button", { name: /Word-by-Word/i })).toBeDisabled();
  });

  test("disables speaker ID when selected backend misses its model path", () => {
    render(
      <ProcessingFeatureGrid
        draft={{
          asrMode: "split",
          diarizationEnabled: true,
          diarizationBackend: "sherpa_onnx",
          diarSherpaEmbeddingModelPath: "",
          profile: "Quality",
          streamingEnabled: false,
          wavEnabled: false
        }}
        locked={false}
        options={{ streamingLockedProfiles: [] }}
        onChange={vi.fn()}
      />
    );

    expect(screen.getByText(/Speaker ID model path not set/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Speaker ID/i })).toBeDisabled();
  });
});

describe("ProcessingStats", () => {
  test("shows live metrics when no download is active", () => {
    render(
      <ProcessingStats
        asrMetrics={{ avgLatencyS: 1.234, segDroppedTotal: 2, segSkippedTotal: 3 }}
        draft={{ model: "medium" }}
        session={{ state: "idle", drops: { droppedOutBlocks: 4, droppedTapBlocks: 5 } }}
        summary={{ model: "large-v3" }}
      />
    );

    expect(screen.getByText("large-v3")).toBeInTheDocument();
    expect(screen.getByText("1.23s")).toBeInTheDocument();
    expect(screen.getByText("2/3")).toBeInTheDocument();
    expect(screen.getByText("4/5")).toBeInTheDocument();
  });

  test("shows model download progress instead of runtime metrics", () => {
    render(
      <ProcessingStats
        asrMetrics={{}}
        draft={{ model: "medium" }}
        session={{ state: "downloading_model", modelDownload: { downloadedBytes: 2 * 1024 * 1024 * 1024, speedBps: 3 * 1024 * 1024 } }}
        summary={{}}
      />
    );

    expect(screen.getByText("Downloading model")).toBeInTheDocument();
    expect(screen.getByText(/2\.0 GB.*3\.0 MB\/s/)).toBeInTheDocument();
  });
});
