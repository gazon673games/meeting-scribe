import { describe, expect, test } from "vitest";

import { formatDownloadProgress } from "../../../src/features/settings/modelDownloadProgress";

describe("model download progress formatting", () => {
  test("uses fallback text when no progress has been reported yet", () => {
    expect(formatDownloadProgress({}, { fallback: "queued" })).toBe("queued");
    expect(formatDownloadProgress({}, { fallback: "" })).toBe("0 B");
  });

  test("formats downloaded, total, and speed labels", () => {
    expect(formatDownloadProgress({ downloadedBytes: 2048, totalBytes: 4096, speedBps: 1024 })).toBe(
      "2 KB / 4 KB - 1 KB/s"
    );
    expect(formatDownloadProgress({ downloadedBytes: 2048, totalBytes: 4096 }, { includeTotal: false })).toBe("2 KB");
    expect(formatDownloadProgress({ downloadedBytes: -1, speedBps: Number.NaN })).toBe("0 B");
  });
});
