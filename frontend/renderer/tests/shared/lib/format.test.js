import { describe, expect, test } from "vitest";

import { formatBytes, formatDelay, formatNumber, formatTime } from "../../../src/shared/lib/format";

describe("format helpers", () => {
  test("formats delay and numeric values for compact UI labels", () => {
    expect(formatDelay(2)).toBe("2");
    expect(formatDelay(2.125)).toBe("2.13");
    expect(formatDelay(Number.NaN)).toBe("0");
    expect(formatNumber("3.456")).toBe("3.46");
    expect(formatNumber("bad")).toBe("0.00");
  });

  test("formats byte and timestamp values with fallbacks", () => {
    expect(formatBytes(0)).toBe("-");
    expect(formatBytes(1024)).toBe("1 KB");
    expect(formatBytes(1024 * 1024 * 1024 * 3.5)).toBe("3.5 GB");
    expect(formatTime("bad")).toBe("--:--");
    expect(formatTime(0)).toMatch(/\d{2}:\d{2}:\d{2}/);
  });
});
