/* @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";

import { RuntimeNoticeTray } from "../../../src/features/runtime/RuntimeNoticeTray";

describe("RuntimeNoticeTray", () => {
  test("renders grouped runtime notices and dismisses by key", async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();
    render(
      <RuntimeNoticeTray
        notices={[
          { key: "diar", severity: "error", title: "Speaker ID", message: "Model failed", detail: "missing file", count: 2 },
          { key: "asr", severity: "warning", title: "ASR", message: "Running on CPU", count: 1 }
        ]}
        onDismiss={onDismiss}
      />
    );

    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText("x2")).toBeInTheDocument();
    expect(screen.getByText("missing file")).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: /Dismiss runtime notice/i })[1]);
    expect(onDismiss).toHaveBeenCalledWith("asr");
  });

  test("renders nothing without notices", () => {
    const { container } = render(<RuntimeNoticeTray notices={[]} onDismiss={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });
});
