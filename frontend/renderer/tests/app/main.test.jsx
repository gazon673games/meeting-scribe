/* @vitest-environment jsdom */
import React from "react";

import { beforeEach, describe, expect, test, vi } from "vitest";

const render = vi.hoisted(() => vi.fn());
const createRoot = vi.hoisted(() => vi.fn(() => ({ render })));

vi.mock("react-dom/client", () => ({ createRoot }));
vi.mock("../../src/app/App", () => ({
  App: function App() {
    return null;
  }
}));
vi.mock("../../src/app/DebugConsoleApp", () => ({
  DebugConsoleApp: function DebugConsoleApp() {
    return null;
  }
}));

async function importEntry(search = "") {
  window.history.pushState(null, "", `/${search}`);
  await import("../../src/main.jsx");
  return render.mock.calls[0][0];
}

describe("renderer entrypoint", () => {
  beforeEach(() => {
    vi.resetModules();
    createRoot.mockClear();
    render.mockClear();
    document.body.innerHTML = '<div id="root"></div>';
    window.history.pushState(null, "", "/");
  });

  test("mounts the main application into the root element", async () => {
    const element = await importEntry();

    expect(createRoot).toHaveBeenCalledWith(document.getElementById("root"));
    expect(element.type.name).toBe("App");
  });

  test("mounts the debug console when the URL requests diagnostics", async () => {
    const element = await importEntry("?debugConsole=1");

    expect(createRoot).toHaveBeenCalledWith(document.getElementById("root"));
    expect(element.type.name).toBe("DebugConsoleApp");
  });
});
