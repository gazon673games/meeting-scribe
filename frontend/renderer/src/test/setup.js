import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import { vi } from "vitest";

afterEach(() => cleanup());

if (typeof HTMLElement !== "undefined" && !HTMLElement.prototype.scrollIntoView) {
  HTMLElement.prototype.scrollIntoView = vi.fn();
}

if (typeof window !== "undefined") {
  const store = new Map();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      clear: () => store.clear(),
      getItem: (key) => store.get(String(key)) ?? null,
      removeItem: (key) => store.delete(String(key)),
      setItem: (key, value) => store.set(String(key), String(value))
    }
  });
}
