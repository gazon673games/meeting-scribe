import { createRequire } from "node:module";

import { describe, expect, test } from "vitest";

const require = createRequire(import.meta.url);
const { chooseRendererEndpoint, endpointFromExplicitUrl } = require("../dev-runner.cjs");
const { parsePort, rendererUrlFromEnv, rendererUrlFromHostPort } = require("../renderer-url.cjs");

describe("renderer dev URL configuration", () => {
  test("builds the Electron renderer URL from environment settings", () => {
    expect(parsePort("6000")).toBe(6000);
    expect(parsePort("not-a-port", 5173)).toBe(5173);
    expect(rendererUrlFromHostPort("::1", 5174)).toBe("http://[::1]:5174");
    expect(rendererUrlFromEnv({ VITE_DEV_SERVER_HOST: "localhost", VITE_DEV_SERVER_PORT: "6200" })).toBe(
      "http://localhost:6200"
    );
    expect(rendererUrlFromEnv({ ELECTRON_RENDERER_URL: "http://example.test:7777/app" })).toBe(
      "http://example.test:7777/app"
    );
  });

  test("chooses a fallback port only when the port was not explicit", async () => {
    const endpoint = await chooseRendererEndpoint({}, async (_host, port) => port === 5175);

    expect(endpoint).toEqual({ host: "127.0.0.1", port: 5175, url: "http://127.0.0.1:5175" });
    await expect(chooseRendererEndpoint({ VITE_DEV_SERVER_PORT: "6200" }, async () => false)).rejects.toThrow("6200");
  });

  test("parses explicit renderer URLs for a single fixed dev port", () => {
    expect(endpointFromExplicitUrl("http://localhost:6300/app")).toEqual({
      host: "localhost",
      port: 6300,
      url: "http://localhost:6300/app"
    });
  });
});
