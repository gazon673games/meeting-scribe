import { describe, expect, test } from "vitest";

import { applyLockedProfileDefaults, applyProfileDefaults, mergeSettingsPatch } from "../../../src/app/useMeetingScribeApp/settingsDraft";

const options = {
  streamingRequiredProfiles: ["Ultra Fast"],
  profileDefaults: {
    Quality: {
      compute_type: "float16",
      overload_strategy: "block",
      streaming_enabled: false,
      streaming_chunk_interval_s: 3,
      streaming_endpoint_silence_ms: 900,
      beam_size: 8
    },
    "Ultra Fast": {
      compute_type: "int8",
      overload_strategy: "drop_old",
      streaming_enabled: true,
      streaming_chunk_interval_s: 0.5
    }
  }
};

function draft() {
  return {
    profile: "Balanced",
    computeType: "int8",
    overloadStrategy: "drop_old",
    streamingEnabled: false,
    streamingChunkIntervalS: 1,
    streamingEndpointSilenceMs: 500,
    asr: { beam_size: 4, temperature: 0 }
  };
}

describe("settings draft helpers", () => {
  test("forces streaming when selected profile requires it", () => {
    expect(mergeSettingsPatch(draft(), { profile: "Ultra Fast", streamingEnabled: false }, options)).toMatchObject({
      profile: "Ultra Fast",
      streamingEnabled: true
    });
  });

  test("applies profile defaults and preserves fallback values", () => {
    expect(applyProfileDefaults(draft(), "Quality", options)).toMatchObject({
      profile: "Quality",
      computeType: "float16",
      overloadStrategy: "block",
      streamingEnabled: false,
      streamingChunkIntervalS: 3,
      streamingEndpointSilenceMs: 900,
      asr: { beam_size: 8, temperature: 0 }
    });
    expect(applyProfileDefaults(draft(), "Custom", options)).toMatchObject({
      profile: "Custom",
      computeType: "int8"
    });
  });

  test("locks profile defaults only for streaming-required profiles", () => {
    expect(applyLockedProfileDefaults({ ...draft(), profile: "Balanced" }, options).profile).toBe("Balanced");
    expect(applyLockedProfileDefaults({ ...draft(), profile: "Ultra Fast" }, options)).toMatchObject({
      profile: "Ultra Fast",
      streamingEnabled: true,
      computeType: "int8"
    });
  });
});
