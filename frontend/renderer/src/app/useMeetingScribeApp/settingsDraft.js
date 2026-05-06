import { ASR_FIELDS, asrProfileRequiresStreaming } from "../../entities/settings/model";

export function mergeSettingsPatch(current, patch, options) {
  const next = { ...current, ...patch };
  return asrProfileRequiresStreaming(next.profile, options)
    ? { ...next, streamingEnabled: true }
    : next;
}

export function applyLockedProfileDefaults(current, options) {
  return asrProfileRequiresStreaming(current.profile, options)
    ? applyProfileDefaults(current, current.profile, options)
    : current;
}

export function applyProfileDefaults(current, profile, options) {
  const defaults = options.profileDefaults?.[profile];
  const requiresStreaming = asrProfileRequiresStreaming(profile, options);
  if (!hasProfileDefaults(profile, defaults)) {
    return withProfileStreaming(current, profile, requiresStreaming);
  }
  return {
    ...current,
    profile,
    computeType: String(defaultValue(defaults.compute_type, current.computeType)),
    overloadStrategy: String(defaultValue(defaults.overload_strategy, current.overloadStrategy)),
    streamingEnabled: profileStreamingEnabled(defaults, current.streamingEnabled, requiresStreaming),
    streamingChunkIntervalS: defaultValue(defaults.streaming_chunk_interval_s, current.streamingChunkIntervalS),
    streamingEndpointSilenceMs: defaultValue(defaults.streaming_endpoint_silence_ms, current.streamingEndpointSilenceMs),
    asr: {
      ...current.asr,
      ...Object.fromEntries(ASR_FIELDS.map((field) => [field.key, defaultValue(defaults[field.key], current.asr[field.key])]))
    }
  };
}

function hasProfileDefaults(profile, defaults) {
  return Boolean(defaults) && profile !== "Custom";
}

function withProfileStreaming(current, profile, requiresStreaming) {
  return {
    ...current,
    profile,
    streamingEnabled: requiresStreaming ? true : current.streamingEnabled
  };
}

function profileStreamingEnabled(defaults, currentValue, requiresStreaming) {
  return requiresStreaming ? true : Boolean(defaultValue(defaults.streaming_enabled, currentValue));
}

function defaultValue(value, fallback) {
  return value ?? fallback;
}
