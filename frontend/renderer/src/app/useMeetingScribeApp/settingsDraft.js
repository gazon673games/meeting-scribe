import {
  ASR_FIELDS,
  asrProfileRequiresStreaming,
} from '../../entities/settings/model';

export const mergeSettingsPatch = (current, patch, options) => {
  const next = { ...current, ...patch };
  if (patch.wavEnabled === false) {
    next.offlineOnStop = false;
  }
  return asrProfileRequiresStreaming(next.profile, options)
    ? { ...next, streamingEnabled: true }
    : next;
};

const hasProfileDefaults = (profile, defaults) =>
  Boolean(defaults) && profile !== 'Custom';

const withProfileStreaming = (current, profile, requiresStreaming) => ({
  ...current,
  profile,
  streamingEnabled: requiresStreaming ? true : current.streamingEnabled,
});

const defaultValue = (value, fallback) => value ?? fallback;

const profileStreamingEnabled = (defaults, requiresStreaming) =>
  requiresStreaming
    ? true
    : Boolean(defaultValue(defaults.streaming_enabled, false));

export const applyProfileDefaults = (current, profile, options) => {
  const defaults = options.profileDefaults?.[profile];
  const requiresStreaming = asrProfileRequiresStreaming(profile, options);
  if (!hasProfileDefaults(profile, defaults)) {
    return withProfileStreaming(current, profile, requiresStreaming);
  }
  return {
    ...current,
    profile,
    model: String(options.profileRecommendedModels?.[profile] || current.model),
    computeType: String(
      defaultValue(defaults.compute_type, current.computeType),
    ),
    overloadStrategy: String(
      defaultValue(defaults.overload_strategy, current.overloadStrategy),
    ),
    streamingEnabled: profileStreamingEnabled(defaults, requiresStreaming),
    streamingChunkIntervalS: defaultValue(
      defaults.streaming_chunk_interval_s,
      current.streamingChunkIntervalS,
    ),
    streamingEndpointSilenceMs: defaultValue(
      defaults.streaming_endpoint_silence_ms,
      current.streamingEndpointSilenceMs,
    ),
    asr: {
      ...current.asr,
      ...Object.fromEntries(
        ASR_FIELDS.map((field) => [
          field.key,
          defaultValue(defaults[field.key], current.asr[field.key]),
        ]),
      ),
    },
  };
};

export const applyLockedProfileDefaults = (current, options) =>
  asrProfileRequiresStreaming(current.profile, options)
    ? applyProfileDefaults(current, current.profile, options)
    : current;
