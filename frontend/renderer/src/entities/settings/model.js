export {
  ASR_FIELDS,
  ASR_PROFILE_ULTRA_FAST,
  ASR_RESOURCE_FIELDS,
  ASR_TIMING_FIELDS,
  ASSISTANT_PROVIDER_OPTIONS,
  ASSISTANT_REASONING_OPTIONS,
  DIARIZATION_BACKENDS,
  DIARIZATION_PROVIDERS,
  FALLBACK_OPTIONS
} from "./modelParts/constants";
export {
  asrProfileRequiresStreaming,
  buildProxyUrl,
  normalizeAssistantProfiles
} from "./modelParts/helpers";
export {
  applySettingsToConfig,
  draftToStartParams,
  makeSettingsDraft
} from "./modelParts/mappers";

export function uniqueOptions(options, currentValue) {
  const result = [];
  const seen = new Set();
  for (const value of [...(options || []), currentValue]) {
    const text = String(value || "").trim();
    if (!text || seen.has(text)) {
      continue;
    }
    seen.add(text);
    result.push(text);
  }
  return result;
}

export function languageLabel(language) {
  const labels = { auto: "Auto", en: "English", ru: "Russian" };
  return labels[language] || language;
}
