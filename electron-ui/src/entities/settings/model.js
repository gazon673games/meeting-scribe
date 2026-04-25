export const FALLBACK_OPTIONS = {
  languages: ["ru", "en", "auto"],
  asrProfiles: ["Realtime", "Balanced", "Quality", "Custom"],
  asrModels: [
    "large-v3",
    "large-v3-turbo",
    "bzikst/faster-whisper-large-v3-russian",
    "bzikst/faster-whisper-podlodka-turbo",
    "medium",
    "small"
  ],
  asrModes: [
    { id: "mix", label: "MIX (master)" },
    { id: "split", label: "SPLIT (all sources)" }
  ],
  computeTypes: ["int8_float16", "float16", "int8", "int8_float32", "float32"],
  overloadStrategies: ["drop_old", "keep_all"],
  profileDefaults: {}
};

export const ASR_FIELDS = [
  { key: "beam_size", label: "Beam", defaultValue: 5, step: 1, min: 1, max: 20, integer: true },
  { key: "endpoint_silence_ms", label: "Endpoint ms", defaultValue: 650, step: 10, min: 50, max: 5000 },
  { key: "max_segment_s", label: "Segment s", defaultValue: 7, step: 0.5, min: 1, max: 60 },
  { key: "overlap_ms", label: "Overlap ms", defaultValue: 200, step: 10, min: 0, max: 2000 },
  { key: "vad_energy_threshold", label: "VAD", defaultValue: 0.0055, step: 0.0001, min: 0.00001, max: 1 },
  { key: "overload_enter_qsize", label: "Overload in", defaultValue: 18, step: 1, min: 1, max: 999, integer: true },
  { key: "overload_exit_qsize", label: "Overload out", defaultValue: 6, step: 1, min: 1, max: 999, integer: true },
  { key: "overload_hard_qsize", label: "Hard q", defaultValue: 28, step: 1, min: 1, max: 999, integer: true },
  { key: "overload_beam_cap", label: "Beam cap", defaultValue: 2, step: 1, min: 1, max: 20, integer: true },
  { key: "overload_max_segment_s", label: "Overload seg", defaultValue: 5, step: 0.5, min: 0.5, max: 60 },
  { key: "overload_overlap_ms", label: "Overload overlap", defaultValue: 120, step: 10, min: 0, max: 2000 }
];

export function makeSettingsDraft(config) {
  const ui = objectSection(config?.ui);
  const asr = objectSection(config?.asr);
  return {
    asrEnabled: boolWithDefault(ui.asr_enabled, true),
    wavEnabled: boolWithDefault(ui.wav_enabled, false),
    offlineOnStop: boolWithDefault(ui.offline_on_stop, false),
    realtimeTranscriptToFile: boolWithDefault(ui.rt_transcript_to_file, false),
    language: String(ui.lang || "ru"),
    asrMode: Number(ui.asr_mode || 0) === 1 ? "split" : "mix",
    model: String(ui.model || "medium"),
    profile: String(ui.profile || "Balanced"),
    outputFile: String(ui.output_file || "capture_mix.wav"),
    computeType: String(asr.compute_type || "float16"),
    overloadStrategy: String(asr.overload_strategy || "drop_old"),
    asr: Object.fromEntries(ASR_FIELDS.map((field) => [field.key, asr[field.key] ?? field.defaultValue]))
  };
}

export function applySettingsToConfig(config, draft) {
  const current = objectSection(config);
  return {
    ...current,
    version: current.version ?? 2,
    ui: {
      ...objectSection(current.ui),
      asr_enabled: Boolean(draft.asrEnabled),
      lang: String(draft.language || "ru"),
      asr_mode: draft.asrMode === "split" ? 1 : 0,
      model: String(draft.model || "medium"),
      profile: String(draft.profile || "Balanced"),
      wav_enabled: Boolean(draft.wavEnabled),
      output_file: String(draft.outputFile || "capture_mix.wav"),
      long_run: boolWithDefault(current.ui?.long_run, true),
      rt_transcript_to_file: Boolean(draft.realtimeTranscriptToFile),
      offline_on_stop: Boolean(draft.offlineOnStop),
      asr_settings_expanded: boolWithDefault(current.ui?.asr_settings_expanded, false)
    },
    asr: {
      ...objectSection(current.asr),
      compute_type: String(draft.computeType || "float16"),
      overload_strategy: String(draft.overloadStrategy || "drop_old"),
      ...Object.fromEntries(ASR_FIELDS.map((field) => [field.key, normalizeNumber(draft.asr[field.key], field)]))
    }
  };
}

export function draftToStartParams(draft) {
  const config = applySettingsToConfig({}, draft);
  return {
    asrEnabled: config.ui.asr_enabled,
    wavEnabled: config.ui.wav_enabled,
    runOfflinePass: config.ui.offline_on_stop,
    realtimeTranscriptToFile: config.ui.rt_transcript_to_file,
    language: config.ui.lang,
    asrMode: draft.asrMode,
    profile: config.ui.profile,
    model: config.ui.model,
    outputFile: config.ui.output_file,
    ...config.asr
  };
}

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

function objectSection(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function boolWithDefault(value, fallback) {
  return value === undefined || value === null ? Boolean(fallback) : Boolean(value);
}

function normalizeNumber(value, field) {
  const parsed = Number(String(value ?? field.defaultValue).replace(",", "."));
  let next = Number.isFinite(parsed) ? parsed : field.defaultValue;
  next = Math.max(Number(field.min), Math.min(Number(field.max), next));
  return field.integer ? Math.round(next) : next;
}
