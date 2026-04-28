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
  asrDevices: ["cuda", "cpu"],
  computeTypes: ["int8_float16", "float16", "int8", "int8_float32", "float32"],
  overloadStrategies: ["drop_old", "keep_all"],
  profileDefaults: {}
};

export const ASR_RESOURCE_FIELDS = [
  { key: "cpu_threads", label: "CPU Threads", defaultValue: 0, step: 1, min: 0, max: 64, integer: true },
  { key: "num_workers", label: "Workers", defaultValue: 1, step: 1, min: 1, max: 16, integer: true }
];

export const ASR_TIMING_FIELDS = [
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

export const ASR_FIELDS = [...ASR_RESOURCE_FIELDS, ...ASR_TIMING_FIELDS];

export function makeSettingsDraft(config) {
  const ui = objectSection(config?.ui);
  const asr = objectSection(config?.asr);
  return {
    wavEnabled: boolWithDefault(ui.wav_enabled, false),
    offlineOnStop: false,
    realtimeTranscriptToFile: boolWithDefault(ui.rt_transcript_to_file, false),
    screenCaptureProtection: boolWithDefault(ui.screen_capture_protection, false),
    theme: normalizeTheme(ui.theme),
    language: String(ui.lang || "ru"),
    asrMode: Number(ui.asr_mode || 0) === 1 ? "split" : "mix",
    model: String(ui.model || "medium"),
    profile: String(ui.profile || "Balanced"),
    outputFile: String(ui.output_file || "capture_mix.wav"),
    device: String(asr.device || "cuda"),
    computeType: String(asr.compute_type || "float16"),
    overloadStrategy: String(asr.overload_strategy || "drop_old"),
    perProcessAudio: boolWithDefault(ui.per_process_audio, false),
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
      asr_enabled: true,
      screen_capture_protection: Boolean(draft.screenCaptureProtection),
      theme: normalizeTheme(draft.theme),
      lang: String(draft.language || "ru"),
      asr_mode: draft.asrMode === "split" ? 1 : 0,
      model: String(draft.model || "medium"),
      profile: String(draft.profile || "Balanced"),
      wav_enabled: Boolean(draft.wavEnabled),
      output_file: String(draft.outputFile || "capture_mix.wav"),
      long_run: boolWithDefault(current.ui?.long_run, true),
      rt_transcript_to_file: Boolean(draft.realtimeTranscriptToFile),
      offline_on_stop: false,
      asr_settings_expanded: boolWithDefault(current.ui?.asr_settings_expanded, false),
      per_process_audio: Boolean(draft.perProcessAudio)
    },
    asr: {
      ...objectSection(current.asr),
      device: String(draft.device || "cuda"),
      compute_type: String(draft.computeType || "float16"),
      overload_strategy: String(draft.overloadStrategy || "drop_old"),
      ...Object.fromEntries(ASR_FIELDS.map((field) => [field.key, normalizeNumber(draft.asr[field.key], field)]))
    }
  };
}

export function draftToStartParams(draft) {
  const config = applySettingsToConfig({}, draft);
  return {
    asrEnabled: true,
    wavEnabled: config.ui.wav_enabled,
    runOfflinePass: false,
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

function normalizeTheme(value) {
  return String(value || "").trim().toLowerCase() === "light" ? "light" : "dark";
}

function normalizeNumber(value, field) {
  const parsed = Number(String(value ?? field.defaultValue).replace(",", "."));
  let next = Number.isFinite(parsed) ? parsed : field.defaultValue;
  next = Math.max(Number(field.min), Math.min(Number(field.max), next));
  return field.integer ? Math.round(next) : next;
}
