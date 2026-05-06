import {
  ASR_FIELDS,
  STREAMING_CHUNK_INTERVAL_FIELD,
  STREAMING_ENDPOINT_SILENCE_FIELD
} from "./constants";
import {
  asrProfileRequiresStreaming,
  boolWithDefault,
  buildProxyUrl,
  normalizeAsrProfile,
  normalizeAssistantProfiles,
  normalizeDiarizationBackend,
  normalizeDiarizationProvider,
  normalizeInteger,
  normalizeNumber,
  normalizeTheme,
  objectSection,
  parseProxy,
  selectedAssistantProfileId
} from "./helpers";

export function makeSettingsDraft(config) {
  const ui = objectSection(config?.ui);
  const asr = objectSection(config?.asr);
  const codex = objectSection(config?.codex);
  const models = objectSection(config?.models);
  const assistantProxy = parseProxy(codex.proxy);
  const modelProxy = parseProxy(models.proxy || codex.proxy);
  const proxy = assistantProxy.enabled ? assistantProxy : modelProxy;
  return {
    wavEnabled: boolWithDefault(ui.wav_enabled, false),
    offlineOnStop: false,
    realtimeTranscriptToFile: boolWithDefault(ui.rt_transcript_to_file, false),
    screenCaptureProtection: boolWithDefault(ui.screen_capture_protection, false),
    theme: normalizeTheme(ui.theme),
    language: String(ui.lang || "ru"),
    asrMode: Number(ui.asr_mode || 0) === 1 ? "split" : "mix",
    model: String(ui.model || "medium"),
    profile: normalizeAsrProfile(ui.profile),
    outputFile: String(ui.output_file || "capture_mix.wav"),
    assistantEnabled: boolWithDefault(codex.enabled, false),
    assistantSelectedProfileId: String(codex.selected_profile || ""),
    assistantProfiles: normalizeAssistantProfiles(codex.profiles),
    modelsDirectory: String(models.cache_dir || ""),
    modelsUseProxy: boolWithDefault(models.use_proxy, false),
    device: String(asr.device || "cuda"),
    computeType: String(asr.compute_type || "float16"),
    overloadStrategy: String(asr.overload_strategy || "drop_old"),
    diarizationEnabled: boolWithDefault(asr.diarization_enabled, false),
    diarizationBackend: normalizeDiarizationBackend(asr.diar_backend),
    diarizationSidecarEnabled: boolWithDefault(asr.diarization_sidecar_enabled, true),
    diarizationQueueSize: normalizeInteger(asr.diarization_queue_size, 50, 1, 500),
    diarSherpaEmbeddingModelPath: String(asr.diar_sherpa_embedding_model_path || ""),
    diarSherpaProvider: normalizeDiarizationProvider(asr.diar_sherpa_provider),
    diarSherpaNumThreads: normalizeInteger(asr.diar_sherpa_num_threads, 1, 1, 32),
    streamingEnabled: boolWithDefault(asr.streaming_enabled, false),
    streamingChunkIntervalS: normalizeNumber(asr.streaming_chunk_interval_s, STREAMING_CHUNK_INTERVAL_FIELD),
    streamingEndpointSilenceMs: normalizeNumber(asr.streaming_endpoint_silence_ms, STREAMING_ENDPOINT_SILENCE_FIELD),
    perProcessAudio: boolWithDefault(ui.per_process_audio, false),
    assistantProxyEnabled: assistantProxy.enabled,
    assistantProxyScheme: proxy.scheme,
    assistantProxyHost: proxy.host,
    assistantProxyPort: proxy.port,
    assistantProxyUsername: proxy.username,
    assistantProxyPassword: proxy.password,
    asr: Object.fromEntries(ASR_FIELDS.map((field) => [field.key, asr[field.key] ?? field.defaultValue]))
  };
}

export function applySettingsToConfig(config, draft) {
  const current = objectSection(config);
  const streamingEnabled = asrProfileRequiresStreaming(draft.profile) || Boolean(draft.streamingEnabled);
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
      profile: normalizeAsrProfile(draft.profile),
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
      diarization_enabled: Boolean(draft.diarizationEnabled),
      diar_backend: normalizeDiarizationBackend(draft.diarizationBackend),
      diarization_sidecar_enabled: Boolean(draft.diarizationSidecarEnabled),
      diarization_queue_size: normalizeInteger(draft.diarizationQueueSize, 50, 1, 500),
      diar_sherpa_embedding_model_path: String(draft.diarSherpaEmbeddingModelPath || "").trim(),
      diar_sherpa_provider: normalizeDiarizationProvider(draft.diarSherpaProvider),
      diar_sherpa_num_threads: normalizeInteger(draft.diarSherpaNumThreads, 1, 1, 32),
      streaming_enabled: streamingEnabled,
      streaming_chunk_interval_s: normalizeNumber(draft.streamingChunkIntervalS, STREAMING_CHUNK_INTERVAL_FIELD),
      streaming_endpoint_silence_ms: normalizeNumber(draft.streamingEndpointSilenceMs, STREAMING_ENDPOINT_SILENCE_FIELD),
      ...Object.fromEntries(ASR_FIELDS.map((field) => [field.key, normalizeNumber(draft.asr[field.key], field)]))
    },
    models: {
      ...objectSection(current.models),
      cache_dir: String(draft.modelsDirectory || "").trim(),
      use_proxy: Boolean(draft.modelsUseProxy),
      proxy: draft.modelsUseProxy ? buildProxyUrl(draft, { forceEnabled: true }) : ""
    },
    codex: {
      ...objectSection(current.codex),
      enabled: Boolean(draft.assistantEnabled),
      selected_profile: selectedAssistantProfileId(draft),
      profiles: normalizeAssistantProfiles(draft.assistantProfiles),
      proxy: buildProxyUrl(draft)
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
    streamingEnabled: config.asr.streaming_enabled,
    streamingChunkIntervalS: config.asr.streaming_chunk_interval_s,
    streamingEndpointSilenceMs: config.asr.streaming_endpoint_silence_ms,
    profile: config.ui.profile,
    model: config.ui.model,
    outputFile: config.ui.output_file,
    ...config.asr
  };
}
