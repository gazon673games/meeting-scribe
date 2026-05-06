export const ASR_PROFILE_ULTRA_FAST = "Ultra Fast";

export const FALLBACK_OPTIONS = {
  languages: ["ru", "en", "auto"],
  asrProfiles: [ASR_PROFILE_ULTRA_FAST, "Realtime", "Quality", "Custom"],
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
  diarizationBackends: ["online", "sherpa_onnx", "nemo", "pyannote"],
  diarizationProviders: ["cpu", "cuda"],
  overloadStrategies: ["drop_old", "keep_all"],
  profileDefaults: {},
  streamingLockedProfiles: [ASR_PROFILE_ULTRA_FAST]
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
export const STREAMING_CHUNK_INTERVAL_FIELD = { defaultValue: 1, min: 0.1, max: 5 };
export const STREAMING_ENDPOINT_SILENCE_FIELD = { defaultValue: 300, min: 50, max: 5000 };

export const ASSISTANT_REASONING_OPTIONS = ["low", "medium", "high", "xhigh"];
export const ASSISTANT_PROVIDER_OPTIONS = [
  { id: "codex", label: "Codex CLI", defaultBaseUrl: "" },
  { id: "ollama", label: "Ollama", defaultBaseUrl: "http://127.0.0.1:11434" },
  { id: "openai_local", label: "Local OpenAI", defaultBaseUrl: "http://127.0.0.1:1234/v1" },
  { id: "local", label: "Local GGUF (llama.cpp)", defaultBaseUrl: "" }
];
export const DIARIZATION_BACKENDS = ["online", "sherpa_onnx", "nemo", "pyannote"];
export const DIARIZATION_PROVIDERS = ["cpu", "cuda"];
