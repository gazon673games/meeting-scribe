import { ASSISTANT_PROVIDER_OPTIONS } from "../../../entities/settings/model";

export function normalizeProvider(provider) {
  const value = String(provider || "codex").trim().toLowerCase();
  return ASSISTANT_PROVIDER_OPTIONS.some((option) => option.id === value) ? value : "codex";
}

export function runtimeLabel(provider) {
  if (provider === "ollama") return "Ollama";
  if (provider === "openai_local") return "OpenAI-compatible";
  if (provider === "local") return "Local GGUF";
  return "Local";
}

export function defaultBaseUrl(provider) {
  return ASSISTANT_PROVIDER_OPTIONS.find((option) => option.id === provider)?.defaultBaseUrl || "";
}

export function modelBasename(model) {
  const value = String(model || "");
  const slash = Math.max(value.lastIndexOf("/"), value.lastIndexOf("\\"));
  const name = slash >= 0 ? value.slice(slash + 1) : value;
  return name.replace(/\.gguf$/i, "");
}

export function newProfile(id, provider) {
  const normalizedProvider = normalizeProvider(provider);
  const labelMap = { codex: "Codex", local: "Local GGUF", ollama: "Ollama", openai_local: "Local Model" };
  return {
    id,
    label: labelMap[normalizedProvider] || "Profile",
    provider: normalizedProvider,
    prompt: "Give concise, practical support based on the session context.",
    model: "",
    reasoning_effort: normalizedProvider === "codex" ? "low" : "",
    codex_profile: "",
    base_url: defaultBaseUrl(normalizedProvider),
    api_key: "",
    temperature: "",
    max_tokens: 0,
    answer_prompt: "",
    extra_args: [],
    offline: normalizedProvider !== "codex",
    gpu_layers: normalizedProvider === "local" ? 0 : undefined,
    context_size: normalizedProvider === "local" ? 4096 : undefined
  };
}

export function runtimePatch(profile, provider) {
  const normalizedProvider = normalizeProvider(provider);
  const previousProvider = normalizeProvider(profile.provider);
  const currentUrl = String(profile.base_url || "");
  const baseUrl =
    currentUrl && currentUrl !== defaultBaseUrl(previousProvider) ? currentUrl : defaultBaseUrl(normalizedProvider);

  const patch = { provider: normalizedProvider, base_url: baseUrl, offline: normalizedProvider !== "codex" };
  if (normalizedProvider === "local") {
    patch.gpu_layers = profile.gpu_layers ?? 0;
    patch.context_size = profile.context_size ?? 4096;
  }
  return patch;
}

export function uniqueProfileId(profiles, provider) {
  const base = normalizeProvider(provider).replace(/[^a-z0-9_]+/g, "_") || "assistant";
  const ids = new Set(profiles.map((profile) => profile.id));
  let index = profiles.length + 1;
  let id = `${base}_${index}`;
  while (ids.has(id)) {
    index += 1;
    id = `${base}_${index}`;
  }
  return id;
}
