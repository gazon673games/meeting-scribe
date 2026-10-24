import {
  ASSISTANT_PROVIDER_OPTIONS,
  ASSISTANT_REASONING_OPTIONS,
  DIARIZATION_BACKENDS,
  DIARIZATION_PROVIDERS,
  FALLBACK_OPTIONS,
  THEME_OPTIONS,
} from './constants.js';

const DEFAULT_ASSISTANT_PROFILES = [
  {
    id: 'default',
    label: 'Default',
    provider: 'codex',
    prompt:
      'Support a realtime interview. Give short, practical, accurate ' +
      'responses from the session log.',
    model: '',
    reasoning_effort: 'low',
    codex_profile: '',
    base_url: '',
    api_key: '',
    temperature: '',
    max_tokens: 0,
    answer_prompt:
      'Command ANSWER: provide a quick candidate response ' +
      'for the latest question.',
    extra_args: [],
  },
];

export const asrProfileRequiresStreaming = (
  profile,
  options = FALLBACK_OPTIONS,
) => {
  const wanted = String(profile || '')
    .trim()
    .toLowerCase();
  const lockedProfiles = Array.isArray(options?.streamingLockedProfiles)
    ? options.streamingLockedProfiles
    : FALLBACK_OPTIONS.streamingLockedProfiles;
  return lockedProfiles.some(
    (item) =>
      String(item || '')
        .trim()
        .toLowerCase() === wanted,
  );
};

export const objectSection = (value) =>
  value && typeof value === 'object' && !Array.isArray(value) ? value : {};

export const normalizeDiarizationBackend = (value) => {
  const backend = String(value || 'online')
    .trim()
    .toLowerCase();
  return DIARIZATION_BACKENDS.includes(backend) ? backend : 'online';
};

export const normalizeDiarizationProvider = (value) => {
  const provider = String(value || 'cpu')
    .trim()
    .toLowerCase();
  return DIARIZATION_PROVIDERS.includes(provider) ? provider : 'cpu';
};

export const normalizeAsrProfile = (value) => {
  const profile = String(value || '').trim();
  if (!profile) {
    return 'Realtime';
  }
  return profile;
};

export const boolWithDefault = (value, fallback) =>
  value === undefined || value === null ? Boolean(fallback) : Boolean(value);

export const normalizeTheme = (value) => {
  const theme = String(value || '')
    .trim()
    .toLowerCase();
  return THEME_OPTIONS.some((option) => option.id === theme) ? theme : 'dark';
};

export const normalizeNumber = (value, field) => {
  const parsed = Number(String(value ?? field.defaultValue).replace(',', '.'));
  let next = Number.isFinite(parsed) ? parsed : field.defaultValue;
  next = Math.max(Number(field.min), Math.min(Number(field.max), next));
  return field.integer ? Math.round(next) : next;
};

export const normalizeInteger = (value, fallback, min, max) => {
  const parsed = Number(String(value ?? fallback).replace(',', '.'));
  const next = Number.isFinite(parsed) ? Math.round(parsed) : fallback;
  return Math.max(Number(min), Math.min(Number(max), next));
};

export const normalizeOptionalNumber = (value, min, max) => {
  const raw = String(value ?? '').trim();
  if (!raw) {
    return '';
  }
  const parsed = Number(raw.replace(',', '.'));
  if (!Number.isFinite(parsed)) {
    return '';
  }
  return Math.max(Number(min), Math.min(Number(max), parsed));
};

const normalizeAssistantProfileId = (item, index) => {
  const fallback = `profile_${index + 1}`;
  const id = String(item.id || item.label || fallback).trim();
  return id || fallback;
};

const normalizeAssistantProfileLabel = (item, id) =>
  String(item.label || id).trim() || id;

const normalizeAssistantProvider = (value) => {
  const provider = String(value || 'codex')
    .trim()
    .toLowerCase();
  return ASSISTANT_PROVIDER_OPTIONS.some((option) => option.id === provider)
    ? provider
    : 'codex';
};

const normalizeReasoningEffort = (value) => {
  const effort = String(value || '')
    .trim()
    .toLowerCase();
  return ASSISTANT_REASONING_OPTIONS.includes(effort) ? effort : '';
};

const normalizeAssistantExtraArgs = (value) => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((arg) => String(arg)).filter(Boolean);
};

const normalizeOptionalInteger = (value, fallback, min, max) => {
  if (value === undefined) {
    return undefined;
  }
  return normalizeInteger(value, fallback, min, max);
};

const firstTruthyProperty = (item, keys, fallback = '') => {
  for (const key of keys) {
    if (item[key]) {
      return item[key];
    }
  }
  return fallback;
};

const firstDefinedProperty = (item, keys, fallback) => {
  for (const key of keys) {
    if (item[key] !== undefined && item[key] !== null) {
      return item[key];
    }
  }
  return fallback;
};

const normalizeAssistantProfile = (profile, index) => {
  const item = objectSection(profile);
  const id = normalizeAssistantProfileId(item, index);
  const provider = firstTruthyProperty(
    item,
    ['provider', 'provider_id', 'providerId'],
    'codex',
  );
  const reasoningEffort = firstTruthyProperty(
    item,
    ['reasoning_effort', 'reasoningEffort'],
    '',
  );
  const codexProfile = firstTruthyProperty(
    item,
    ['codex_profile', 'codexProfile'],
    '',
  );
  const baseUrl = firstTruthyProperty(
    item,
    ['base_url', 'baseUrl', 'endpoint'],
    '',
  );
  const apiKey = firstTruthyProperty(item, ['api_key', 'apiKey'], '');
  const answerPrompt = firstTruthyProperty(
    item,
    ['answer_prompt', 'answerPrompt'],
    '',
  );
  const extraArgs = firstTruthyProperty(item, ['extra_args', 'extraArgs'], []);
  const maxTokens = firstDefinedProperty(item, ['max_tokens', 'maxTokens'], 0);
  return {
    id,
    label: normalizeAssistantProfileLabel(item, id),
    provider: normalizeAssistantProvider(provider),
    prompt: String(item.prompt || ''),
    model: String(item.model || ''),
    reasoning_effort: normalizeReasoningEffort(reasoningEffort),
    codex_profile: String(codexProfile),
    base_url: String(baseUrl),
    api_key: String(apiKey),
    temperature: normalizeOptionalNumber(item.temperature, 0, 2),
    max_tokens: normalizeInteger(maxTokens, 0, 0, 200000),
    answer_prompt: String(answerPrompt),
    extra_args: normalizeAssistantExtraArgs(extraArgs),
    offline: Boolean(item.offline),
    gpu_layers: normalizeOptionalInteger(item.gpu_layers, 0, 0, 999),
    context_size: normalizeOptionalInteger(item.context_size, 4096, 64, 131072),
  };
};

export const normalizeAssistantProfiles = (value) => {
  const rawProfiles = Array.isArray(value) ? value : DEFAULT_ASSISTANT_PROFILES;
  const used = new Set();
  const profiles = [];
  for (const [index, raw] of rawProfiles.entries()) {
    const profile = normalizeAssistantProfile(raw, index);
    let id = profile.id;
    let suffix = 2;
    while (used.has(id)) {
      id = `${profile.id}-${suffix}`;
      suffix += 1;
    }
    used.add(id);
    profiles.push({ ...profile, id });
  }
  return profiles;
};

export const selectedAssistantProfileId = (draft) => {
  const profiles = normalizeAssistantProfiles(draft.assistantProfiles);
  const wanted = String(draft.assistantSelectedProfileId || '').trim();
  if (profiles.some((profile) => profile.id === wanted)) {
    return wanted;
  }
  return profiles[0]?.id || '';
};

const emptyProxyDraft = () => ({
  enabled: false,
  scheme: 'http',
  host: '127.0.0.1',
  port: '10808',
  username: '',
  password: '',
});

const normalizeProxyScheme = (value) => {
  const scheme = String(value || 'http')
    .trim()
    .toLowerCase();
  return ['http', 'https', 'socks5'].includes(scheme) ? scheme : 'http';
};

export const buildProxyUrl = (draft, options = {}) => {
  if (!draft.assistantProxyEnabled && !options.forceEnabled) {
    return '';
  }
  const scheme = normalizeProxyScheme(draft.assistantProxyScheme);
  const host =
    String(draft.assistantProxyHost || '127.0.0.1').trim() || '127.0.0.1';
  const port = String(draft.assistantProxyPort || '').trim();
  const username = String(draft.assistantProxyUsername || '').trim();
  const password = String(draft.assistantProxyPassword || '');
  const passwordPart = password ? `:${encodeURIComponent(password)}` : '';
  const auth = username
    ? `${encodeURIComponent(username)}${passwordPart}@`
    : '';
  return port
    ? `${scheme}://${auth}${host}:${port}`
    : `${scheme}://${auth}${host}`;
};

const decodeUrlPart = (value) => {
  try {
    return decodeURIComponent(String(value || ''));
  } catch {
    return String(value || '');
  }
};

export const parseProxy = (value) => {
  const raw = String(value ?? '').trim();
  if (!raw) {
    return emptyProxyDraft();
  }
  try {
    const url = raw.includes('://') ? new URL(raw) : new URL(`http://${raw}`);
    return {
      enabled: true,
      scheme: normalizeProxyScheme(url.protocol.replace(':', '')),
      host: url.hostname || '127.0.0.1',
      port: url.port || '',
      username: decodeUrlPart(url.username),
      password: decodeUrlPart(url.password),
    };
  } catch {
    return {
      ...emptyProxyDraft(),
      enabled: true,
      host: raw.replace(/^(https?|socks5):\/\//, ''),
    };
  }
};
