const DEFAULT_RENDERER_HOST = "127.0.0.1";
const DEFAULT_RENDERER_PORT = 5173;

function parsePort(value, fallback = DEFAULT_RENDERER_PORT) {
  const raw = String(value ?? "").trim();
  if (!raw) return fallback;
  const port = Number.parseInt(raw, 10);
  return Number.isInteger(port) && port > 0 && port <= 65535 ? port : fallback;
}

function rendererHostFromEnv(env = process.env) {
  return String(env.VITE_DEV_SERVER_HOST || env.ELECTRON_RENDERER_HOST || DEFAULT_RENDERER_HOST).trim() || DEFAULT_RENDERER_HOST;
}

function formatUrlHost(host) {
  const value = String(host || DEFAULT_RENDERER_HOST).trim() || DEFAULT_RENDERER_HOST;
  return value.includes(":") && !value.startsWith("[") ? `[${value}]` : value;
}

function rendererUrlFromHostPort(host, port) {
  return `http://${formatUrlHost(host)}:${parsePort(port)}`;
}

function rendererUrlFromEnv(env = process.env) {
  const explicit = String(env.ELECTRON_RENDERER_URL || "").trim();
  if (explicit) return explicit;
  const port = parsePort(env.VITE_DEV_SERVER_PORT || env.ELECTRON_RENDERER_PORT || env.PORT);
  return rendererUrlFromHostPort(rendererHostFromEnv(env), port);
}

module.exports = {
  DEFAULT_RENDERER_HOST,
  DEFAULT_RENDERER_PORT,
  formatUrlHost,
  parsePort,
  rendererHostFromEnv,
  rendererUrlFromEnv,
  rendererUrlFromHostPort
};
