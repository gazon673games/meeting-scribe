'use strict';

const DEFAULT_RENDERER_HOST = '127.0.0.1';
const DEFAULT_RENDERER_PORT = 5173;

const parsePort = (value, fallback = DEFAULT_RENDERER_PORT) => {
  const raw = String(value ?? '').trim();
  if (!raw) return fallback;
  const port = Number.parseInt(raw, 10);
  return Number.isInteger(port) && port > 0 && port <= 65535 ? port : fallback;
};

const rendererHostFromEnv = (env = process.env) =>
  String(
    env.VITE_DEV_SERVER_HOST ||
      env.ELECTRON_RENDERER_HOST ||
      DEFAULT_RENDERER_HOST,
  ).trim() || DEFAULT_RENDERER_HOST;

const formatUrlHost = (host) => {
  const value =
    String(host || DEFAULT_RENDERER_HOST).trim() || DEFAULT_RENDERER_HOST;
  return value.includes(':') && !value.startsWith('[') ? `[${value}]` : value;
};

const rendererUrlFromHostPort = (host, port) =>
  `http://${formatUrlHost(host)}:${parsePort(port)}`;

const rendererUrlFromEnv = (env = process.env) => {
  const explicit = String(env.ELECTRON_RENDERER_URL || '').trim();
  if (explicit) return explicit;
  const port = parsePort(
    env.VITE_DEV_SERVER_PORT || env.ELECTRON_RENDERER_PORT || env.PORT,
  );
  return rendererUrlFromHostPort(rendererHostFromEnv(env), port);
};

module.exports = {
  DEFAULT_RENDERER_HOST,
  DEFAULT_RENDERER_PORT,
  formatUrlHost,
  parsePort,
  rendererHostFromEnv,
  rendererUrlFromEnv,
  rendererUrlFromHostPort,
};
