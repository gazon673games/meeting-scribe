const { spawn } = require("node:child_process");
const http = require("node:http");
const net = require("node:net");
const path = require("node:path");

const {
  DEFAULT_RENDERER_HOST,
  DEFAULT_RENDERER_PORT,
  parsePort,
  rendererHostFromEnv,
  rendererUrlFromHostPort
} = require("./renderer-url.cjs");

const projectRoot = path.resolve(__dirname, "..", "..");

function firstConfiguredPort(env = process.env) {
  for (const key of ["VITE_DEV_SERVER_PORT", "ELECTRON_RENDERER_PORT", "PORT"]) {
    if (String(env[key] || "").trim()) {
      return { key, port: parsePort(env[key], 0) };
    }
  }
  return null;
}

function endpointFromExplicitUrl(rawUrl) {
  const parsed = new URL(rawUrl);
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error(`ELECTRON_RENDERER_URL must be http(s), got ${parsed.protocol}`);
  }
  const host = parsed.hostname.replace(/^\[(.*)]$/, "$1") || DEFAULT_RENDERER_HOST;
  const port = parsePort(parsed.port, parsed.protocol === "https:" ? 443 : 80);
  return { host, port, url: rawUrl };
}

async function isPortAvailable(host, port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.listen({ host, port }, () => {
      server.close(() => resolve(true));
    });
  });
}

async function findOpenPort(host, preferredPort, { allowFallback, attempts = 30, probe = isPortAvailable } = {}) {
  const start = parsePort(preferredPort, DEFAULT_RENDERER_PORT);
  for (let offset = 0; offset < (allowFallback ? attempts : 1); offset += 1) {
    const port = start + offset;
    if (port > 65535) break;
    if (await probe(host, port)) return port;
  }
  const range = allowFallback ? `${start}-${Math.min(65535, start + attempts - 1)}` : String(start);
  throw new Error(`No free renderer dev port found on ${host} (${range}).`);
}

async function chooseRendererEndpoint(env = process.env, probe = isPortAvailable) {
  const explicitUrl = String(env.ELECTRON_RENDERER_URL || "").trim();
  if (explicitUrl) {
    const endpoint = endpointFromExplicitUrl(explicitUrl);
    await findOpenPort(endpoint.host, endpoint.port, { allowFallback: false, probe });
    return endpoint;
  }

  const configuredPort = firstConfiguredPort(env);
  const host = rendererHostFromEnv(env);
  const preferredPort = configuredPort?.port || DEFAULT_RENDERER_PORT;
  const port = await findOpenPort(host, preferredPort, { allowFallback: !configuredPort, probe });
  return { host, port, url: rendererUrlFromHostPort(host, port) };
}

function resolveViteCli() {
  const packageJsonPath = require.resolve("vite/package.json", { paths: [projectRoot] });
  const packageJson = require(packageJsonPath);
  const binPath = typeof packageJson.bin === "object" ? packageJson.bin.vite : packageJson.bin;
  return path.join(path.dirname(packageJsonPath), binPath || "bin/vite.js");
}

function waitForHttp(url, { timeoutMs = 30000, intervalMs = 250, isAlive = () => true } = {}) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const check = () => {
      if (!isAlive()) {
        reject(new Error(`Renderer dev server exited before ${url} became reachable.`));
        return;
      }
      const request = http.get(url, (response) => {
        response.resume();
        resolve();
      });
      request.once("error", () => {
        if (Date.now() >= deadline) {
          reject(new Error(`Renderer dev server did not become reachable at ${url}.`));
          return;
        }
        setTimeout(check, intervalMs);
      });
      request.setTimeout(intervalMs, () => request.destroy());
    };
    check();
  });
}

function terminate(child) {
  if (child && child.exitCode === null && !child.killed) {
    child.kill();
  }
}

async function runDev() {
  const endpoint = await chooseRendererEndpoint(process.env);
  const env = {
    ...process.env,
    ELECTRON_RENDERER_URL: endpoint.url,
    VITE_DEV_SERVER_HOST: endpoint.host,
    VITE_DEV_SERVER_PORT: String(endpoint.port)
  };

  console.log(`Starting renderer at ${endpoint.url}`);
  const vite = spawn(process.execPath, [resolveViteCli(), "--config", "vite.config.js"], {
    cwd: projectRoot,
    env,
    stdio: "inherit"
  });

  let electron = null;
  let shuttingDown = false;
  let viteRunning = true;

  function finish(code = 0) {
    if (shuttingDown) return;
    shuttingDown = true;
    terminate(electron);
    terminate(vite);
    process.exit(code);
  }

  vite.once("exit", () => {
    viteRunning = false;
    if (electron && !shuttingDown) {
      console.error("Renderer dev server exited; stopping Electron.");
      finish(1);
    }
  });

  await waitForHttp(endpoint.url, { isAlive: () => viteRunning });

  electron = spawn(process.execPath, [path.join(__dirname, "run-electron.cjs")], {
    cwd: projectRoot,
    env,
    stdio: "inherit"
  });

  process.once("SIGINT", () => finish(0));
  process.once("SIGTERM", () => finish(0));

  electron.once("exit", (code) => {
    finish(code ?? 0);
  });
}

if (require.main === module) {
  runDev().catch((error) => {
    console.error(error.message || error);
    process.exit(1);
  });
}

module.exports = {
  chooseRendererEndpoint,
  endpointFromExplicitUrl,
  findOpenPort,
  firstConfiguredPort,
  resolveViteCli,
  waitForHttp
};
