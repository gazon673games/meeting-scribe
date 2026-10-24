'use strict';

const path = require('node:path');
const {
  parseBackendMessage,
  stoppedResultForMethod,
  toNumber,
} = require('./backend-protocol.cjs');

const EVENT_UPDATES = Object.freeze({
  backend_ready: (backend) => {
    backend.ready = true;
  },
  model_download_updated: (backend, event) => {
    backend.activeModelDownloads = toNumber(event.activeDownloads);
  },
  diarization_model_download_updated: (backend, event) => {
    backend.activeSpeakerIdDownloads = toNumber(event.activeDownloads);
  },
  llm_model_download_updated: (backend, event) => {
    backend.activeLlmDownloads = toNumber(event.activeDownloads);
  },
});

const createPythonBackendClass = ({
  spawn,
  existsSync,
  isPackaged,
  isQuitting,
  logger = console,
}) => {
  class PythonBackend {
    constructor({
      root,
      backendRoot,
      runtimeRoot,
      maxPendingRequests = 256,
      requestTimeoutMs = 600000,
    }) {
      if (!Number.isInteger(maxPendingRequests) || maxPendingRequests < 1) {
        throw new RangeError(
          'Pending request limit must be a positive integer',
        );
      }
      if (!Number.isFinite(requestTimeoutMs) || requestTimeoutMs <= 0) {
        throw new RangeError('Request timeout must be positive');
      }
      this.root = root;
      this.backendRoot = backendRoot;
      this.runtimeRoot = runtimeRoot;
      this.process = null;
      this.nextId = 1;
      this.pending = new Map();
      this.stdoutBuffer = '';
      this.windows = new Set();
      this.eventBuffer = [];
      this.ready = false;
      this.lastError = '';
      this.activeDownloads = 0;
      this.activeModelDownloads = 0;
      this.activeSpeakerIdDownloads = 0;
      this.activeLlmDownloads = 0;
      this.stoppingProcess = null;
      this.maxPendingRequests = maxPendingRequests;
      this.requestTimeoutMs = requestTimeoutMs;
    }

    attachWindow(window) {
      if (!window) {
        return;
      }
      this.windows.add(window);
      window.on('closed', () => this.detachWindow(window));
    }

    detachWindow(window) {
      this.windows.delete(window);
    }

    recentEvents() {
      return this.eventBuffer.map((event) => ({ ...event }));
    }

    start() {
      if (this.process) {
        return;
      }
      const command = this.findBackendCommand();
      const child = spawn(command.executable, command.args, {
        cwd: command.cwd,
        env: {
          ...process.env,
          MEETING_SCRIBE_RUNTIME_ROOT: this.runtimeRoot,
          PYTHONIOENCODING: 'utf-8',
          PYTHONUNBUFFERED: '1',
          PYTHONUTF8: '1',
        },
        windowsHide: true,
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      this.process = child;
      this.stoppingProcess = null;
      this.stdoutBuffer = '';

      child.stdout.setEncoding('utf8');
      child.stdout.on('data', (chunk) => {
        if (this.process === child) this.handleStdout(chunk);
      });
      child.stderr.setEncoding('utf8');
      child.stderr.on('data', (chunk) => {
        if (this.process !== child) return;
        this.lastError = String(chunk);
        logger.error('[python-backend]', String(chunk).trim());
        this.sendEvent({ type: 'backend_stderr', text: String(chunk) });
      });
      child.on('exit', (code, signal) => {
        if (this.process !== child && this.stoppingProcess !== child) return;
        const expectedStop = this.stoppingProcess === child || isQuitting();
        this.ready = false;
        if (this.process === child) {
          this.process = null;
        }
        if (this.stoppingProcess === child) {
          this.stoppingProcess = null;
        }
        this.activeDownloads = 0;
        this.activeModelDownloads = 0;
        this.activeSpeakerIdDownloads = 0;
        this.activeLlmDownloads = 0;
        const status = `${code ?? 'null'}, ${signal ?? 'null'}`;
        const message = `Python backend exited (${status})`;
        for (const pending of this.pending.values()) {
          clearTimeout(pending.timer);
          if (expectedStop) {
            pending.resolve(stoppedResultForMethod(pending.method));
          } else {
            pending.reject(new Error(message));
          }
        }
        this.pending.clear();
        if (!expectedStop) {
          this.sendEvent({ type: 'backend_exit', code, signal });
        }
      });
      const handleTransportError = (error) => {
        if (this.process !== child) return;
        this.process = null;
        this.ready = false;
        this.lastError = error.message;
        this.activeDownloads = 0;
        this.activeModelDownloads = 0;
        this.activeSpeakerIdDownloads = 0;
        this.activeLlmDownloads = 0;
        this.rejectPending(error);
        child.kill();
        logger.error('[python-backend] process failed', error);
        this.sendEvent({ type: 'backend_exit', code: null, signal: null });
      };
      child.on('error', handleTransportError);
      child.stdin.on('error', handleTransportError);
      child.stdout.on('error', handleTransportError);
      child.stderr.on('error', handleTransportError);
    }

    stop() {
      if (!this.process) {
        return;
      }
      const child = this.process;
      this.stoppingProcess = child;
      this.process = null;
      this.ready = false;
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timer);
        pending.resolve(stoppedResultForMethod(pending.method));
      }
      this.pending.clear();
      child.kill();
    }

    rejectPending(error) {
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timer);
        pending.reject(error);
      }
      this.pending.clear();
    }

    status() {
      return {
        ready: this.ready,
        running: Boolean(this.process),
        lastError: this.lastError,
      };
    }

    request(method, params = {}) {
      if (isQuitting()) {
        return Promise.resolve(stoppedResultForMethod(method));
      }
      if (this.pending.size >= this.maxPendingRequests) {
        return Promise.reject(new Error('Backend request queue is full'));
      }
      try {
        this.start();
      } catch (error) {
        return Promise.reject(error);
      }
      if (!this.process || !this.process.stdin.writable) {
        return Promise.reject(new Error('Python backend is not writable'));
      }

      const id = String(this.nextId++);
      let payload;
      try {
        payload = JSON.stringify({ id, method, params });
      } catch (error) {
        return Promise.reject(error);
      }
      return new Promise((resolve, reject) => {
        const rejectRequest = (error) => {
          const pending = this.pending.get(id);
          if (!pending) return;
          clearTimeout(pending.timer);
          this.pending.delete(id);
          reject(error);
        };
        const timer = setTimeout(() => {
          rejectRequest(new Error(`Backend request timed out: ${method}`));
        }, this.requestTimeoutMs);
        timer.unref();
        this.pending.set(id, { resolve, reject, method, timer });
        try {
          this.process.stdin.write(`${payload}\n`, 'utf8', (error) => {
            if (error) rejectRequest(error);
          });
        } catch (error) {
          rejectRequest(error);
        }
      });
    }

    handleStdout(chunk) {
      this.stdoutBuffer += chunk;
      while (true) {
        const newlineIndex = this.stdoutBuffer.indexOf('\n');
        if (newlineIndex < 0) {
          break;
        }
        const line = this.stdoutBuffer.slice(0, newlineIndex).trim();
        this.stdoutBuffer = this.stdoutBuffer.slice(newlineIndex + 1);
        if (line) {
          this.handleMessageLine(line);
        }
      }
    }

    handleMessageLine(line) {
      let message;
      try {
        message = parseBackendMessage(line);
      } catch (error) {
        this.lastError = `Invalid backend message: ${line}`;
        logger.error(this.lastError, error);
        return;
      }

      if (message.event) {
        if (Object.hasOwn(EVENT_UPDATES, message.event.type)) {
          EVENT_UPDATES[message.event.type](this, message.event);
          this.activeDownloads =
            this.activeModelDownloads +
            this.activeSpeakerIdDownloads +
            this.activeLlmDownloads;
        }
        this.sendEvent(message.event);
        return;
      }

      const pending = this.pending.get(String(message.id));
      if (!pending) {
        return;
      }
      this.pending.delete(String(message.id));
      clearTimeout(pending.timer);
      if (message.ok) {
        pending.resolve(message.result);
      } else {
        const backendError = new Error(
          message.error?.message || 'Backend request failed',
        );
        backendError.name = message.error?.type || 'BackendError';
        pending.reject(backendError);
      }
    }

    sendEvent(event) {
      const record = { ts: Date.now() / 1000, ...event };
      this.eventBuffer.push(record);
      if (this.eventBuffer.length > 1200) {
        this.eventBuffer.splice(0, this.eventBuffer.length - 1200);
      }
      for (const window of this.windows) {
        if (window.isDestroyed()) {
          this.windows.delete(window);
          continue;
        }
        window.webContents.send('backend:event', record);
      }
    }

    findPython() {
      const candidates =
        process.platform === 'win32'
          ? [
              path.join(this.root, '.venv', 'Scripts', 'python.exe'),
              path.join(this.root, '.venv', 'Scripts', 'python3.exe'),
              'python',
            ]
          : [
              path.join(this.root, '.venv', 'bin', 'python'),
              'python3',
              'python',
            ];
      return (
        candidates.find((candidate) =>
          candidate.includes(path.sep) ? existsSync(candidate) : true,
        ) || 'python'
      );
    }

    findBackendCommand() {
      if (isPackaged()) {
        const executableName =
          process.platform === 'win32'
            ? 'meeting-scribe-backend.exe'
            : 'meeting-scribe-backend';
        const executable = path.join(this.backendRoot, executableName);
        if (!existsSync(executable)) {
          throw new Error(
            `Packaged Python backend was not found: ${executable}`,
          );
        }
        return { executable, args: [], cwd: this.runtimeRoot };
      }

      return {
        executable: this.findPython(),
        args: [path.join(this.root, 'backend', 'main_electron_backend.py')],
        cwd: this.root,
      };
    }
  }

  return PythonBackend;
};

module.exports = { createPythonBackendClass };
