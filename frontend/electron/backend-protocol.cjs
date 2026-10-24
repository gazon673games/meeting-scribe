'use strict';

const toNumber = (value) => {
  const parsed = Number(String(value || '').trim());
  return Number.isFinite(parsed) ? parsed : 0;
};

const stoppedResultForMethod = (method) => {
  if (method === 'get_resource_usage') {
    return { pid: 0, cpuPct: 0, memoryBytes: 0 };
  }
  if (method === 'list_devices') {
    return { loopback: [], input: [], errors: ['Python backend is stopping'] };
  }
  return null;
};

const isRecord = (value) =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

const parseBackendMessage = (line) => {
  const message = JSON.parse(line);
  if (!isRecord(message)) {
    throw new TypeError('Backend message must be an object');
  }
  if (Object.hasOwn(message, 'event')) {
    if (!isRecord(message.event) || typeof message.event.type !== 'string') {
      throw new TypeError('Backend event must have a string type');
    }
  } else if (
    !Object.hasOwn(message, 'id') ||
    typeof message.ok !== 'boolean' ||
    (!message.ok && !isRecord(message.error))
  ) {
    throw new TypeError('Invalid backend response');
  }
  return message;
};

module.exports = { parseBackendMessage, stoppedResultForMethod, toNumber };
