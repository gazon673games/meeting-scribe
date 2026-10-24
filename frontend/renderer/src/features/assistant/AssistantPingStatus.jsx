const AUTH_ERROR_PATTERN = /auth|unauthorized|not_logged|login/i;

const shouldHidePingStatus = (ping, idleRequiresTs) => {
  if (!ping) return true;
  if (!idleRequiresTs) return false;
  return !ping.busy && (ping.ts === null || ping.ts === undefined);
};

const pingStatusClass = (ping) => {
  if (ping.busy) return 'ping-status ping-busy';
  if (ping.ok) return 'ping-status ping-ok';
  return 'ping-status ping-err';
};

const fallbackText = (value, fallback) => {
  const text = String(value || '').trim();
  return text ? text : fallback;
};

const pingStatusLabel = (ping, { errorLabel, showBusy, showMessage }) => {
  if (ping.busy) {
    return showBusy ? 'pinging...' : '';
  }
  if (ping.ok) {
    return 'ok';
  }
  if (AUTH_ERROR_PATTERN.test(String(ping.errorCode))) {
    return 'not authorized';
  }
  return showMessage ? fallbackText(ping.message, errorLabel) : errorLabel;
};

export const AssistantPingStatus = ({
  ping,
  errorLabel = 'error',
  idleRequiresTs = false,
  showBusy = true,
  showMessage = false,
}) => {
  if (shouldHidePingStatus(ping, idleRequiresTs)) {
    return null;
  }
  const label = pingStatusLabel(ping, { errorLabel, showBusy, showMessage });
  return label ? <span className={pingStatusClass(ping)}>{label}</span> : null;
};
