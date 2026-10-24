import { classNames } from '../../shared/lib/classNames';
import React from 'react';
import { Info, LogIn } from 'lucide-react';

const AssistantEmpty = ({
  hasProviderNotice,
  isDisabled,
  providerMessage,
  providerHint,
  canAuthorize,
  busy,
  onAuthorize,
}) => (
  <div className="assistant-chat-empty">
    {hasProviderNotice ? (
      <>
        <p className="muted assistant-provider-notice">
          <Info aria-hidden="true" size={16} />
          <span>{isDisabled ? 'Assistant is disabled' : providerMessage}</span>
        </p>
        {providerHint ? (
          <em className="assistant-provider-hint">{providerHint}</em>
        ) : null}
        {canAuthorize ? (
          <div className="assistant-provider-actions">
            <button
              className="assistant-auth-button"
              disabled={busy}
              onClick={onAuthorize}
              type="button"
            >
              <LogIn size={15} />
              Authorize Codex
            </button>
          </div>
        ) : null}
      </>
    ) : (
      <p className="muted">{busy ? 'Working...' : 'No responses yet'}</p>
    )}
  </div>
);

const ChatMessage = ({ message }) => {
  if (message.role === 'user') {
    return (
      <div className="chat-msg chat-msg-user">
        <span className="chat-msg-label">{message.sourceLabel || 'you'}</span>
        <p>{message.text}</p>
      </div>
    );
  }
  const isError = message.ok === false;
  const error = isError ? message.error || {} : null;
  return (
    <div
      className={classNames(
        'chat-msg chat-msg-assistant',
        isError && 'chat-msg-error',
      )}
    >
      <p>{message.text}</p>
      {error?.code || error?.suggestion ? (
        <div className="assistant-error-meta">
          {error.code ? <b>{error.code}</b> : null}
          {error.suggestion ? <em>{error.suggestion}</em> : null}
        </div>
      ) : null}
    </div>
  );
};

export const AssistantResponse = ({
  assistant,
  busy,
  messages,
  onAuthorize,
}) => {
  const bottomRef = React.useRef(null);
  const providerUnavailable =
    assistant?.enabled && assistant?.providerAvailable === false;
  const providerUnchecked =
    assistant?.enabled && assistant?.providerAvailable === undefined;
  const providerMessage =
    assistant?.providerMessage ||
    (providerUnchecked
      ? 'Connection not checked'
      : 'Assistant provider is unavailable');
  const providerSuggestion = assistant?.providerSuggestion || '';
  const providerHint =
    assistant?.enabled === false
      ? 'Enable the assistant in Settings to use this panel.'
      : providerSuggestion;
  const canAuthorize =
    providerUnavailable &&
    assistant?.providerErrorCode === 'auth_error' &&
    assistant?.providerLoginSupported !== false &&
    typeof onAuthorize === 'function';

  const hasProviderNotice =
    providerUnavailable || providerUnchecked || assistant?.enabled === false;
  const isEmptyUnavailable = !messages.length && hasProviderNotice;
  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, busy]);

  return (
    <div
      className={classNames(
        'assistant-response assistant-chat',
        isEmptyUnavailable && 'assistant-chat-unavailable',
      )}
    >
      {messages.length === 0 ? (
        <AssistantEmpty
          hasProviderNotice={hasProviderNotice}
          isDisabled={assistant?.enabled === false}
          providerMessage={providerMessage}
          providerHint={providerHint}
          canAuthorize={canAuthorize}
          busy={busy}
          onAuthorize={onAuthorize}
        />
      ) : (
        <div className="assistant-chat-list">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}
          {busy ? <p className="assistant-chat-busy">Working...</p> : null}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
};
