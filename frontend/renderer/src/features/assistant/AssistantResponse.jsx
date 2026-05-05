import React from "react";
import { LogIn } from "lucide-react";

export function AssistantResponse({ assistant, busy, messages, onAuthorize }) {
  const bottomRef = React.useRef(null);
  const providerUnavailable = assistant?.enabled && assistant?.providerAvailable === false;
  const providerMessage = assistant?.providerMessage || "Assistant provider is unavailable";
  const providerSuggestion = assistant?.providerSuggestion || "";
  const canAuthorize =
    providerUnavailable &&
    assistant?.providerErrorCode === "auth_error" &&
    assistant?.providerLoginSupported !== false &&
    typeof onAuthorize === "function";

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  return (
    <div className="assistant-response assistant-chat">
      {messages.length === 0 ? (
        <div className="assistant-chat-empty">
          {providerUnavailable ? (
            <>
              <p className="muted">{providerMessage}</p>
              {providerSuggestion ? <em className="assistant-provider-hint">{providerSuggestion}</em> : null}
              {canAuthorize ? (
                <div className="assistant-provider-actions">
                  <button className="assistant-auth-button" disabled={busy} onClick={onAuthorize} type="button">
                    <LogIn size={15} />
                    Authorize Codex
                  </button>
                </div>
              ) : null}
            </>
          ) : (
            <p className="muted">{busy ? "Working..." : "No responses yet"}</p>
          )}
        </div>
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
}

function ChatMessage({ message }) {
  if (message.role === "user") {
    return (
      <div className="chat-msg chat-msg-user">
        <span className="chat-msg-label">{message.sourceLabel || "you"}</span>
        <p>{message.text}</p>
      </div>
    );
  }
  const isError = message.ok === false;
  const error = isError ? (message.error || {}) : null;
  return (
    <div className={`chat-msg chat-msg-assistant${isError ? " chat-msg-error" : ""}`}>
      <p>{message.text}</p>
      {error?.code || error?.suggestion ? (
        <div className="assistant-error-meta">
          {error.code ? <b>{error.code}</b> : null}
          {error.suggestion ? <em>{error.suggestion}</em> : null}
        </div>
      ) : null}
    </div>
  );
}
