import { LogIn } from "lucide-react";

export function AssistantResponse({ assistant, busy, response, onAuthorize }) {
  const error = response?.ok === false ? response.error || {} : null;
  const providerUnavailable = assistant?.enabled && assistant?.providerAvailable === false;
  const providerMessage = assistant?.providerMessage || "Assistant provider is unavailable";
  const providerSuggestion = assistant?.providerSuggestion || "";
  const canAuthorize =
    providerUnavailable &&
    assistant?.providerErrorCode === "auth_error" &&
    assistant?.providerLoginSupported !== false &&
    typeof onAuthorize === "function";

  return (
    <div className="assistant-response">
      <span>Assistant Response</span>
      {response.text ? (
        <>
          <p className={error ? "error" : ""}>{response.text}</p>
          {error?.code || error?.suggestion ? (
            <div className="assistant-error-meta">
              {error.code ? <b>{error.code}</b> : null}
              {error.suggestion ? <em>{error.suggestion}</em> : null}
            </div>
          ) : null}
        </>
      ) : (
        <>
          <p className="muted">{busy ? "Working..." : providerUnavailable ? providerMessage : "No response yet"}</p>
          {providerSuggestion && providerUnavailable ? <em className="assistant-provider-hint">{providerSuggestion}</em> : null}
          {canAuthorize ? (
            <div className="assistant-provider-actions">
              <button className="assistant-auth-button" disabled={busy} onClick={onAuthorize} type="button">
                <LogIn size={15} />
                Authorize Codex
              </button>
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
