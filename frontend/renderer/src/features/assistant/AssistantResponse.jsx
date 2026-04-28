export function AssistantResponse({ assistant, busy, response }) {
  const error = response?.ok === false ? response.error || {} : null;
  const providerUnavailable = assistant?.enabled && assistant?.providerAvailable === false;
  const providerMessage = assistant?.providerMessage || "Assistant provider is unavailable";

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
        <p className="muted">{busy ? "Working..." : providerUnavailable ? providerMessage : "No response yet"}</p>
      )}
    </div>
  );
}
