export function AssistantResponse({ busy, response }) {
  return (
    <div className="assistant-response">
      <span>Assistant Response</span>
      {response.text ? <p>{response.text}</p> : <p className="muted">{busy ? "Working..." : "No response yet"}</p>}
    </div>
  );
}
