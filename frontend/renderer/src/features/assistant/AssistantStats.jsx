import { formatNumber } from "../../shared/lib/format";
import { StatLine } from "../../shared/ui/StatLine";

function modelBasename(model) {
  const s = String(model || "");
  const slash = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
  const name = slash >= 0 ? s.slice(slash + 1) : s;
  return name.replace(/\.gguf$/i, "");
}

export function AssistantStats({ assistant, response, selectedProfile }) {
  const provider = selectedProfile.providerId || selectedProfile.provider || response.provider || "-";
  const activeModel = selectedProfile.model ? modelBasename(selectedProfile.model) : "-";
  return (
    <div className="settings-strip">
      <StatLine label="Provider" value={provider} />
      <StatLine label="Active Model" value={activeModel} />
      <StatLine label="Response Time" value={response.dtS ? `${formatNumber(response.dtS)}s` : assistant.busy ? "running" : "-"} accent="green" />
      <StatLine label="Last Action" value={assistant.lastRequest?.sourceLabel || "-"} accent="blue" />
    </div>
  );
}
