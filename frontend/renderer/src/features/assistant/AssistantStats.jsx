import { formatNumber } from "../../shared/lib/format";
import { StatLine } from "../../shared/ui/StatLine";

export function AssistantStats({ assistant, response, selectedProfile }) {
  const provider = selectedProfile.providerId || selectedProfile.provider || response.provider || "-";
  return (
    <div className="settings-strip">
      <StatLine label="Provider" value={provider} />
      <StatLine label="Active Model" value={selectedProfile.model || "-"} />
      <StatLine label="Response Time" value={response.dtS ? `${formatNumber(response.dtS)}s` : assistant.busy ? "running" : "-"} accent="green" />
      <StatLine label="Last Action" value={assistant.lastRequest?.sourceLabel || "-"} accent="blue" />
    </div>
  );
}
