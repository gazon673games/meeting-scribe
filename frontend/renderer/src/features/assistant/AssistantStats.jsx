import { formatNumber, modelBasename } from '../../shared/lib/format';
import { StatLine } from '../../shared/ui/StatLine';

export const AssistantStats = ({ assistant, response, selectedProfile }) => {
  const provider =
    selectedProfile.providerId ||
    selectedProfile.provider ||
    response.provider ||
    '-';
  const activeModel = selectedProfile.model
    ? modelBasename(selectedProfile.model)
    : '-';
  const idleResponseTime = assistant.busy ? 'running' : '-';
  const responseTime = response.dtS
    ? `${formatNumber(response.dtS)}s`
    : idleResponseTime;
  return (
    <div className="settings-strip">
      <StatLine label="Provider" value={provider} />
      <StatLine label="Active Model" value={activeModel} />
      <StatLine label="Response Time" value={responseTime} accent="green" />
      <StatLine
        label="Last Action"
        value={assistant.lastRequest?.sourceLabel || '-'}
        accent="blue"
      />
    </div>
  );
};
