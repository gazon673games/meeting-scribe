import {
  FALLBACK_OPTIONS,
  languageLabel,
  uniqueOptions,
} from '../../entities/settings/model';
import { SelectShell } from '../../shared/ui/forms/SelectShell';
import { PipelinePanel } from '../../shared/ui/pipeline/PipelinePanel';
import { SectionLabel } from '../../shared/ui/SectionLabel';
import { ProcessingFeatureGrid } from './ProcessingFeatureGrid';
import { ProcessingStats } from './ProcessingStats';
import { QualityProfileSelector } from './QualityProfileSelector';
import { ResourceUsage } from './ResourceUsage';

export const ProcessingColumn = ({
  asrMetrics,
  draft,
  headerProps,
  layoutControls,
  locked,
  offlinePass,
  options,
  resourceUsage,
  session,
  summary,
  onChange,
  onProfileChange,
}) => {
  const languageOptions = uniqueOptions(
    options.languages?.length ? options.languages : FALLBACK_OPTIONS.languages,
    draft.language,
  );

  return (
    <PipelinePanel
      title="Processing"
      active={session.running || offlinePass.running}
      activeTone={offlinePass.running ? 'warn' : 'muted'}
      headerControls={layoutControls}
      headerProps={headerProps}
    >
      <SectionLabel>Speed vs Quality</SectionLabel>
      <QualityProfileSelector
        disabled={locked}
        profiles={options.asrProfiles}
        selectedProfile={draft.profile}
        onProfileChange={onProfileChange}
      />

      <SectionLabel>Language</SectionLabel>
      <SelectShell>
        <select
          disabled={locked}
          value={draft.language}
          onChange={(event) => onChange({ language: event.target.value })}
        >
          {languageOptions.map((language) => (
            <option key={language} value={language}>
              {languageLabel(language)}
            </option>
          ))}
        </select>
      </SelectShell>

      <SectionLabel>Features</SectionLabel>
      <ProcessingFeatureGrid
        draft={draft}
        locked={locked}
        options={options}
        onChange={onChange}
      />

      <details
        className="panel-details processing-details"
        open={session.state === 'downloading_model' ? true : undefined}
      >
        <summary>
          Runtime & resources
          {Number(asrMetrics.segDroppedTotal || 0) +
            Number(asrMetrics.segSkippedTotal || 0) +
            Number(session.drops?.droppedOutBlocks || 0) +
            Number(session.drops?.droppedTapBlocks || 0) >
          0 ? (
            <span className="details-warning">Drops detected</span>
          ) : null}
        </summary>
        <ProcessingStats
          asrMetrics={asrMetrics}
          draft={draft}
          session={session}
          summary={summary}
        />
        <ResourceUsage usage={resourceUsage} />
      </details>
    </PipelinePanel>
  );
};
