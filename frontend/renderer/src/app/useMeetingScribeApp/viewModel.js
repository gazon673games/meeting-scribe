import { FALLBACK_OPTIONS } from "../../entities/settings/model";

export function buildAppViewModel(state, settingsDraft, config) {
  const options = state?.options || FALLBACK_OPTIONS;
  const summary = state?.configSummary || {};
  const session = state?.session || {};
  const assistant = state?.assistant || {};
  const capabilities = state?.capabilities || {};
  const hardware = state?.hardware || {};
  const assistantProfiles = assistant.profiles || settingsDraft.assistantProfiles || config?.codex?.profiles || [];
  const sources = session.sources || [];
  const transcript = session.transcript || [];
  const assistantContextReady = transcript.some((line) => String(line?.text || "").trim());
  const asrMetrics = session.asrMetrics || {};
  const offlinePass = session.offlinePass || {};
  const canStart = Boolean(capabilities.sessionControl && sources.length > 0 && !session.running && session.state !== "downloading_model");
  const canStop = Boolean(capabilities.sessionControl && session.running);
  const status = session.running ? "recording" : offlinePass.running ? "processing" : "idle";

  return {
    assistant,
    assistantContextReady,
    assistantProfiles,
    asrMetrics,
    canStart,
    canStop,
    capabilities,
    hardware,
    offlinePass,
    options,
    session,
    sources,
    status,
    summary,
    transcript
  };
}
