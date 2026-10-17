import { FALLBACK_OPTIONS } from "../../entities/settings/model";

export function buildAppViewModel(state, settingsDraft, config) {
  const snapshot = normalizeStateSnapshot(state);
  const assistantProfiles = resolveAssistantProfiles(snapshot.assistant, settingsDraft, config);
  const sources = snapshot.session.sources || [];
  const transcript = snapshot.session.transcript || [];
  const assistantContextReady = hasTranscriptText(transcript);
  const asrMetrics = snapshot.session.asrMetrics || {};
  const offlinePass = snapshot.session.offlinePass || {};
  const canStart = canStartSession(snapshot.capabilities, snapshot.session, sources);
  const canStop = canStopSession(snapshot.capabilities, snapshot.session);
  const status = sessionStatus(snapshot.session, offlinePass);

  return {
    assistant: snapshot.assistant,
    assistantContextReady,
    assistantProfiles,
    asrMetrics,
    canStart,
    canStop,
    capabilities: snapshot.capabilities,
    hardware: snapshot.hardware,
    offlinePass,
    options: snapshot.options,
    session: snapshot.session,
    sources,
    startUnavailableReason: startUnavailableReason(snapshot.capabilities, snapshot.session, sources),
    status,
    summary: snapshot.summary,
    transcript
  };
}

function normalizeStateSnapshot(state) {
  return {
    options: state?.options || FALLBACK_OPTIONS,
    summary: state?.configSummary || {},
    session: state?.session || {},
    assistant: state?.assistant || {},
    capabilities: state?.capabilities || {},
    hardware: state?.hardware || {},
  };
}

function resolveAssistantProfiles(assistant, settingsDraft, config) {
  return assistant.profiles || settingsDraft.assistantProfiles || config?.codex?.profiles || [];
}

function hasTranscriptText(transcript) {
  return transcript.some((line) => String(line?.text || "").trim());
}

function canStartSession(capabilities, session, sources) {
  return Boolean(capabilities.sessionControl && sources.some((source) => Boolean(source?.enabled)) && !session.running && session.state !== "downloading_model");
}

function startUnavailableReason(capabilities, session, sources) {
  if (!capabilities.sessionControl) return "Session control is unavailable.";
  if (session.state === "downloading_model") return "Wait for the transcription model to finish downloading.";
  if (!sources.length) return "Add an audio source to start.";
  if (!sources.some((source) => Boolean(source?.enabled))) return "Enable an audio source to start.";
  return "";
}

function canStopSession(capabilities, session) {
  return Boolean(capabilities.sessionControl && session.running);
}

function sessionStatus(session, offlinePass) {
  if (session.running) {
    return "recording";
  }
  if (offlinePass.running) {
    return "processing";
  }
  return "idle";
}
