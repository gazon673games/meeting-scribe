export const REFRESHING_EVENTS = [
  "session_started",
  "session_stopped",
  "transcript_cleared",
  "assistant_started",
  "assistant_fallback",
  "assistant_result",
  "offline_pass_started",
  "offline_pass_done",
  "offline_pass_error"
];

export const DEVICE_CACHE_KEY = "meeting-scribe.devices.v1";

export function emptyResourceUsage() {
  return { app: {}, system: {}, gpus: [] };
}
