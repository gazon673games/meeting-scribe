const preloadDeps = global.__MEETING_SCRIBE_PRELOAD_DEPS__ || {};
const { contextBridge, ipcRenderer } = preloadDeps.electron || require("electron");
const { createMeetingScribeApi } = preloadDeps.preloadApi || require("./preload-api.cjs");

const meetingScribeApi = createMeetingScribeApi(ipcRenderer);

contextBridge.exposeInMainWorld("meetingScribe", meetingScribeApi);

module.exports = { meetingScribeApi };
