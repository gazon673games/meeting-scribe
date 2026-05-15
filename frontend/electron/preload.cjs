const { contextBridge, ipcRenderer } = require("electron");
const { createMeetingScribeApi } = require("./preload-api.cjs");

contextBridge.exposeInMainWorld("meetingScribe", createMeetingScribeApi(ipcRenderer));
