"use strict";

const { contextBridge, ipcRenderer } = require("electron");

function subscribe(channel, callback) {
  const handler = (_event, payload) => callback(payload);
  ipcRenderer.on(channel, handler);
  return () => ipcRenderer.removeListener(channel, handler);
}

contextBridge.exposeInMainWorld("danglish", {
  start: (settings) => ipcRenderer.invoke("backend:start", settings),
  stop: () => ipcRenderer.invoke("backend:stop"),
  state: () => ipcRenderer.invoke("backend:state"),
  listDevices: () => ipcRenderer.invoke("audio:list"),
  listContexts: () => ipcRenderer.invoke("contexts:list"),
  appInfo: () => ipcRenderer.invoke("app:info"),
  openLegal: () => ipcRenderer.invoke("legal:open"),
  onEvent: (callback) => subscribe("backend:event", callback),
  onLog: (callback) => subscribe("backend:log", callback),
  onStatus: (callback) => subscribe("backend:status", callback),
  onInfo: (callback) => subscribe("backend:info", callback),
});
