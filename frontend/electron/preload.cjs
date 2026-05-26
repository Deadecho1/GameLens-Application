const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('gamelens', {
  request(method, params = {}) {
    return ipcRenderer.invoke('gamelens:ipc', method, params);
  },
  chooseFolder() {
    return ipcRenderer.invoke('gamelens:choose-folder');
  },
  chooseFile(opts = {}) {
    return ipcRenderer.invoke('gamelens:choose-file', opts);
  },
  windowControls: {
    minimize() {
      ipcRenderer.send('window:minimize');
    },
    maximize() {
      ipcRenderer.send('window:maximize');
    },
    close() {
      ipcRenderer.send('window:close');
    },
    isMaximized() {
      return ipcRenderer.invoke('window:isMaximized');
    },
    onMaximizeChange(callback) {
      if (typeof callback !== 'function') return () => {};
      const handler = (_event, isMaximized) => callback(Boolean(isMaximized));
      ipcRenderer.on('window:maximized-changed', handler);
      return () => ipcRenderer.removeListener('window:maximized-changed', handler);
    },
  },
});
