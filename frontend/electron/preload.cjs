const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('gamelens', {
  request(method, params = {}) {
    return ipcRenderer.invoke('gamelens:ipc', method, params);
  },
  chooseFolder() {
    return ipcRenderer.invoke('gamelens:choose-folder');
  },
});
