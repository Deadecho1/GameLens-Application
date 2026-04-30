const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const WebSocket = require("ws");

let qtProcess = null;

function spawnQtBackend() {
  const appRoot = path.join(__dirname, "..", "..");
  const frontendRoot = path.join(__dirname, "..");
  qtProcess = spawn(
    "/home/deadecho/.local/bin/uv",
    ["run", "python", "-m", "gui.main", "--headless"],
    {
      cwd: frontendRoot,
      env: { ...process.env, PYTHONPATH: appRoot },
      stdio: ["ignore", "pipe", "pipe"],
      detached: false,
    }
  );
  qtProcess.stdout.on("data", (d) => process.stdout.write("[qt] " + d));
  qtProcess.stderr.on("data", (d) => process.stderr.write("[qt] " + d));
  qtProcess.on("error", (e) => console.error("[qt spawn error]", e.message));
}

app.on("will-quit", () => {
  if (qtProcess) {
    qtProcess.kill();
    qtProcess = null;
  }
});

const IPC_URL = "ws://127.0.0.1:8765";
let ws = null;
let nextId = 1;
const pending = new Map();

function connectQtIpc() {
  ws = new WebSocket(IPC_URL);

  ws.on("open", () => {});

  ws.on("message", (buf) => {
    let msg;
    try {
      msg = JSON.parse(buf.toString());
    } catch {
      return;
    }
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    if (msg.ok) p.resolve(msg.result);
    else p.reject(new Error(msg.error || "IPC error"));
  });

  ws.on("close", () => {
    for (const [, p] of pending) {
      p.reject(new Error("Qt IPC disconnected"));
    }
    pending.clear();
    setTimeout(connectQtIpc, 1200);
  });

  ws.on("error", () => {});
}

function ipcRequest(method, params = {}) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return Promise.reject(
      new Error("Qt IPC not connected. Backend still starting up."),
    );
  }
  const id = String(nextId++);
  const payload = { id, method, params };
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify(payload));
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const devUrl = "http://localhost:5173";
  const distIndex = path.join(__dirname, "..", "dist", "index.html");

  if (!app.isPackaged) {
    win.loadURL(devUrl);
    return;
  }

  if (fs.existsSync(distIndex)) {
    win.loadFile(distIndex);
  } else {
    win.loadURL(devUrl);
  }
}

app.whenReady().then(() => {
  spawnQtBackend();
  connectQtIpc();
  createWindow();
});

ipcMain.handle("gamelens:ipc", (_evt, method, params) =>
  ipcRequest(method, params || {}),
);

ipcMain.handle("gamelens:choose-folder", async () => {
  const res = await dialog.showOpenDialog({
    properties: ["openDirectory"],
  });
  if (res.canceled || !res.filePaths.length) return null;
  return res.filePaths[0];
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
