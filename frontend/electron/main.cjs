const { app, BrowserWindow, ipcMain, dialog, protocol, Menu } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const { Readable } = require("stream");
const WebSocket = require("ws");

// Register custom scheme before app is ready — required by Electron
protocol.registerSchemesAsPrivileged([
  {
    scheme: "gamelens-local",
    privileges: { secure: true, supportFetchAPI: true, stream: true },
  },
]);

let qtProcess = null;
let mainWindow = null;

function resolveUv() {
  // Prefer uv on PATH; fall back to common install locations
  const { execSync } = require("child_process");
  try {
    return execSync("which uv", { encoding: "utf8" }).trim();
  } catch {
    for (const p of [
      "/home/" + (process.env.USER || "") + "/.local/bin/uv",
      "/usr/local/bin/uv",
      "/usr/bin/uv",
    ]) {
      if (fs.existsSync(p)) return p;
    }
    return "uv"; // last resort — let spawn fail with a clear error
  }
}

function spawnQtBackend() {
  const appRoot = path.join(__dirname, "..", "..");
  const frontendRoot = path.join(__dirname, "..");
  const uvBin = resolveUv();

  function trySpawn(pythonCmd) {
    const proc = spawn(
      uvBin,
      ["run", pythonCmd, "-m", "gui.main", "--headless"],
      {
        cwd: frontendRoot,
        env: { ...process.env, PYTHONPATH: appRoot },
        stdio: ["ignore", "pipe", "pipe"],
        detached: false,
      }
    );
    proc.stdout.on("data", (d) => process.stdout.write("[qt] " + d));
    proc.stderr.on("data", (d) => process.stderr.write("[qt] " + d));
    proc.on("error", (e) => console.error("[qt spawn error]", e.message));
    proc.on("close", (code) => {
      if (code === null) return; // killed intentionally
      if (pythonCmd === "python" && code !== 0) {
        console.warn("[qt] python failed, retrying with python3");
        qtProcess = trySpawn("python3");
      }
    });
    return proc;
  }

  qtProcess = trySpawn("python");
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
  Menu.setApplicationMenu(null);

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    frame: false,
    backgroundColor: "#0f172a",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const notifyMaximize = () => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    mainWindow.webContents.send(
      "window:maximized-changed",
      mainWindow.isMaximized(),
    );
  };
  mainWindow.on("maximize", notifyMaximize);
  mainWindow.on("unmaximize", notifyMaximize);

  const devUrl = "http://localhost:5173";
  const distIndex = path.join(__dirname, "..", "dist", "index.html");

  if (!app.isPackaged) {
    mainWindow.loadURL(devUrl);
    return;
  }

  if (fs.existsSync(distIndex)) {
    mainWindow.loadFile(distIndex);
  } else {
    mainWindow.loadURL(devUrl);
  }
}

app.whenReady().then(() => {
  // Serve local video files via gamelens-local:// protocol with Range support for seeking
  protocol.handle("gamelens-local", (request) => {
    const url = new URL(request.url);
    const pathname = decodeURIComponent(url.pathname);
    const filePath =
      process.platform === "win32" && /^\/[A-Za-z]:/.test(pathname)
        ? pathname.slice(1)
        : pathname;

    if (!fs.existsSync(filePath)) {
      return new Response("Not found", { status: 404 });
    }

    const fileSize = fs.statSync(filePath).size;
    const rangeHeader = request.headers.get("Range");

    if (rangeHeader) {
      const match = rangeHeader.match(/bytes=(\d+)-(\d*)/);
      if (match) {
        const start = parseInt(match[1], 10);
        const end = match[2] ? parseInt(match[2], 10) : fileSize - 1;
        const chunkSize = end - start + 1;
        return new Response(Readable.toWeb(fs.createReadStream(filePath, { start, end })), {
          status: 206,
          headers: {
            "Content-Range": `bytes ${start}-${end}/${fileSize}`,
            "Accept-Ranges": "bytes",
            "Content-Length": String(chunkSize),
            "Content-Type": "video/mp4",
          },
        });
      }
    }

    return new Response(Readable.toWeb(fs.createReadStream(filePath)), {
      status: 200,
      headers: {
        "Content-Length": String(fileSize),
        "Content-Type": "video/mp4",
        "Accept-Ranges": "bytes",
      },
    });
  });

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

ipcMain.handle("gamelens:choose-file", async (_evt, opts = {}) => {
  const res = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: opts.filters || [{ name: "Video", extensions: ["mp4"] }],
  });
  if (res.canceled || !res.filePaths.length) return null;
  return res.filePaths[0];
});

ipcMain.on("window:minimize", () => {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.minimize();
});

ipcMain.on("window:maximize", () => {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  if (mainWindow.isMaximized()) mainWindow.unmaximize();
  else mainWindow.maximize();
});

ipcMain.on("window:close", () => {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.close();
});

ipcMain.handle("window:isMaximized", () => {
  if (!mainWindow || mainWindow.isDestroyed()) return false;
  return mainWindow.isMaximized();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
