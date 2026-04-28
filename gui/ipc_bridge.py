from __future__ import annotations

import json
import uuid
from typing import Any

from PySide6.QtCore import QObject
from PySide6.QtNetwork import QHostAddress
from PySide6.QtWebSockets import QWebSocketServer

from .main_window import MainWindow


class IpcBridge(QObject):
    def __init__(
        self,
        window: MainWindow,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        super().__init__(window)
        self._window = window
        self._host = host
        self._port = port
        self._server = QWebSocketServer(
            "GameLens IPC",
            QWebSocketServer.SslMode.NonSecureMode,
            self,
        )

    def start(self) -> bool:
        ok = self._server.listen(QHostAddress(self._host), self._port)
        if ok:
            self._server.newConnection.connect(self._on_new_connection)
        return ok

    def _on_new_connection(self) -> None:
        sock = self._server.nextPendingConnection()
        if sock is None:
            return
        sock.textMessageReceived.connect(
            lambda msg, socket=sock: self._on_message(socket, msg)
        )

    def _on_message(self, sock, raw: str) -> None:
        req_id: str | None = None
        try:
            msg = json.loads(raw)
            req_id = str(msg.get("id") or uuid.uuid4())
            method = str(msg.get("method") or "")
            params = msg.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("params must be an object")

            result = self._dispatch(method, params)
            payload = {"id": req_id, "ok": True, "result": result}
        except Exception as e:
            payload = {"id": req_id, "ok": False, "error": str(e)}

        sock.sendTextMessage(json.dumps(payload))

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "ping":
            return {"pong": True, "app": "GameLens"}

        if method == "state:get":
            return self._window.get_frontend_state_from_ipc()

        if method == "ui:patch":
            self._window.patch_ui_from_ipc(dict(params))
            return self._window.get_frontend_state_from_ipc()

        if method == "setup:select_game":
            self._window.select_game_from_ipc(str(params["game"]))
            return self._window.get_frontend_state_from_ipc()

        if method == "setup:select_version":
            self._window.select_version_from_ipc(str(params["version"]))
            return self._window.get_frontend_state_from_ipc()

        if method == "setup:add_game":
            self._window.add_game_from_ipc(str(params["name"]))
            return self._window.get_frontend_state_from_ipc()

        if method == "setup:add_version":
            self._window.add_version_from_ipc(
                game_name=str(params["game_name"]),
                version_name=str(params["version_name"]),
            )
            return self._window.get_frontend_state_from_ipc()

        if method == "processing:stage_folder":
            self._window.stage_folder_from_ipc(str(params["pipeline_path"]))
            return self._window.get_frontend_state_from_ipc()

        if method == "processing:set_option":
            self._window.set_processing_option_from_ipc(str(params["option"]))
            return self._window.get_frontend_state_from_ipc()

        if method == "processing:run":
            self._window.run_pipeline_from_ipc()
            return self._window.get_frontend_state_from_ipc()

        if method == "processing:stop":
            self._window.stop_pipeline_from_ipc()
            return self._window.get_frontend_state_from_ipc()

        if method == "processing:clear_logs":
            self._window.clear_processing_logs_from_ipc()
            return self._window.get_frontend_state_from_ipc()

        raise ValueError(f"Unknown IPC method: {method}")
