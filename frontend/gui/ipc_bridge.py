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

        if method == "processing:set_pipeline_path":
            self._window.set_pipeline_path_from_ipc(str(params["pipeline_path"]))
            return self._window.get_frontend_state_from_ipc()

        if method == "processing:stage_files":
            raw_files = params.get("file_names") or []
            if not isinstance(raw_files, list):
                raise ValueError("file_names must be an array")
            raw_file_paths = params.get("file_paths") or []
            if not isinstance(raw_file_paths, list):
                raise ValueError("file_paths must be an array")
            raw_path = params.get("pipeline_path")
            pipeline_path = str(raw_path) if raw_path is not None else None
            self._window.stage_files_from_ipc(
                [str(name) for name in raw_files],
                pipeline_path=pipeline_path,
                file_paths=[str(p) for p in raw_file_paths],
            )
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

        if method == "setup:save_settings":
            self._window.save_settings_from_ipc(str(params.get("openAiKey") or ""))
            return self._window.get_frontend_state_from_ipc()

        if method == "processing:set_option":
            self._window.set_processing_option_from_ipc(str(params["option"]))
            return self._window.get_frontend_state_from_ipc()

        if method == "processing:set_model":
            self._window.set_model_from_ipc(str(params["model"]))
            return self._window.get_frontend_state_from_ipc()

        if method == "tuning:start":
            raw_videos = params.get("videos")
            if not isinstance(raw_videos, list):
                raise ValueError("videos must be an array")
            raw_model_ids = params.get("enabledModelIds")
            if not isinstance(raw_model_ids, list):
                raise ValueError("enabledModelIds must be an array")
            model_name = str(params.get("modelName") or "").strip()
            if not model_name:
                raise ValueError("modelName is required")
            self._window.start_tuning_from_ipc(raw_videos, raw_model_ids, model_name)
            return self._window.get_frontend_state_from_ipc()

        if method == "tuning:stop":
            self._window.stop_tuning_from_ipc()
            return self._window.get_frontend_state_from_ipc()

        if method == "tuning:clear_logs":
            self._window.clear_tuning_logs_from_ipc()
            return self._window.get_frontend_state_from_ipc()

        if method == "auth:login":
            email = str(params.get("email") or "").strip()
            if not email:
                raise ValueError("email is required")
            auth = self._window.login_from_ipc(email)
            state = self._window.get_frontend_state_from_ipc()
            state["auth"] = auth
            return state

        if method == "auth:logout":
            self._window.logout_from_ipc()
            return self._window.get_frontend_state_from_ipc()

        if method == "auth:sync":
            self._window.sync_now_from_ipc()
            return self._window.get_frontend_state_from_ipc()

        if method == "auth:state":
            state = self._window.get_frontend_state_from_ipc()
            return state["auth"]

        if method == "dashboard:items":
            game_name = str(params.get("game_name") or "").strip()
            version_name = str(params.get("version_name") or "").strip() or None
            if not game_name:
                raise ValueError("game_name is required")
            return self._window.get_dashboard_items_from_ipc(game_name, version_name)

        if method == "dashboard:bosses":
            game_name = str(params.get("game_name") or "").strip()
            version_name = str(params.get("version_name") or "").strip() or None
            if not game_name:
                raise ValueError("game_name is required")
            return self._window.get_dashboard_bosses_from_ipc(game_name, version_name)

        if method == "dashboard:runs":
            game_name = str(params.get("game_name") or "").strip()
            version_name = str(params.get("version_name") or "").strip() or None
            if not game_name:
                raise ValueError("game_name is required")
            return self._window.get_dashboard_runs_from_ipc(game_name, version_name)

        if method == "dashboard:stats":
            game_name = str(params.get("game_name") or "").strip()
            version_name = str(params.get("version_name") or "").strip() or None
            if not game_name:
                raise ValueError("game_name is required")
            return self._window.get_dashboard_stats_from_ipc(game_name, version_name)

        raise ValueError(f"Unknown IPC method: {method}")
