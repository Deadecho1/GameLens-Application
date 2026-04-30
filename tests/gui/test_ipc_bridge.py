"""Unit tests for IpcBridge._dispatch and _on_message.

Qt is mocked at sys.modules level before gui.ipc_bridge is imported so that
IpcBridge inherits from plain object instead of QObject. This lets us bypass
__init__ via object.__new__ without needing a Qt application.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock Qt before importing gui.ipc_bridge
# ---------------------------------------------------------------------------
_qt_mocks = {
    "PySide6": MagicMock(),
    "PySide6.QtCore": MagicMock(QObject=object),
    "PySide6.QtGui": MagicMock(),
    "PySide6.QtWidgets": MagicMock(),
    "PySide6.QtNetwork": MagicMock(),
    "PySide6.QtWebSockets": MagicMock(),
    # Prevent the transitive main_window import from pulling in real Qt classes
    "gui.main_window": MagicMock(),
}
for _k, _v in _qt_mocks.items():
    sys.modules[_k] = _v

# Remove any cached gui.ipc_bridge so the fresh import picks up the mocked Qt
for _k in [k for k in list(sys.modules) if k == "gui.ipc_bridge"]:
    del sys.modules[_k]

import json

import pytest

from gui.ipc_bridge import IpcBridge  # noqa: E402 — must come after Qt mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bridge(state: dict | None = None) -> tuple[IpcBridge, MagicMock]:
    """Return (bridge, mock_window) without touching Qt."""
    window = MagicMock()
    window.get_frontend_state_from_ipc.return_value = (
        state or {"ui": {}, "setup": {}, "processing": {}, "dashboard": {}}
    )
    bridge = object.__new__(IpcBridge)
    bridge._window = window
    return bridge, window


def _send(bridge: IpcBridge, method: str, params: dict | None = None, req_id: str = "1") -> dict:
    """Drive _on_message and return the parsed response payload."""
    sock = MagicMock()
    raw = json.dumps({"id": req_id, "method": method, "params": params or {}})
    bridge._on_message(sock, raw)
    sock.sendTextMessage.assert_called_once()
    return json.loads(sock.sendTextMessage.call_args[0][0])


# ---------------------------------------------------------------------------
# _dispatch — routing
# ---------------------------------------------------------------------------

class TestDispatchPing:
    def test_ping_returns_pong(self):
        bridge, _ = _make_bridge()
        result = bridge._dispatch("ping", {})
        assert result == {"pong": True, "app": "GameLens"}

    def test_unknown_method_raises(self):
        bridge, _ = _make_bridge()
        with pytest.raises(ValueError, match="Unknown IPC method"):
            bridge._dispatch("no_such_method", {})


class TestDispatchStateGet:
    def test_delegates_to_window(self):
        fake_state = {"ui": {"activeMainTab": "workflow"}, "setup": {}, "processing": {}, "dashboard": {}}
        bridge, window = _make_bridge(fake_state)
        result = bridge._dispatch("state:get", {})
        window.get_frontend_state_from_ipc.assert_called_once()
        assert result == fake_state


class TestDispatchUiPatch:
    def test_calls_patch_then_returns_state(self):
        bridge, window = _make_bridge()
        bridge._dispatch("ui:patch", {"activeMainTab": "analytics"})
        window.patch_ui_from_ipc.assert_called_once_with({"activeMainTab": "analytics"})
        window.get_frontend_state_from_ipc.assert_called()


class TestDispatchSetupMethods:
    def test_select_game(self):
        bridge, window = _make_bridge()
        bridge._dispatch("setup:select_game", {"game": "Elden Ring"})
        window.select_game_from_ipc.assert_called_once_with("Elden Ring")

    def test_select_version(self):
        bridge, window = _make_bridge()
        bridge._dispatch("setup:select_version", {"version": "v1.0"})
        window.select_version_from_ipc.assert_called_once_with("v1.0")

    def test_add_game(self):
        bridge, window = _make_bridge()
        bridge._dispatch("setup:add_game", {"name": "Hades"})
        window.add_game_from_ipc.assert_called_once_with("Hades")

    def test_add_version(self):
        bridge, window = _make_bridge()
        bridge._dispatch("setup:add_version", {"game_name": "Hades", "version_name": "v2.0"})
        window.add_version_from_ipc.assert_called_once_with(game_name="Hades", version_name="v2.0")


class TestDispatchProcessingMethods:
    def test_stage_folder(self):
        bridge, window = _make_bridge()
        bridge._dispatch("processing:stage_folder", {"pipeline_path": "/videos"})
        window.stage_folder_from_ipc.assert_called_once_with("/videos")

    def test_set_pipeline_path(self):
        bridge, window = _make_bridge()
        bridge._dispatch("processing:set_pipeline_path", {"pipeline_path": "/videos"})
        window.set_pipeline_path_from_ipc.assert_called_once_with("/videos")

    def test_set_option(self):
        bridge, window = _make_bridge()
        bridge._dispatch("processing:set_option", {"option": "only event"})
        window.set_processing_option_from_ipc.assert_called_once_with("only event")

    def test_run(self):
        bridge, window = _make_bridge()
        bridge._dispatch("processing:run", {})
        window.run_pipeline_from_ipc.assert_called_once()

    def test_stop(self):
        bridge, window = _make_bridge()
        bridge._dispatch("processing:stop", {})
        window.stop_pipeline_from_ipc.assert_called_once()

    def test_clear_logs(self):
        bridge, window = _make_bridge()
        bridge._dispatch("processing:clear_logs", {})
        window.clear_processing_logs_from_ipc.assert_called_once()

    def test_stage_files_passes_lists(self):
        bridge, window = _make_bridge()
        bridge._dispatch(
            "processing:stage_files",
            {"file_names": ["a.mp4", "b.mp4"], "file_paths": ["/v/a.mp4", "/v/b.mp4"], "pipeline_path": "/v"},
        )
        window.stage_files_from_ipc.assert_called_once_with(
            ["a.mp4", "b.mp4"],
            pipeline_path="/v",
            file_paths=["/v/a.mp4", "/v/b.mp4"],
        )

    def test_stage_files_non_list_file_names_raises(self):
        bridge, _ = _make_bridge()
        with pytest.raises(ValueError, match="file_names must be an array"):
            bridge._dispatch("processing:stage_files", {"file_names": "bad"})

    def test_stage_files_non_list_file_paths_raises(self):
        bridge, _ = _make_bridge()
        with pytest.raises(ValueError, match="file_paths must be an array"):
            bridge._dispatch("processing:stage_files", {"file_names": [], "file_paths": "bad"})


# ---------------------------------------------------------------------------
# _on_message — envelope parsing
# ---------------------------------------------------------------------------

class TestOnMessage:
    def test_valid_message_returns_ok_true(self):
        bridge, _ = _make_bridge()
        resp = _send(bridge, "ping")
        assert resp["ok"] is True
        assert resp["id"] == "1"
        assert resp["result"] == {"pong": True, "app": "GameLens"}

    def test_unknown_method_returns_ok_false(self):
        bridge, _ = _make_bridge()
        resp = _send(bridge, "no_such_method")
        assert resp["ok"] is False
        assert "Unknown IPC method" in resp["error"]

    def test_malformed_json_returns_ok_false(self):
        bridge, _ = _make_bridge()
        sock = MagicMock()
        bridge._on_message(sock, "not json {{")
        resp = json.loads(sock.sendTextMessage.call_args[0][0])
        assert resp["ok"] is False

    def test_non_dict_params_returns_ok_false(self):
        bridge, _ = _make_bridge()
        sock = MagicMock()
        raw = json.dumps({"id": "42", "method": "ping", "params": ["not", "a", "dict"]})
        bridge._on_message(sock, raw)
        resp = json.loads(sock.sendTextMessage.call_args[0][0])
        assert resp["ok"] is False
        assert resp["id"] == "42"

    def test_response_id_matches_request_id(self):
        bridge, _ = _make_bridge()
        resp = _send(bridge, "ping", req_id="abc-123")
        assert resp["id"] == "abc-123"

    def test_window_error_propagates_as_ok_false(self):
        bridge, window = _make_bridge()
        window.add_game_from_ipc.side_effect = ValueError("Game name cannot be empty.")
        resp = _send(bridge, "setup:add_game", {"name": "  "})
        assert resp["ok"] is False
        assert "Game name cannot be empty" in resp["error"]
