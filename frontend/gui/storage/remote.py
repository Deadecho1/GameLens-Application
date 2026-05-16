from __future__ import annotations

import requests

from .base import StorageBackend


class RemoteCollectorBackend(StorageBackend):
    """Reads analytics from the GameLens Collector over HTTP. Used when logged in."""

    def __init__(self, base_url: str, user_id: int, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self._session = requests.Session()
        self._session.headers.update({"X-User-ID": str(user_id)})
        self._timeout = timeout

    def _get(self, path: str, params: dict) -> object:
        resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=self._timeout)
        if resp.status_code == 404:
            return None  # caller checks and returns empty
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        resp = self._session.post(f"{self.base_url}{path}", json=body, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def get_items(self, user_id: int, game_name: str, version_name: str | None) -> list[dict]:
        params = {"game_name": game_name, "user_id": str(user_id)}
        if version_name:
            params["version_name"] = version_name
        return self._get("/api/v1/dashboard/items", params) or []

    def get_bosses(self, user_id: int, game_name: str, version_name: str | None) -> list[dict]:
        params = {"game_name": game_name, "user_id": str(user_id)}
        if version_name:
            params["version_name"] = version_name
        return self._get("/api/v1/dashboard/bosses", params) or []

    def get_runs(self, user_id: int, game_name: str, version_name: str | None) -> list[dict]:
        params = {"game_name": game_name, "user_id": str(user_id)}
        if version_name:
            params["version_name"] = version_name
        return self._get("/api/v1/dashboard/runs", params) or []

    def get_stats(self, user_id: int, game_name: str, version_name: str | None) -> dict:
        params = {"game_name": game_name, "user_id": str(user_id)}
        if version_name:
            params["version_name"] = version_name
        return self._get("/api/v1/dashboard/stats", params) or {
            "totalRuns": 0, "avgDuration": None, "longestRun": None,
            "bossKillPercent": 0.0, "mostPopularItem": None,
        }

    # ------------------------------------------------------------------
    # Write helpers used by SyncWorker
    # ------------------------------------------------------------------

    def list_games(self, user_id: int) -> list[dict]:
        return self._get("/api/v1/games", {"user_id": str(user_id)}) or []

    def list_versions(self, game_id: int) -> list[dict]:
        return self._get(f"/api/v1/games/{game_id}/versions", {}) or []

    def export_runs(self, game_id: int, version_id: int) -> list[dict]:
        return self._get(f"/api/v1/games/{game_id}/versions/{version_id}/runs/export", {}) or []

    def ensure_game(self, name: str) -> int:
        return self._post("/api/v1/games", {"name": name})["game_id"]

    def ensure_version(self, game_id: int, name: str) -> int:
        return self._post(f"/api/v1/games/{game_id}/versions", {"name": name})["version_id"]

    def upload_runs_batch(self, game_id: int, version_id: int, runs: list[dict]) -> int:
        result = self._post(
            f"/api/v1/games/{game_id}/versions/{version_id}/runs/batch",
            {"runs": runs},
        )
        return result.get("inserted", 0)
