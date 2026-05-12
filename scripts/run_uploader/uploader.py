from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import requests

from app_core.logging import get_logger

logger = get_logger(__name__)


class RunUploader:
    """
    Reads completed run JSONs from a directory and POSTs them in a single
    batch to the GameLens Collector service.

    Run JSON shape expected (output of boss_processor / run_exporter):
      {
        "start_time": float,
        "end_time": float,
        "duration_seconds": float,
        "choices": [
          { "options": [...], "selected_option": "...", "picked_at_seconds": float }
        ],
        "boss_fights": [
          { "boss_names": [...], "start_time": float, "end_time": float,
            "duration_seconds": float, "player_died": bool }
        ]
      }
    """

    def __init__(self, collector_url: str, user_id: str, timeout: float = 30.0) -> None:
        self.base_url = collector_url.rstrip("/")
        self.user_id = user_id
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"X-User-ID": user_id})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        resp = self._session.post(url, json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Game / version registration (idempotent — create-or-get)
    # ------------------------------------------------------------------

    def ensure_game(self, game_name: str) -> str:
        """Returns the game_id for game_name, creating it if needed."""
        result = self._post("/api/v1/games", {"name": game_name})
        return result["game_id"]

    def ensure_version(self, game_id: str, version_name: str) -> str:
        """Returns the version_id for version_name under game_id, creating if needed."""
        result = self._post(f"/api/v1/games/{game_id}/versions", {"name": version_name})
        return result["version_id"]

    # ------------------------------------------------------------------
    # Run JSON → batch payload conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _run_json_to_payload(run_data: dict, video_filename: str) -> dict:
        item_pickups = [
            {
                "item_name": c.get("selected_option") or "",
                "picked_at_seconds": c.get("picked_at_seconds"),
                "options": c.get("options", []),
            }
            for c in run_data.get("choices", [])
            if c.get("selected_option")
        ]

        boss_fights = run_data.get("boss_fights", [])
        # Outcome: death if any fight ended with player_died=True on the last fight.
        # Win detection is not yet implemented — always "death".
        # TODO: detect win condition when pipeline can identify it.
        outcome = "death"

        boss_encounters = [
            {
                "boss_name": (bf.get("boss_names") or ["Unknown"])[0],
                "start_time": bf.get("start_time"),
                "end_time": bf.get("end_time"),
                "duration_seconds": bf.get("duration_seconds"),
                "player_died": bf.get("player_died", False),
            }
            for bf in boss_fights
        ]

        return {
            "video_filename": video_filename,
            "start_time": run_data.get("start_time"),
            "end_time": run_data.get("end_time"),
            "duration_seconds": run_data.get("duration_seconds"),
            "outcome": outcome,
            "item_pickups": item_pickups,
            "boss_encounters": boss_encounters,
        }

    # ------------------------------------------------------------------
    # Public upload entry point
    # ------------------------------------------------------------------

    def upload_from_dir(
        self,
        run_json_dir: Path,
        game_name: str,
        version_name: str,
    ) -> None:
        run_files = sorted(run_json_dir.glob("*_run_*.json"))
        if not run_files:
            logger.warning("No run JSON files found in: %s", run_json_dir)
            return

        logger.info("Uploading %d run(s) for game=%r version=%r", len(run_files), game_name, version_name)

        game_id = self.ensure_game(game_name)
        logger.info("  game_id=%s", game_id)

        version_id = self.ensure_version(game_id, version_name)
        logger.info("  version_id=%s", version_id)

        runs_payload = []
        for path in run_files:
            try:
                run_data = json.loads(path.read_text(encoding="utf-8"))
                runs_payload.append(self._run_json_to_payload(run_data, path.name))
            except Exception as e:
                logger.error("  skipping %s: %s", path.name, e)

        if not runs_payload:
            logger.warning("No valid run payloads to upload.")
            return

        self._post(
            f"/api/v1/games/{game_id}/versions/{version_id}/runs/batch",
            {"runs": runs_payload},
        )

        logger.info("Batch upload complete: %d run(s) submitted.", len(runs_payload))
