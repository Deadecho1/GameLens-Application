from __future__ import annotations

import json
from pathlib import Path

from app_core.local_storage import LOCAL_USER_ID, open_local_db
from app_core.logging import get_logger

logger = get_logger(__name__)


class LocalRunUploader:
    """Writes processed run JSONs directly to the local SQLite database."""

    def __init__(
        self,
        db_path: Path | None = None,
        user_id: int = LOCAL_USER_ID,
        cleanup_run_json: bool = False,
    ) -> None:
        self._db_path = db_path
        self._user_id = user_id
        self._cleanup_run_json = cleanup_run_json

    # ------------------------------------------------------------------
    # Entity helpers (create-or-get, idempotent)
    # ------------------------------------------------------------------

    def _ensure_game(self, conn, name: str) -> int:
        conn.execute(
            "INSERT OR IGNORE INTO dash_games (user_id, name) VALUES (?, ?)",
            (self._user_id, name),
        )
        row = conn.execute(
            "SELECT id FROM dash_games WHERE user_id = ? AND name = ?",
            (self._user_id, name),
        ).fetchone()
        return row[0]

    def _ensure_version(self, conn, game_id: int, name: str) -> int:
        conn.execute(
            "INSERT OR IGNORE INTO dash_game_versions (game_id, name) VALUES (?, ?)",
            (game_id, name),
        )
        row = conn.execute(
            "SELECT id FROM dash_game_versions WHERE game_id = ? AND name = ?",
            (game_id, name),
        ).fetchone()
        return row[0]

    def _ensure_item(self, conn, game_id: int, name: str, version_id: int) -> int:
        conn.execute(
            "INSERT OR IGNORE INTO dash_items (game_id, name, first_seen_version_id) VALUES (?, ?, ?)",
            (game_id, name, version_id),
        )
        row = conn.execute(
            "SELECT id FROM dash_items WHERE game_id = ? AND name = ?",
            (game_id, name),
        ).fetchone()
        return row[0]

    def _ensure_boss(self, conn, game_id: int, name: str, version_id: int) -> int:
        conn.execute(
            "INSERT OR IGNORE INTO dash_bosses (game_id, name, first_seen_version_id) VALUES (?, ?, ?)",
            (game_id, name, version_id),
        )
        row = conn.execute(
            "SELECT id FROM dash_bosses WHERE game_id = ? AND name = ?",
            (game_id, name),
        ).fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Run JSON → internal payload conversion (shared with RunUploader)
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
        boss_encounters = [
            {
                "boss_name": (bf.get("boss_names") or ["Unknown"])[0],
                "start_time": bf.get("start_time"),
                "end_time": bf.get("end_time"),
                "duration_seconds": bf.get("duration_seconds"),
                "player_died": bf.get("player_died", False),
            }
            for bf in run_data.get("boss_fights", [])
        ]
        return {
            "video_filename": video_filename,
            "start_time": run_data.get("start_time"),
            "end_time": run_data.get("end_time"),
            "duration_seconds": run_data.get("duration_seconds"),
            "outcome": "death",  # TODO: detect win condition when pipeline can identify it
            "item_pickups": item_pickups,
            "boss_encounters": boss_encounters,
        }

    def _insert_run(self, conn, game_id: int, version_id: int, payload: dict) -> bool:
        """Insert run and its pickups/encounters. Returns True if newly inserted, False if already existed."""
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO dash_runs
                (version_id, video_filename, start_time, end_time, duration_seconds, outcome)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                payload["video_filename"],
                payload["start_time"],
                payload["end_time"],
                payload["duration_seconds"],
                payload["outcome"],
            ),
        )
        if cur.rowcount == 0:
            return (
                False  # already existed — skip pickups/encounters to avoid duplicates
            )

        run_id = cur.lastrowid

        for pickup in payload.get("item_pickups", []):
            item_name = (pickup.get("item_name") or "").strip()
            if not item_name:
                continue
            item_id = self._ensure_item(conn, game_id, item_name, version_id)
            conn.execute(
                """
                INSERT INTO dash_item_pickups (run_id, item_id, picked_at_seconds, options)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    item_id,
                    pickup.get("picked_at_seconds"),
                    json.dumps(pickup.get("options", [])),
                ),
            )

        for encounter in payload.get("boss_encounters", []):
            boss_name = (encounter.get("boss_name") or "").strip()
            if not boss_name:
                continue
            boss_id = self._ensure_boss(conn, game_id, boss_name, version_id)
            player_died_raw = encounter.get("player_died")
            player_died = (
                1 if player_died_raw else (0 if player_died_raw is not None else None)
            )
            conn.execute(
                """
                INSERT INTO dash_boss_encounters
                    (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    boss_id,
                    encounter.get("start_time"),
                    encounter.get("end_time"),
                    encounter.get("duration_seconds"),
                    player_died,
                ),
            )
        return True

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def upload_from_dir(
        self, run_json_dir: Path, game_name: str, version_name: str
    ) -> None:
        run_files = sorted(run_json_dir.glob("*_run_*.json"))
        if not run_files:
            logger.warning("No run JSON files found in: %s", run_json_dir)
            return

        logger.info(
            "Saving %d run(s) to local DB: game=%r version=%r",
            len(run_files),
            game_name,
            version_name,
        )

        conn = open_local_db(self._db_path)
        try:
            game_id = self._ensure_game(conn, game_name)
            version_id = self._ensure_version(conn, game_id, version_name)

            saved = skipped = 0
            uploaded_files: list[Path] = []
            for path in run_files:
                try:
                    run_data = json.loads(path.read_text(encoding="utf-8"))
                    inserted = self._insert_run(
                        conn,
                        game_id,
                        version_id,
                        self._run_json_to_payload(run_data, path.name),
                    )
                    if inserted:
                        saved += 1
                    else:
                        skipped += 1
                    uploaded_files.append(path)
                except Exception as e:
                    logger.error("  skipping %s: %s", path.name, e)

            conn.commit()

            if self._cleanup_run_json:
                for path in uploaded_files:
                    try:
                        path.unlink()
                    except Exception as e:
                        logger.warning("  could not delete %s: %s", path.name, e)
                logger.info(
                    "Local save complete: %d new, %d already existed. JSONs cleaned up.",
                    saved,
                    skipped,
                )
            else:
                logger.info(
                    "Local save complete: %d new, %d already existed. JSON files retained.",
                    saved,
                    skipped,
                )
        finally:
            conn.close()
