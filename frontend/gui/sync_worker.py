from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app_core.local_storage import LOCAL_USER_ID, open_local_db

from .storage.remote import RemoteCollectorBackend


class SyncWorker(QThread):
    """Pushes all unsynced local SQLite data to the remote Collector after login."""

    sync_progress = Signal(str)
    sync_finished = Signal(bool, str)

    def __init__(self, remote: RemoteCollectorBackend, db_path: Path | None = None) -> None:
        super().__init__()
        self._remote = remote
        self._db_path = db_path

    def run(self) -> None:
        try:
            self._run_sync()
            self.sync_finished.emit(True, "Sync complete.")
        except Exception as e:
            self.sync_finished.emit(False, f"Sync failed: {e}")

    def _run_sync(self) -> None:
        conn = open_local_db(self._db_path)
        try:
            games = conn.execute(
                "SELECT id, name FROM dash_games WHERE user_id = ?", (LOCAL_USER_ID,)
            ).fetchall()

            for local_game_id, game_name in games:
                self.sync_progress.emit(f"Syncing game: {game_name}")
                remote_game_id = self._remote.ensure_game(game_name)

                versions = conn.execute(
                    "SELECT id, name FROM dash_game_versions WHERE game_id = ?",
                    (local_game_id,),
                ).fetchall()

                for local_version_id, version_name in versions:
                    self.sync_progress.emit(f"  Syncing version: {version_name}")
                    remote_version_id = self._remote.ensure_version(remote_game_id, version_name)

                    runs = conn.execute(
                        """
                        SELECT id, video_filename, start_time, end_time, duration_seconds, outcome
                        FROM dash_runs
                        WHERE version_id = ? AND synced_at IS NULL
                        """,
                        (local_version_id,),
                    ).fetchall()

                    if not runs:
                        self.sync_progress.emit(f"    No unsynced runs.")
                        continue

                    run_payloads = []
                    for run in runs:
                        run_id, video_filename, start_time, end_time, duration_seconds, outcome = run

                        pickups = conn.execute(
                            """
                            SELECT i.name, ip.picked_at_seconds, ip.options
                            FROM dash_item_pickups ip
                            JOIN dash_items i ON i.id = ip.item_id
                            WHERE ip.run_id = ?
                            """,
                            (run_id,),
                        ).fetchall()

                        encounters = conn.execute(
                            """
                            SELECT b.name, be.start_time, be.end_time, be.duration_seconds, be.player_died
                            FROM dash_boss_encounters be
                            JOIN dash_bosses b ON b.id = be.boss_id
                            WHERE be.run_id = ?
                            """,
                            (run_id,),
                        ).fetchall()

                        run_payloads.append({
                            "video_filename": video_filename,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration_seconds": duration_seconds,
                            "outcome": outcome,
                            "item_pickups": [
                                {
                                    "item_name": name,
                                    "picked_at_seconds": picked_at,
                                    "options": json.loads(options) if options else [],
                                }
                                for name, picked_at, options in pickups
                            ],
                            "boss_encounters": [
                                {
                                    "boss_name": name,
                                    "start_time": bs,
                                    "end_time": be,
                                    "duration_seconds": dur,
                                    "player_died": bool(pd) if pd is not None else False,
                                }
                                for name, bs, be, dur, pd in encounters
                            ],
                        })

                    self.sync_progress.emit(f"    Uploading {len(run_payloads)} run(s)...")
                    self._remote.upload_runs_batch(remote_game_id, remote_version_id, run_payloads)

                    run_ids = [r[0] for r in runs]
                    conn.execute(
                        f"UPDATE dash_runs SET synced_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
                        f"WHERE id IN ({','.join('?' * len(run_ids))})",
                        run_ids,
                    )
                    conn.commit()
                    self.sync_progress.emit(f"    Done.")
        finally:
            conn.close()
