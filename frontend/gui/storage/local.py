from __future__ import annotations

from pathlib import Path

from app_core.local_storage import LOCAL_USER_ID, open_local_db

from .base import StorageBackend


def _fmt_mm_ss(seconds) -> str | None:
    if seconds is None:
        return None
    total = max(0, int(round(float(seconds))))
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"


def _fmt_hh_mm_ss(seconds) -> str | None:
    if seconds is None:
        return None
    total = max(0, int(round(float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class LocalSQLiteBackend(StorageBackend):
    """Read-only analytics from local SQLite. Writes come from LocalRunUploader subprocess."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def _connect(self):
        return open_local_db(self._db_path)

    def _get_game_id(self, conn, user_id: int, game_name: str) -> int | None:
        row = conn.execute(
            "SELECT id FROM dash_games WHERE user_id = ? AND name = ?",
            (user_id, game_name),
        ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------

    def get_items(self, user_id: int, game_name: str, version_name: str | None) -> list[dict]:
        conn = self._connect()
        try:
            game_id = self._get_game_id(conn, user_id, game_name)
            if game_id is None:
                return []

            vf = "AND gv.name = ?" if version_name else ""
            vp = (version_name,) if version_name else ()

            rows = conn.execute(
                f"""
                WITH filtered_runs AS (
                    SELECT r.id
                    FROM dash_runs r
                    JOIN dash_game_versions gv ON gv.id = r.version_id
                    WHERE gv.game_id = ? {vf}
                ),
                run_count AS (
                    SELECT CAST(COUNT(*) AS REAL) AS total_runs FROM filtered_runs
                )
                SELECT
                    i.id,
                    i.name,
                    CASE
                        WHEN rc.total_runs = 0 THEN NULL
                        WHEN COUNT(DISTINCT fr.id) = 0 THEN NULL
                        ELSE COUNT(DISTINCT fr.id) * 100.0 / rc.total_runs
                    END AS popularity,
                    AVG(CASE WHEN fr.id IS NOT NULL THEN ip.picked_at_seconds END) / 60.0
                        AS avg_possession_minutes
                FROM dash_items i
                CROSS JOIN run_count rc
                LEFT JOIN dash_item_pickups ip ON ip.item_id = i.id
                LEFT JOIN filtered_runs fr ON fr.id = ip.run_id
                WHERE i.game_id = ?
                GROUP BY i.id, i.name, rc.total_runs
                HAVING COUNT(DISTINCT fr.id) > 0
                ORDER BY i.name
                """,
                (game_id,) + vp + (game_id,),
            ).fetchall()

            result = []
            for row in rows:
                item_id, name, popularity, avg_pm = row
                impact = (
                    "High" if popularity is not None and popularity >= 66.0
                    else "Medium" if popularity is not None and popularity >= 33.0
                    else "Low" if popularity is not None
                    else None
                )
                result.append({
                    "id": item_id,
                    "name": name,
                    "popularity": round(float(popularity), 2) if popularity is not None else None,
                    "impact": impact,
                    "category": None,
                    "logicTag": None,
                    "rarity": None,
                    "avgPossessionMinutes": round(float(avg_pm), 2) if avg_pm is not None else None,
                })
            return result
        finally:
            conn.close()

    def get_bosses(self, user_id: int, game_name: str, version_name: str | None) -> list[dict]:
        conn = self._connect()
        try:
            game_id = self._get_game_id(conn, user_id, game_name)
            if game_id is None:
                return []

            # Pass version_name twice for the two CASE WHEN guards; None → SQLite NULL → IS NULL passes
            boss_rows = conn.execute(
                """
                SELECT
                    b.id,
                    b.name,
                    AVG(CASE WHEN (? IS NULL OR gv.name = ?) THEN be.duration_seconds END)
                        AS avg_duration_seconds,
                    MAX(CASE WHEN be.player_died = 0 AND (? IS NULL OR gv.name = ?) THEN 1 ELSE 0 END)
                        AS any_defeated
                FROM dash_bosses b
                LEFT JOIN dash_boss_encounters be ON be.boss_id = b.id
                LEFT JOIN dash_runs r ON r.id = be.run_id
                LEFT JOIN dash_game_versions gv ON gv.id = r.version_id
                WHERE b.game_id = ?
                GROUP BY b.id, b.name
                ORDER BY b.name
                """,
                (version_name, version_name, version_name, version_name, game_id),
            ).fetchall()

            sample_rows = conn.execute(
                """
                SELECT b.id, be.duration_seconds
                FROM dash_bosses b
                LEFT JOIN dash_boss_encounters be ON be.boss_id = b.id
                LEFT JOIN dash_runs r ON r.id = be.run_id
                LEFT JOIN dash_game_versions gv ON gv.id = r.version_id
                WHERE b.game_id = ?
                  AND be.duration_seconds IS NOT NULL
                  AND (? IS NULL OR gv.name = ?)
                ORDER BY b.id, be.id
                """,
                (game_id, version_name, version_name),
            ).fetchall()

            samples_by_boss: dict[int, list[str]] = {}
            for boss_id, duration in sample_rows:
                samples_by_boss.setdefault(boss_id, []).append(_fmt_mm_ss(duration))

            result = []
            for boss_id, boss_name, avg_dur, any_def in boss_rows:
                result.append({
                    "id": boss_id,
                    "name": boss_name,
                    "lifespan": _fmt_mm_ss(avg_dur),
                    "status": (
                        "Defeated" if avg_dur is not None and any_def
                        else "Alive" if avg_dur is not None
                        else None
                    ),
                    "globalLifespanSamples": samples_by_boss.get(boss_id, []),
                    "gearSynergies": [],
                    "itemEffectiveness": [],
                })
            return result
        finally:
            conn.close()

    def get_runs(self, user_id: int, game_name: str, version_name: str | None) -> list[dict]:
        conn = self._connect()
        try:
            game_id = self._get_game_id(conn, user_id, game_name)
            if game_id is None:
                return []

            vf = "AND gv.name = ?" if version_name else ""
            vp = (version_name,) if version_name else ()

            run_rows = conn.execute(
                f"""
                SELECT r.id, substr(r.recorded_at, 1, 10), r.duration_seconds, r.outcome,
                       gv.name AS game_version
                FROM dash_runs r
                JOIN dash_game_versions gv ON gv.id = r.version_id
                WHERE gv.game_id = ? {vf}
                ORDER BY r.id DESC
                """,
                (game_id,) + vp,
            ).fetchall()

            if not run_rows:
                return []

            run_ids = [r[0] for r in run_rows]
            placeholders = ",".join("?" * len(run_ids))

            enc_rows = conn.execute(
                f"""
                SELECT
                    be.run_id,
                    be.boss_id,
                    be.duration_seconds,
                    be.start_time,
                    GROUP_CONCAT(DISTINCT ip.item_id) AS loadout_csv
                FROM dash_boss_encounters be
                LEFT JOIN dash_item_pickups ip
                  ON ip.run_id = be.run_id
                 AND ip.picked_at_seconds IS NOT NULL
                 AND (be.start_time IS NULL OR ip.picked_at_seconds <= be.start_time)
                WHERE be.run_id IN ({placeholders})
                GROUP BY be.id, be.run_id, be.boss_id, be.duration_seconds, be.start_time
                ORDER BY be.run_id, be.id
                """,
                run_ids,
            ).fetchall()

            encounters_by_run: dict[int, list[dict]] = {}
            for run_id, boss_id, lifespan_sec, start_time, loadout_csv in enc_rows:
                loadout = [int(x) for x in loadout_csv.split(",") if x] if loadout_csv else []
                encounters_by_run.setdefault(run_id, []).append({
                    "bossId": boss_id,
                    "lifespan": _fmt_mm_ss(lifespan_sec),
                    "loadout": loadout,
                    "startTime": start_time,
                })

            pickup_rows = conn.execute(
                f"""
                SELECT run_id, item_id, picked_at_seconds, options
                FROM dash_item_pickups
                WHERE run_id IN ({placeholders})
                  AND picked_at_seconds IS NOT NULL
                ORDER BY run_id, picked_at_seconds
                """,
                run_ids,
            ).fetchall()

            import json as _json
            pickups_by_run: dict[int, list[dict]] = {}
            for run_id, item_id, picked_at_sec, options_json in pickup_rows:
                try:
                    options = _json.loads(options_json) if options_json else []
                except Exception:
                    options = []
                pickups_by_run.setdefault(run_id, []).append({
                    "itemId": item_id,
                    "pickedAtSeconds": picked_at_sec,
                    "options": options,
                })

            return [
                {
                    "id": run_id,
                    "date": run_date,
                    "duration": _fmt_hh_mm_ss(duration_sec),
                    "outcome": outcome,
                    "gameVersion": game_version,
                    "bossEncounters": encounters_by_run.get(run_id, []),
                    "itemPickups": pickups_by_run.get(run_id, []),
                }
                for run_id, run_date, duration_sec, outcome, game_version in run_rows
            ]
        finally:
            conn.close()

    def get_stats(self, user_id: int, game_name: str, version_name: str | None) -> dict:
        conn = self._connect()
        try:
            game_id = self._get_game_id(conn, user_id, game_name)
            if game_id is None:
                return self._empty_stats()

            vf = "AND gv.name = ?" if version_name else ""
            vp = (version_name,) if version_name else ()

            total_runs, avg_dur, max_dur = conn.execute(
                f"""
                SELECT COUNT(*), AVG(r.duration_seconds), MAX(r.duration_seconds)
                FROM dash_runs r
                JOIN dash_game_versions gv ON gv.id = r.version_id
                WHERE gv.game_id = ? {vf}
                """,
                (game_id,) + vp,
            ).fetchone()

            defeated, total_enc = conn.execute(
                f"""
                SELECT
                    CAST(SUM(CASE WHEN be.player_died = 0 THEN 1 ELSE 0 END) AS REAL),
                    CAST(COUNT(*) AS REAL)
                FROM dash_boss_encounters be
                JOIN dash_runs r ON r.id = be.run_id
                JOIN dash_game_versions gv ON gv.id = r.version_id
                WHERE gv.game_id = ? {vf}
                """,
                (game_id,) + vp,
            ).fetchone()

            item_row = conn.execute(
                f"""
                WITH filtered_runs AS (
                    SELECT r.id FROM dash_runs r
                    JOIN dash_game_versions gv ON gv.id = r.version_id
                    WHERE gv.game_id = ? {vf}
                )
                SELECT i.name
                FROM dash_item_pickups ip
                JOIN dash_items i ON i.id = ip.item_id
                WHERE ip.run_id IN (SELECT id FROM filtered_runs)
                GROUP BY i.id, i.name
                ORDER BY COUNT(DISTINCT ip.run_id) DESC, i.name ASC
                LIMIT 1
                """,
                (game_id,) + vp,
            ).fetchone()

            boss_kill_pct = 0.0
            if total_enc and total_enc > 0:
                boss_kill_pct = round((defeated or 0.0) * 100.0 / total_enc, 2)

            return {
                "totalRuns": int(total_runs or 0),
                "avgDuration": _fmt_hh_mm_ss(avg_dur),
                "longestRun": _fmt_hh_mm_ss(max_dur),
                "bossKillPercent": boss_kill_pct,
                "mostPopularItem": item_row[0] if item_row else None,
            }
        finally:
            conn.close()

    def ensure_game(self, user_id: int, game_name: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO dash_games (user_id, name) VALUES (?, ?)",
                (user_id, game_name),
            )
            conn.commit()
        finally:
            conn.close()

    def ensure_version(self, user_id: int, game_name: str, version_name: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO dash_games (user_id, name) VALUES (?, ?)",
                (user_id, game_name),
            )
            game_row = conn.execute(
                "SELECT id FROM dash_games WHERE user_id = ? AND name = ?",
                (user_id, game_name),
            ).fetchone()
            if game_row:
                conn.execute(
                    "INSERT OR IGNORE INTO dash_game_versions (game_id, name) VALUES (?, ?)",
                    (game_row[0], version_name),
                )
            conn.commit()
        finally:
            conn.close()

    def list_game_names(self, user_id: int) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT name FROM dash_games WHERE user_id = ? ORDER BY name",
                (user_id,),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def list_version_names(self, user_id: int, game_name: str) -> list[str]:
        conn = self._connect()
        try:
            game_row = conn.execute(
                "SELECT id FROM dash_games WHERE user_id = ? AND name = ?",
                (user_id, game_name),
            ).fetchone()
            if not game_row:
                return []
            rows = conn.execute(
                "SELECT name FROM dash_game_versions WHERE game_id = ? ORDER BY name",
                (game_row[0],),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def check_processed_videos(
        self, user_id: int, game_name: str, version_name: str | None, video_names: list[str]
    ) -> list[str]:
        if not video_names:
            return []
        conn = self._connect()
        try:
            game_id = self._get_game_id(conn, user_id, game_name)
            if game_id is None:
                return []
            vf = "AND gv.name = ?" if version_name else ""
            vp = (version_name,) if version_name else ()
            rows = conn.execute(
                f"""
                SELECT DISTINCT r.video_filename
                FROM dash_runs r
                JOIN dash_game_versions gv ON gv.id = r.version_id
                WHERE gv.game_id = ? {vf}
                """,
                (game_id,) + vp,
            ).fetchall()
            existing = {row[0] for row in rows}
            duplicates = []
            for name in video_names:
                stem = Path(name).stem
                if any(fn.startswith(f"{stem}_run_") for fn in existing):
                    duplicates.append(name)
            return duplicates
        finally:
            conn.close()

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "totalRuns": 0,
            "avgDuration": None,
            "longestRun": None,
            "bossKillPercent": 0.0,
            "mostPopularItem": None,
        }
