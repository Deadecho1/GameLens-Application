"""
GameLens Dashboard API — stub endpoints.

All routes return 501 until the DB implementation is added.

Write endpoints (called by the pipeline after processing):
  POST /api/v1/games
  POST /api/v1/games/<game_id>/versions
  POST /api/v1/games/<game_id>/versions/<version_id>/runs/batch

Read endpoints (called by the frontend analytics tabs):
  GET  /api/v1/dashboard/items
  GET  /api/v1/dashboard/bosses
  GET  /api/v1/dashboard/runs
  GET  /api/v1/dashboard/stats

Auth: X-User-ID header (placeholder — swap for JWT when auth is added).
Query params for reads: game_name, version_name (optional — omit for all versions).

Expected DB schema (to be implemented):
  users            id, email, created_at
  games            id, user_id→users, name, created_at
  game_versions    id, game_id→games, name, created_at
  runs             id, version_id→game_versions, video_filename,
                   start_time, end_time, duration_seconds, outcome, recorded_at
  items            id, game_id→games, name, first_seen_version_id, created_at
  item_pickups     id, run_id→runs, item_id→items, picked_at_seconds, options (JSON)
  bosses           id, game_id→games, name, first_seen_version_id, created_at
  boss_encounters  id, run_id→runs, boss_id→bosses,
                   start_time, end_time, duration_seconds, player_died
"""

from flasgger import swag_from
from flask import Blueprint, jsonify, request
from psycopg.types.json import Json

from src.db import DatabaseConnection
from src.errors import MissingCollectorParam

Dashboard = Blueprint("dashboard", __name__)


def _user_id():
    return request.headers.get("X-User-ID")


def _format_mm_ss(seconds):
    if seconds is None:
        return None

    total_seconds = max(int(round(float(seconds))), 0)
    minutes, sec = divmod(total_seconds, 60)
    return f"{minutes:02d}:{sec:02d}"


def _format_hh_mm_ss(seconds):
    if seconds is None:
        return None

    total_seconds = max(int(round(float(seconds))), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


# ---------------------------------------------------------------------------
# Sync download endpoints — local client pulls remote data
# ---------------------------------------------------------------------------


@Dashboard.route("/games", methods=["GET"])
def list_games():
    """List all games for the requesting user. Response: [{"id", "name"}]"""
    user_id_raw = (_user_id() or request.args.get("user_id") or "").strip()
    if not user_id_raw:
        raise MissingCollectorParam("X-User-ID header or user_id query param is required")
    try:
        user_id = int(user_id_raw)
    except ValueError:
        raise MissingCollectorParam("user_id must be an integer")

    with DatabaseConnection.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name FROM dashboard.games WHERE user_id = %s ORDER BY name",
                (user_id,),
            )
            rows = cur.fetchall()
    return jsonify([{"id": r[0], "name": r[1]} for r in rows]), 200


@Dashboard.route("/games/<game_id>/versions", methods=["GET"])
def list_versions(game_id):
    """List all versions for a game. Response: [{"id", "name"}]"""
    try:
        game_id_int = int(game_id)
    except ValueError:
        raise MissingCollectorParam("game_id must be an integer")

    with DatabaseConnection.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name FROM dashboard.game_versions WHERE game_id = %s ORDER BY name",
                (game_id_int,),
            )
            rows = cur.fetchall()
    return jsonify([{"id": r[0], "name": r[1]} for r in rows]), 200


@Dashboard.route("/games/<game_id>/versions/<version_id>/runs/export", methods=["GET"])
def export_runs(game_id, version_id):
    """
    Export all runs for a version with full pickup and encounter data.
    Used by the local client to pull remote data into local SQLite.
    Response: [{ video_filename, start_time, end_time, duration_seconds, outcome,
                 recorded_at, item_pickups: [{item_name, picked_at_seconds, options}],
                 boss_encounters: [{boss_name, start_time, end_time, duration_seconds, player_died}] }]
    """
    try:
        game_id_int = int(game_id)
        version_id_int = int(version_id)
    except ValueError:
        raise MissingCollectorParam("game_id and version_id must be integers")

    with DatabaseConnection.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id, r.video_filename, r.start_time, r.end_time,
                       r.duration_seconds, r.outcome, r.recorded_at
                FROM dashboard.runs r
                WHERE r.version_id = %s
                ORDER BY r.recorded_at, r.id
                """,
                (version_id_int,),
            )
            run_rows = cur.fetchall()

            if not run_rows:
                return jsonify([]), 200

            run_ids = [r[0] for r in run_rows]

            cur.execute(
                """
                SELECT ip.run_id, i.name, ip.picked_at_seconds, ip.options
                FROM dashboard.item_pickups ip
                JOIN dashboard.items i ON i.id = ip.item_id
                WHERE ip.run_id = ANY(%s)
                ORDER BY ip.run_id, ip.id
                """,
                (run_ids,),
            )
            pickup_rows = cur.fetchall()

            cur.execute(
                """
                SELECT be.run_id, b.name, be.start_time, be.end_time,
                       be.duration_seconds, be.player_died
                FROM dashboard.boss_encounters be
                JOIN dashboard.bosses b ON b.id = be.boss_id
                WHERE be.run_id = ANY(%s)
                ORDER BY be.run_id, be.id
                """,
                (run_ids,),
            )
            encounter_rows = cur.fetchall()

    pickups_by_run: dict = {}
    for run_id, item_name, picked_at, options in pickup_rows:
        pickups_by_run.setdefault(run_id, []).append({
            "item_name": item_name,
            "picked_at_seconds": picked_at,
            "options": options if isinstance(options, list) else (options or []),
        })

    encounters_by_run: dict = {}
    for run_id, boss_name, bs, be, bd, pd in encounter_rows:
        encounters_by_run.setdefault(run_id, []).append({
            "boss_name": boss_name,
            "start_time": bs,
            "end_time": be,
            "duration_seconds": bd,
            "player_died": bool(pd) if pd is not None else False,
        })

    result = []
    for run_id, video_filename, start_t, end_t, dur, outcome, recorded_at in run_rows:
        result.append({
            "video_filename": video_filename,
            "start_time": start_t,
            "end_time": end_t,
            "duration_seconds": dur,
            "outcome": outcome,
            "recorded_at": recorded_at.isoformat() if hasattr(recorded_at, "isoformat") else str(recorded_at),
            "item_pickups": pickups_by_run.get(run_id, []),
            "boss_encounters": encounters_by_run.get(run_id, []),
        })

    return jsonify(result), 200


# ---------------------------------------------------------------------------
# Write endpoints — pipeline → Collector
# ---------------------------------------------------------------------------


@Dashboard.route("/games", methods=["POST"])
@swag_from("../docs/dashboard_create_or_get_game.yml")
def create_or_get_game():
    """
    Create a game for the requesting user, or return the existing one.

    Body: { "name": str }
    Response: { "game_id": str }
    """
    body = request.get_json() or {}
    name = (body.get("name") or "").strip()
    user_id_raw = (_user_id() or "").strip()

    if not name:
        raise MissingCollectorParam("name is required")
    if not user_id_raw:
        raise MissingCollectorParam("X-User-ID header is required")

    try:
        user_id = int(user_id_raw)
    except ValueError:
        raise MissingCollectorParam("X-User-ID header must be an integer")

    try:
        with DatabaseConnection.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM dashboard.games
                    WHERE user_id = %s AND name = %s
                    LIMIT 1;
                    """,
                    (user_id, name),
                )
                existing = cur.fetchone()

                if existing:
                    game_id = existing[0]
                else:
                    cur.execute(
                        """
                        INSERT INTO dashboard.games (user_id, name)
                        VALUES (%s, %s)
                        RETURNING id;
                        """,
                        (user_id, name),
                    )
                    game_id = cur.fetchone()[0]
                conn.commit()

    except Exception as e:
        return jsonify(
            {"error": "Client Side Error", "message": str(e), "type": type(e).__name__}
        ), 400

    return jsonify({"game_id": game_id}), 200


@Dashboard.route("/games/<game_id>/versions", methods=["POST"])
@swag_from("../docs/dashboard_create_or_get_version.yml")
def create_or_get_version(game_id):
    """
    Create a version under game_id, or return the existing one.

    Body: { "name": str }
    Response: { "version_id": str }
    """
    body = request.get_json() or {}
    name = (body.get("name") or "").strip()

    if not name:
        raise MissingCollectorParam("name is required")

    try:
        game_id_int = int((game_id or "").strip())
    except ValueError:
        raise MissingCollectorParam("game_id must be an integer")

    try:
        with DatabaseConnection.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM dashboard.game_versions
                    WHERE game_id = %s AND name = %s
                    LIMIT 1;
                    """,
                    (game_id_int, name),
                )
                existing = cur.fetchone()

                if existing:
                    version_id = existing[0]
                else:
                    cur.execute(
                        """
                        INSERT INTO dashboard.game_versions (game_id, name)
                        VALUES (%s, %s)
                        RETURNING id;
                        """,
                        (game_id_int, name),
                    )
                    version_id = cur.fetchone()[0]
                conn.commit()

    except Exception as e:
        return jsonify(
            {"error": "Client Side Error", "message": str(e), "type": type(e).__name__}
        ), 400

    return jsonify({"version_id": version_id}), 200


@Dashboard.route("/games/<game_id>/versions/<version_id>/runs/batch", methods=["POST"])
@swag_from("../docs/dashboard_upload_runs_batch.yml")
def upload_runs_batch(game_id, version_id):
    """
    Batch-insert runs for a game version. Atomic — all or nothing.

    Body:
    {
      "runs": [
        {
          "video_filename": str,
          "start_time": float,
          "end_time": float,
          "duration_seconds": float,
          "outcome": "death" | "win",
          "item_pickups": [
            { "item_name": str, "picked_at_seconds": float, "options": [str] }
          ],
          "boss_encounters": [
            { "boss_name": str, "start_time": float, "end_time": float,
              "duration_seconds": float, "player_died": bool }
          ]
        }
      ]
    }

    For each item_name and boss_name, the backend resolves or creates the
    corresponding row in items / bosses (scoped to game_id), recording
    first_seen_version_id on first discovery.

    Response: { "inserted": int }
    """
    body = request.get_json() or {}
    runs = body.get("runs")

    if runs is None:
        raise MissingCollectorParam("runs is required")
    if not isinstance(runs, list):
        raise MissingCollectorParam("runs must be an array")

    try:
        game_id_int = int((game_id or "").strip())
        version_id_int = int((version_id or "").strip())
    except ValueError:
        raise MissingCollectorParam("game_id and version_id must be integers")

    try:
        with DatabaseConnection.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM dashboard.game_versions
                    WHERE id = %s AND game_id = %s
                    LIMIT 1;
                    """,
                    (version_id_int, game_id_int),
                )
                if not cur.fetchone():
                    raise MissingCollectorParam(
                        "version_id does not exist for the provided game_id"
                    )

                inserted = 0

                for run in runs:
                    if not isinstance(run, dict):
                        raise MissingCollectorParam("each run must be an object")

                    video_filename = run.get("video_filename")
                    start_time = run.get("start_time")
                    end_time = run.get("end_time")
                    duration_seconds = run.get("duration_seconds")
                    outcome = run.get("outcome")

                    if not video_filename:
                        raise MissingCollectorParam("video_filename is required")
                    if start_time is None:
                        raise MissingCollectorParam("start_time is required")
                    if end_time is None:
                        raise MissingCollectorParam("end_time is required")
                    if duration_seconds is None:
                        raise MissingCollectorParam("duration_seconds is required")
                    if outcome not in ("death", "win"):
                        raise MissingCollectorParam("outcome must be 'death' or 'win'")

                    cur.execute(
                        """
                        INSERT INTO dashboard.runs (
                            version_id,
                            video_filename,
                            start_time,
                            end_time,
                            duration_seconds,
                            outcome
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (
                            version_id_int,
                            video_filename,
                            start_time,
                            end_time,
                            duration_seconds,
                            outcome,
                        ),
                    )
                    run_id = cur.fetchone()[0]

                    item_pickups = run.get("item_pickups", [])
                    if not isinstance(item_pickups, list):
                        raise MissingCollectorParam("item_pickups must be an array")

                    for pickup in item_pickups:
                        if not isinstance(pickup, dict):
                            raise MissingCollectorParam(
                                "each item_pickup must be an object"
                            )

                        item_name = (pickup.get("item_name") or "").strip()
                        picked_at_seconds = pickup.get("picked_at_seconds")
                        options = pickup.get("options", [])

                        if not item_name:
                            raise MissingCollectorParam("item_name is required")

                        # DO UPDATE is a no-op on name (keeps existing value) — used
                        # solely so RETURNING id fires on both insert and conflict paths.
                        cur.execute(
                            """
                            INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (game_id, name)
                            DO UPDATE SET name = EXCLUDED.name
                            RETURNING id;
                            """,
                            (game_id_int, item_name, version_id_int),
                        )
                        item_id = cur.fetchone()[0]

                        cur.execute(
                            """
                            INSERT INTO dashboard.item_pickups (
                                run_id,
                                item_id,
                                picked_at_seconds,
                                options
                            )
                            VALUES (%s, %s, %s, %s);
                            """,
                            (run_id, item_id, picked_at_seconds, Json(options)),
                        )

                    boss_encounters = run.get("boss_encounters", [])
                    if not isinstance(boss_encounters, list):
                        raise MissingCollectorParam("boss_encounters must be an array")

                    for encounter in boss_encounters:
                        if not isinstance(encounter, dict):
                            raise MissingCollectorParam(
                                "each boss_encounter must be an object"
                            )

                        boss_name = (encounter.get("boss_name") or "").strip()
                        boss_start_time = encounter.get("start_time")
                        boss_end_time = encounter.get("end_time")
                        boss_duration_seconds = encounter.get("duration_seconds")
                        player_died = encounter.get("player_died")

                        if not boss_name:
                            raise MissingCollectorParam("boss_name is required")

                        # DO UPDATE is a no-op on name (keeps existing value) — used
                        # solely so RETURNING id fires on both insert and conflict paths.
                        cur.execute(
                            """
                            INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (game_id, name)
                            DO UPDATE SET name = EXCLUDED.name
                            RETURNING id;
                            """,
                            (game_id_int, boss_name, version_id_int),
                        )
                        boss_id = cur.fetchone()[0]

                        cur.execute(
                            """
                            INSERT INTO dashboard.boss_encounters (
                                run_id,
                                boss_id,
                                start_time,
                                end_time,
                                duration_seconds,
                                player_died
                            )
                            VALUES (%s, %s, %s, %s, %s, %s);
                            """,
                            (
                                run_id,
                                boss_id,
                                boss_start_time,
                                boss_end_time,
                                boss_duration_seconds,
                                player_died,
                            ),
                        )

                    inserted += 1

                conn.commit()

    except Exception as e:
        return jsonify(
            {"error": "Client Side Error", "message": str(e), "type": type(e).__name__}
        ), 400

    return jsonify({"inserted": inserted}), 200


# ---------------------------------------------------------------------------
# Read endpoints — frontend analytics tabs
# ---------------------------------------------------------------------------


@Dashboard.route("/dashboard/items", methods=["GET"])
@swag_from("../docs/dashboard_get_items.yml")
def get_items():
    """
    Return all items for a game, with stats aggregated from runs in the
    given version. Items with no runs in the requested version have null stats.

    Query params:
      game_name   — required
      version_name — optional; omit for cross-version aggregation
      user_id     — may also come from X-User-ID header

    Response: [
      {
        "id": str,
        "name": str,
        "popularity": float,       -- pick rate across runs (0–100)
        "impact": str,             -- "High" | "Medium" | "Low" (derived)
        "category": str,           -- discovered from run data (future)
        "logicTag": str,           -- discovered from run data (future)
        "rarity": str,             -- discovered from run data (future)
        "avgPossessionMinutes": float  -- avg seconds held / 60
      }
    ]
    """
    game_name = (request.args.get("game_name") or "").strip()
    version_name = (request.args.get("version_name") or "").strip() or None
    user_id_raw = (_user_id() or request.args.get("user_id") or "").strip()

    if not game_name:
        raise MissingCollectorParam("game_name is required")
    if not user_id_raw:
        raise MissingCollectorParam(
            "X-User-ID header or user_id query param is required"
        )

    try:
        user_id = int(user_id_raw)
    except ValueError:
        raise MissingCollectorParam("user_id must be an integer")

    try:
        with DatabaseConnection.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM dashboard.games
                    WHERE name = %s AND user_id = %s
                    LIMIT 1;
                    """,
                    (game_name, user_id),
                )
                game = cur.fetchone()
                if not game:
                    return jsonify({"error": "Game not found"}), 404

                game_id = game[0]

                cur.execute(
                    """
                    WITH filtered_runs AS (
                        SELECT r.id
                        FROM dashboard.runs r
                        JOIN dashboard.game_versions gv ON gv.id = r.version_id
                        WHERE gv.game_id = %s
                          AND (%s::text IS NULL OR gv.name = %s::text)
                    ),
                    run_count AS (
                        SELECT COUNT(*)::FLOAT AS total_runs
                        FROM filtered_runs
                    )
                    SELECT
                        i.id,
                        i.name,
                        CASE
                            WHEN rc.total_runs = 0 THEN NULL
                            WHEN COUNT(DISTINCT fr.id) = 0 THEN NULL
                            ELSE COUNT(DISTINCT fr.id) * 100.0 / rc.total_runs
                        END AS popularity,
                        AVG(ip.picked_at_seconds) FILTER (
                            WHERE fr.id IS NOT NULL
                        ) / 60.0 AS avg_possession_minutes
                    FROM dashboard.items i
                    CROSS JOIN run_count rc
                    LEFT JOIN dashboard.item_pickups ip ON ip.item_id = i.id
                    LEFT JOIN filtered_runs fr ON fr.id = ip.run_id
                    WHERE i.game_id = %s
                    GROUP BY i.id, i.name, rc.total_runs
                    HAVING COUNT(DISTINCT fr.id) > 0
                    ORDER BY i.name;
                    """,
                    (game_id, version_name, version_name, game_id),
                )
                item_rows = cur.fetchall()

    except Exception as e:
        return jsonify(
            {"error": "Client Side Error", "message": str(e), "type": type(e).__name__}
        ), 400

    response = []
    for item_id, item_name, popularity, avg_possession_minutes in item_rows:
        if popularity is None:
            impact = None
        elif popularity >= 66.0:
            impact = "High"
        elif popularity >= 33.0:
            impact = "Medium"
        else:
            impact = "Low"

        response.append(
            {
                "id": item_id,
                "name": item_name,
                "popularity": round(float(popularity), 2)
                if popularity is not None
                else None,
                "impact": impact,
                "category": None,
                "logicTag": None,
                "rarity": None,
                "avgPossessionMinutes": (
                    round(float(avg_possession_minutes), 2)
                    if avg_possession_minutes is not None
                    else None
                ),
            }
        )

    return jsonify(response), 200


@Dashboard.route("/dashboard/bosses", methods=["GET"])
@swag_from("../docs/dashboard_get_bosses.yml")
def get_bosses():
    """
    Return all bosses for a game with aggregated encounter stats.
    Bosses with no encounters in the requested version have null stats.

    Query params: game_name, version_name (optional), user_id

    Response: [
      {
        "id": str,
        "name": str,
        "lifespan": str,              -- avg encounter duration formatted "MM:SS"
        "status": str,                -- "Defeated" | "Alive" (derived from player_died)
        "globalLifespanSamples": [str],
        "gearSynergies": [
          { "itemIds": [str], "timeReductionPct": float }
        ],
        "itemEffectiveness": [
          { "itemId": str, "timeReductionVsGlobalPct": float }
        ]
      }
    ]
    """
    game_name = (request.args.get("game_name") or "").strip()
    version_name = (request.args.get("version_name") or "").strip() or None
    user_id_raw = (_user_id() or request.args.get("user_id") or "").strip()

    if not game_name:
        raise MissingCollectorParam("game_name is required")
    if not user_id_raw:
        raise MissingCollectorParam(
            "X-User-ID header or user_id query param is required"
        )

    try:
        user_id = int(user_id_raw)
    except ValueError:
        raise MissingCollectorParam("user_id must be an integer")

    try:
        with DatabaseConnection.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM dashboard.games
                    WHERE name = %s AND user_id = %s
                    LIMIT 1;
                    """,
                    (game_name, user_id),
                )
                game = cur.fetchone()
                if not game:
                    return jsonify({"error": "Game not found"}), 404

                game_id = game[0]

                cur.execute(
                    """
                    SELECT
                        b.id,
                        b.name,
                        AVG(be.duration_seconds) FILTER (
                            WHERE (%s::text IS NULL OR gv.name = %s::text)
                        ) AS avg_duration_seconds,
                        -- NULL player_died is treated as "player died" (TRUE).
                        -- A boss is "Defeated" only if there is an explicit player_died=FALSE encounter.
                        COALESCE(
                            BOOL_OR(COALESCE(be.player_died, TRUE) = FALSE) FILTER (
                                WHERE (%s::text IS NULL OR gv.name = %s::text)
                            ),
                            FALSE
                        ) AS any_defeated
                    FROM dashboard.bosses b
                    LEFT JOIN dashboard.boss_encounters be ON be.boss_id = b.id
                    LEFT JOIN dashboard.runs r ON r.id = be.run_id
                    LEFT JOIN dashboard.game_versions gv ON gv.id = r.version_id
                    WHERE b.game_id = %s
                    GROUP BY b.id, b.name
                    ORDER BY b.name;
                    """,
                    (version_name, version_name, version_name, version_name, game_id),
                )
                boss_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT b.id, be.duration_seconds
                    FROM dashboard.bosses b
                    LEFT JOIN dashboard.boss_encounters be ON be.boss_id = b.id
                    LEFT JOIN dashboard.runs r ON r.id = be.run_id
                    LEFT JOIN dashboard.game_versions gv ON gv.id = r.version_id
                    WHERE b.game_id = %s
                      AND be.duration_seconds IS NOT NULL
                      AND (%s::text IS NULL OR gv.name = %s::text)
                    ORDER BY b.id, be.id;
                    """,
                    (game_id, version_name, version_name),
                )
                sample_rows = cur.fetchall()

    except Exception as e:
        return jsonify(
            {"error": "Client Side Error", "message": str(e), "type": type(e).__name__}
        ), 400

    samples_by_boss = {}
    for boss_id, duration in sample_rows:
        samples_by_boss.setdefault(boss_id, []).append(_format_mm_ss(duration))

    response = []
    for boss_id, boss_name, avg_duration_seconds, any_defeated in boss_rows:
        response.append(
            {
                "id": boss_id,
                "name": boss_name,
                "lifespan": _format_mm_ss(avg_duration_seconds),
                "status": (
                    "Defeated"
                    if avg_duration_seconds is not None and any_defeated
                    else "Alive"
                    if avg_duration_seconds is not None
                    else None
                ),
                "globalLifespanSamples": samples_by_boss.get(boss_id, []),
                "gearSynergies": [],
                "itemEffectiveness": [],
            }
        )

    return jsonify(response), 200


@Dashboard.route("/dashboard/runs", methods=["GET"])
@swag_from("../docs/dashboard_get_runs.yml")
def get_runs():
    """
    Return run history for a game/version.

    Query params: game_name, version_name (optional), user_id

    Response: [
      {
        "id": str,
        "date": str,                   -- ISO date "YYYY-MM-DD"
        "duration": str,               -- formatted "HH:MM:SS"
        "outcome": str,                -- "death" | "win"
        "bossEncounters": [
          {
            "bossId": str,
            "lifespan": str,           -- formatted "MM:SS"
            "loadout": [str]           -- item IDs carried at time of encounter
          }
        ]
      }
    ]
    """
    game_name = (request.args.get("game_name") or "").strip()
    version_name = (request.args.get("version_name") or "").strip() or None
    user_id_raw = (_user_id() or request.args.get("user_id") or "").strip()

    if not game_name:
        raise MissingCollectorParam("game_name is required")
    if not user_id_raw:
        raise MissingCollectorParam(
            "X-User-ID header or user_id query param is required"
        )

    try:
        user_id = int(user_id_raw)
    except ValueError:
        raise MissingCollectorParam("user_id must be an integer")

    try:
        with DatabaseConnection.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM dashboard.games
                    WHERE name = %s AND user_id = %s
                    LIMIT 1;
                    """,
                    (game_name, user_id),
                )
                game = cur.fetchone()
                if not game:
                    return jsonify({"error": "Game not found"}), 404

                game_id = game[0]

                version_filter = "AND gv.name = %s" if version_name else ""
                version_params = (version_name,) if version_name else ()

                cur.execute(
                    f"""
                    SELECT
                        r.id,
                        r.recorded_at::date,
                        r.duration_seconds,
                        r.outcome,
                        gv.name AS game_version
                    FROM dashboard.runs r
                    JOIN dashboard.game_versions gv ON gv.id = r.version_id
                    WHERE gv.game_id = %s {version_filter}
                    ORDER BY r.id DESC;
                    """,
                    (game_id,) + version_params,
                )
                run_rows = cur.fetchall()

                run_ids = [row[0] for row in run_rows]
                encounters_by_run = {}

                pickups_by_run = {}

                if run_ids:
                    cur.execute(
                        """
                        SELECT
                            be.run_id,
                            be.boss_id,
                            be.duration_seconds,
                            be.start_time,
                            COALESCE(
                                ARRAY_AGG(DISTINCT ip.item_id) FILTER (
                                    WHERE ip.item_id IS NOT NULL
                                ),
                                '{}'
                            ) AS loadout
                        FROM dashboard.boss_encounters be
                        LEFT JOIN dashboard.item_pickups ip
                          ON ip.run_id = be.run_id
                         AND ip.picked_at_seconds IS NOT NULL
                         AND (
                            be.start_time IS NULL
                            OR ip.picked_at_seconds <= be.start_time
                         )
                        WHERE be.run_id = ANY(%s)
                        GROUP BY be.id, be.run_id, be.boss_id, be.duration_seconds, be.start_time
                        ORDER BY be.run_id, be.id;
                        """,
                        (run_ids,),
                    )

                    for run_id, boss_id, lifespan_seconds, start_time, loadout in cur.fetchall():
                        encounters_by_run.setdefault(run_id, []).append(
                            {
                                "bossId": boss_id,
                                "lifespan": _format_mm_ss(lifespan_seconds),
                                "loadout": loadout or [],
                                "startTime": float(start_time) if start_time is not None else None,
                            }
                        )

                    cur.execute(
                        """
                        SELECT run_id, item_id, picked_at_seconds, options
                        FROM dashboard.item_pickups
                        WHERE run_id = ANY(%s)
                          AND picked_at_seconds IS NOT NULL
                        ORDER BY run_id, picked_at_seconds;
                        """,
                        (run_ids,),
                    )

                    for run_id, item_id, picked_at_sec, options in cur.fetchall():
                        pickups_by_run.setdefault(run_id, []).append(
                            {
                                "itemId": item_id,
                                "pickedAtSeconds": float(picked_at_sec),
                                "options": options if isinstance(options, list) else [],
                            }
                        )

    except Exception as e:
        return jsonify(
            {"error": "Client Side Error", "message": str(e), "type": type(e).__name__}
        ), 400

    response = []
    for run_id, run_date, duration_seconds, outcome, game_version in run_rows:
        response.append(
            {
                "id": run_id,
                "date": run_date.isoformat() if run_date else None,
                "duration": _format_hh_mm_ss(duration_seconds),
                "outcome": outcome,
                "gameVersion": game_version,
                "bossEncounters": encounters_by_run.get(run_id, []),
                "itemPickups": pickups_by_run.get(run_id, []),
            }
        )

    return jsonify(response), 200


@Dashboard.route("/dashboard/stats", methods=["GET"])
@swag_from("../docs/dashboard_get_stats.yml")
def get_stats():
    """
    Return aggregated KPIs for the general analytics tab.

    Query params: game_name, version_name (optional), user_id

    Response:
    {
      "totalRuns": int,
      "avgDuration": str,        -- formatted "HH:MM:SS"
      "longestRun": str,         -- formatted "HH:MM:SS"
      "bossKillPercent": float,  -- % of boss encounters where player_died=False
      "mostPopularItem": str     -- item name with highest pick rate
    }
    """
    game_name = (request.args.get("game_name") or "").strip()
    version_name = (request.args.get("version_name") or "").strip() or None
    user_id_raw = (_user_id() or request.args.get("user_id") or "").strip()

    if not game_name:
        raise MissingCollectorParam("game_name is required")
    if not user_id_raw:
        raise MissingCollectorParam(
            "X-User-ID header or user_id query param is required"
        )

    try:
        user_id = int(user_id_raw)
    except ValueError:
        raise MissingCollectorParam("user_id must be an integer")

    try:
        with DatabaseConnection.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM dashboard.games
                    WHERE name = %s AND user_id = %s
                    LIMIT 1;
                    """,
                    (game_name, user_id),
                )
                game = cur.fetchone()
                if not game:
                    return jsonify({"error": "Game not found"}), 404

                game_id = game[0]

                cur.execute(
                    """
                    SELECT
                        COUNT(*)::INT AS total_runs,
                        AVG(r.duration_seconds) AS avg_duration_seconds,
                        MAX(r.duration_seconds) AS longest_duration_seconds
                    FROM dashboard.runs r
                    JOIN dashboard.game_versions gv ON gv.id = r.version_id
                    WHERE gv.game_id = %s
                      AND (%s::text IS NULL OR gv.name = %s::text);
                    """,
                    (game_id, version_name, version_name),
                )
                total_runs, avg_duration_seconds, longest_duration_seconds = (
                    cur.fetchone()
                )

                cur.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE be.player_died = FALSE)::FLOAT,
                        COUNT(*)::FLOAT
                    FROM dashboard.boss_encounters be
                    JOIN dashboard.runs r ON r.id = be.run_id
                    JOIN dashboard.game_versions gv ON gv.id = r.version_id
                    WHERE gv.game_id = %s
                      AND (%s::text IS NULL OR gv.name = %s::text);
                    """,
                    (game_id, version_name, version_name),
                )
                defeated_count, total_encounters = cur.fetchone()

                cur.execute(
                    """
                    WITH filtered_runs AS (
                        SELECT r.id
                        FROM dashboard.runs r
                        JOIN dashboard.game_versions gv ON gv.id = r.version_id
                        WHERE gv.game_id = %s
                          AND (%s::text IS NULL OR gv.name = %s::text)
                    )
                    SELECT i.name
                    FROM dashboard.item_pickups ip
                    JOIN dashboard.items i ON i.id = ip.item_id
                    WHERE ip.run_id IN (SELECT id FROM filtered_runs)
                    GROUP BY i.id, i.name
                    ORDER BY COUNT(DISTINCT ip.run_id) DESC, i.name ASC
                    LIMIT 1;
                    """,
                    (game_id, version_name, version_name),
                )
                item_row = cur.fetchone()

    except Exception as e:
        return jsonify(
            {"error": "Client Side Error", "message": str(e), "type": type(e).__name__}
        ), 400

    boss_kill_percent = 0.0
    if total_encounters and total_encounters > 0:
        boss_kill_percent = round((defeated_count or 0.0) * 100.0 / total_encounters, 2)

    return jsonify(
        {
            "totalRuns": total_runs or 0,
            "avgDuration": _format_hh_mm_ss(avg_duration_seconds),
            "longestRun": _format_hh_mm_ss(longest_duration_seconds),
            "bossKillPercent": boss_kill_percent,
            "mostPopularItem": item_row[0] if item_row else None,
        }
    ), 200
