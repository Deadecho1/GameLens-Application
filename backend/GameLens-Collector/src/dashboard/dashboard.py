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

from flask import Blueprint, jsonify, request

Dashboard = Blueprint("dashboard", __name__)


def _user_id():
    return request.headers.get("X-User-ID")


# ---------------------------------------------------------------------------
# Write endpoints — pipeline → Collector
# ---------------------------------------------------------------------------

@Dashboard.route("/games", methods=["POST"])
def create_or_get_game():
    """
    Create a game for the requesting user, or return the existing one.

    Body: { "name": str }
    Response: { "game_id": str }
    """
    # TODO: implement
    # body = request.get_json() or {}
    # name = body.get("name")
    # user_id = _user_id()
    # INSERT INTO games (user_id, name) ... ON CONFLICT DO NOTHING RETURNING id
    return jsonify({"error": "not implemented"}), 501


@Dashboard.route("/games/<game_id>/versions", methods=["POST"])
def create_or_get_version(game_id):
    """
    Create a version under game_id, or return the existing one.

    Body: { "name": str }
    Response: { "version_id": str }
    """
    # TODO: implement
    # body = request.get_json() or {}
    # name = body.get("name")
    # INSERT INTO game_versions (game_id, name) ... ON CONFLICT DO NOTHING RETURNING id
    return jsonify({"error": "not implemented"}), 501


@Dashboard.route("/games/<game_id>/versions/<version_id>/runs/batch", methods=["POST"])
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
    # TODO: implement
    # body = request.get_json() or {}
    # runs = body.get("runs", [])
    # Within a single transaction:
    #   For each run:
    #     INSERT INTO runs → run_id
    #     For each item_pickup:
    #       UPSERT items (game_id, name) → item_id
    #       INSERT item_pickups
    #     For each boss_encounter:
    #       UPSERT bosses (game_id, name) → boss_id
    #       INSERT boss_encounters
    return jsonify({"error": "not implemented"}), 501


# ---------------------------------------------------------------------------
# Read endpoints — frontend analytics tabs
# ---------------------------------------------------------------------------

@Dashboard.route("/dashboard/items", methods=["GET"])
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
    # TODO: implement
    # game_name = request.args.get("game_name")
    # version_name = request.args.get("version_name")
    # user_id = _user_id() or request.args.get("user_id")
    #
    # SELECT i.id, i.name,
    #   COUNT(DISTINCT ip.run_id) * 100.0 / NULLIF(COUNT(DISTINCT r.id), 0) AS popularity,
    #   AVG(ip.picked_at_seconds) / 60.0 AS avg_possession_minutes
    # FROM items i
    # JOIN games g ON g.id = i.game_id
    # LEFT JOIN item_pickups ip ON ip.item_id = i.id
    # LEFT JOIN runs r ON r.id = ip.run_id
    # LEFT JOIN game_versions v ON v.id = r.version_id
    # WHERE g.name = %(game_name)s AND g.user_id = %(user_id)s
    #   AND (%(version_name)s IS NULL OR v.name = %(version_name)s)
    # GROUP BY i.id, i.name
    return jsonify([]), 501


@Dashboard.route("/dashboard/bosses", methods=["GET"])
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
    # TODO: implement
    return jsonify([]), 501


@Dashboard.route("/dashboard/runs", methods=["GET"])
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
    # TODO: implement
    return jsonify([]), 501


@Dashboard.route("/dashboard/stats", methods=["GET"])
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
    # TODO: implement
    return jsonify({}), 501
