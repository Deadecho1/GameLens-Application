from flask import Blueprint, jsonify, request
from psycopg.rows import dict_row

from src.db import DatabaseConnection
from src.errors import MissingCollectorParam
from src.util import validate_data

Boss = Blueprint("boss", __name__)


@Boss.route("/collect/boss", methods=["POST"])
def insert_boss_fight():
    data = request.get_json() or {}
    validate_data(["run_id", "boss_name", "duration", "player_died", "start_time", "end_time"], data)

    run_id = data.get("run_id")
    boss_name = data.get("boss_name")
    duration = data.get("duration")
    player_died = data.get("player_died")
    start_time = data.get("start_time")
    end_time = data.get("end_time")

    try:
        with DatabaseConnection.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO boss_fight (
                        run_id,
                        boss_name,
                        duration,
                        player_died,
                        start_time,
                        end_time
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (run_id, boss_name, duration, player_died, start_time, end_time),
                )
                boss_fight_id = cur.fetchone()[0]
                conn.commit()

    except Exception as e:
        return jsonify(
            {"error": "Client Side Error", "message": str(e), "type": type(e).__name__}
        ), 400

    return jsonify({"message": "Boss fight inserted successfully", "boss_fight_id": boss_fight_id}), 200


@Boss.route("/collect/boss", methods=["GET"])
def get_boss_fights():
    run_id = request.args.get("run_id")
    if not run_id:
        raise MissingCollectorParam("run_id is required")

    try:
        with DatabaseConnection.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                boss_fights = cur.execute(
                    """
                    SELECT
                        id AS boss_fight_id,
                        run_id,
                        boss_name,
                        duration,
                        player_died,
                        start_time,
                        end_time
                    FROM boss_fight
                    WHERE run_id = %s;
                    """,
                    (run_id,),
                ).fetchall()

    except Exception as e:
        return jsonify(
            {"error": "Client Side Error", "message": str(e), "type": type(e).__name__}
        ), 400

    return jsonify({"data": boss_fights}), 200
