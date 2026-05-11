"""
Auth blueprint — lightweight email-based identity (no password yet).

POST /api/v1/auth/login   { email } → { user_id }
POST /api/v1/auth/logout  → { ok: true }  (stateless — client drops credentials)

Real password auth and token validation to be wired once auth requirements are settled.
"""

from flask import Blueprint, jsonify, request

from src.db import DatabaseConnection
from src.errors import MissingCollectorParam

Auth = Blueprint("auth", __name__)


@Auth.route("/auth/login", methods=["POST"])
def login():
    body = request.get_json() or {}
    email = (body.get("email") or "").strip().lower()
    if not email:
        raise MissingCollectorParam("email is required")

    with DatabaseConnection.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dashboard.users (email)
                VALUES (%s)
                ON CONFLICT (email) DO NOTHING;
                """,
                (email,),
            )
            cur.execute(
                "SELECT id FROM dashboard.users WHERE email = %s LIMIT 1;",
                (email,),
            )
            row = cur.fetchone()
            conn.commit()

    if row is None:
        return jsonify({"error": "Failed to create user"}), 500

    return jsonify({"user_id": row[0]}), 200


@Auth.route("/auth/logout", methods=["POST"])
def logout():
    return jsonify({"ok": True}), 200
