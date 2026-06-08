from datetime import date

import pytest
from flask import Flask, jsonify
from src.dashboard.dashboard import Dashboard
from src.db import DatabaseConnection
from src.errors import MissingCollectorParam


class _FakeCursor:
    def __init__(self, executed):
        self._executed = executed
        self._result = []

    def execute(self, query, params):
        self._executed.append((query, params))
        q = " ".join(query.split()).lower()

        if "from dashboard.games" in q:
            self._result = [(123,)]
            return

        if "from dashboard.runs r" in q:
            # Simulate DB honoring ORDER BY r.id DESC and version filter.
            self._result = [
                (102, date(2026, 3, 2), 600, "win", "v1.0"),
                (97, date(2026, 3, 1), 540, "death", "v1.0"),
            ]
            return

        if "from dashboard.boss_encounters" in q:
            self._result = []
            return

        if "from dashboard.item_pickups" in q:
            self._result = []
            return

        self._result = []

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


class _FakeCursorContext:
    def __init__(self, executed):
        self._cursor = _FakeCursor(executed)

    def __enter__(self):
        return self._cursor

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self):
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self, *args, **kwargs):
        return _FakeCursorContext(self.executed)


@pytest.fixture()
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(Dashboard, url_prefix="/api/v1")

    @app.errorhandler(MissingCollectorParam)
    def handle_collection_error(e):
        return (
            jsonify(
                {
                    "error": e.name,
                    "message": e.description,
                    "status_code": e.code,
                }
            ),
            e.code,
        )

    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def test_get_runs_filters_by_version_and_sorts_desc(monkeypatch, client):
    fake_conn = _FakeConnection()
    monkeypatch.setattr(
        DatabaseConnection,
        "get_connection",
        classmethod(lambda cls: fake_conn),
    )

    response = client.get(
        "/api/v1/dashboard/runs",
        query_string={"game_name": "Holocure", "version_name": "v1.0", "user_id": "7"},
    )

    assert response.status_code == 200
    body = response.get_json()

    # Response-level assertions: only requested version, newest-first by id.
    assert [r["id"] for r in body] == [102, 97]
    assert {r["gameVersion"] for r in body} == {"v1.0"}

    # Query-level assertions: SQL includes version filter and descending sort.
    runs_queries = [q for q, _ in fake_conn.executed if "FROM dashboard.runs r" in q]
    assert len(runs_queries) == 1
    runs_query = runs_queries[0]
    assert "AND gv.name = %s" in runs_query
    assert "ORDER BY r.id DESC" in runs_query
