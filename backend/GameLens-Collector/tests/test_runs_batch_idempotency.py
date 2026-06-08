import pytest
from flask import Flask
from src.dashboard.dashboard import Dashboard
from src.db import DatabaseConnection


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self._last_fetchone = None

    def execute(self, query, params=None):
        self._conn.executed.append((query, params))
        compact_query = " ".join(query.split()).lower()

        if "select 1 from dashboard.game_versions" in compact_query:
            # Pretend the requested version exists for the game.
            self._last_fetchone = (1,)
            return self

        if "insert into dashboard.runs" in compact_query:
            self._conn.run_insert_attempts += 1
            version_id = params[0]
            video_filename = params[1]
            dedup_key = (version_id, video_filename)

            if dedup_key in self._conn.inserted_runs:
                raise Exception(
                    'duplicate key value violates unique constraint "runs_version_filename_uk"'
                )

            self._conn.next_run_id += 1
            run_id = self._conn.next_run_id
            self._conn.inserted_runs[dedup_key] = run_id
            self._last_fetchone = (run_id,)
            return self

        # Queries not used by this test path (item/boss loops are empty in payload).
        self._last_fetchone = None
        return self

    def fetchone(self):
        return self._last_fetchone


class _FakeCursorContext:
    def __init__(self, conn):
        self._cursor = _FakeCursor(conn)

    def __enter__(self):
        return self._cursor

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self):
        self.executed = []
        self.commit_calls = 0
        self.run_insert_attempts = 0
        self.next_run_id = 0
        self.inserted_runs = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self, *args, **kwargs):
        return _FakeCursorContext(self)

    def commit(self):
        self.commit_calls += 1


@pytest.fixture()
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(Dashboard, url_prefix="/api/v1")
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def test_upload_runs_batch_duplicate_video_for_same_version_is_rejected(
    monkeypatch, client
):
    """duplicate run upload should not create a second run row."""
    fake_conn = _FakeConnection()
    monkeypatch.setattr(
        DatabaseConnection,
        "get_connection",
        classmethod(lambda cls: fake_conn),
    )

    payload = {
        "runs": [
            {
                "video_filename": "tc07_duplicate.mp4",
                "start_time": 10.0,
                "end_time": 120.0,
                "duration_seconds": 110.0,
                "outcome": "death",
                "item_pickups": [],
                "boss_encounters": [],
            }
        ]
    }

    endpoint = "/api/v1/games/1/versions/10/runs/batch"

    first_response = client.post(endpoint, json=payload)
    assert first_response.status_code == 200
    assert first_response.get_json() == {"inserted": 1}
    assert fake_conn.commit_calls == 1
    assert len(fake_conn.inserted_runs) == 1

    second_response = client.post(endpoint, json=payload)
    assert second_response.status_code == 400
    body = second_response.get_json()
    assert body["error"] == "Client Side Error"
    assert "runs_version_filename_uk" in body["message"]

    # Idempotency/dedup assertion: second upload did not create another run row.
    assert fake_conn.commit_calls == 1
    assert len(fake_conn.inserted_runs) == 1
    assert fake_conn.run_insert_attempts == 2
