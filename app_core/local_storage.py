from __future__ import annotations

import sqlite3
from pathlib import Path

LOCAL_DB_FILENAME = "gamelens_local.db"
LOCAL_USER_ID = 0  # reserved for unauthenticated local user


def get_local_db_path() -> Path:
    from .config import AppConfig
    cfg = AppConfig.load()
    return cfg.project_root / "data" / LOCAL_DB_FILENAME


def open_local_db(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or get_local_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _apply_schema(conn)
    return conn


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS dash_games (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL DEFAULT 0,
            name       TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            UNIQUE(user_id, name)
        );

        CREATE TABLE IF NOT EXISTS dash_game_versions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id    INTEGER NOT NULL REFERENCES dash_games(id) ON DELETE CASCADE,
            name       TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            UNIQUE(game_id, name)
        );

        CREATE TABLE IF NOT EXISTS dash_runs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id       INTEGER NOT NULL REFERENCES dash_game_versions(id) ON DELETE CASCADE,
            video_filename   TEXT NOT NULL,
            start_time       REAL NOT NULL,
            end_time         REAL NOT NULL,
            duration_seconds REAL NOT NULL,
            outcome          TEXT NOT NULL CHECK(outcome IN ('death', 'win')),
            recorded_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            synced_at        TEXT,
            UNIQUE(version_id, video_filename)
        );

        CREATE TABLE IF NOT EXISTS dash_items (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id               INTEGER NOT NULL REFERENCES dash_games(id) ON DELETE CASCADE,
            name                  TEXT NOT NULL,
            first_seen_version_id INTEGER REFERENCES dash_game_versions(id) ON DELETE SET NULL,
            UNIQUE(game_id, name)
        );

        CREATE TABLE IF NOT EXISTS dash_item_pickups (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id            INTEGER NOT NULL REFERENCES dash_runs(id) ON DELETE CASCADE,
            item_id           INTEGER NOT NULL REFERENCES dash_items(id) ON DELETE CASCADE,
            picked_at_seconds REAL,
            options           TEXT
        );

        CREATE TABLE IF NOT EXISTS dash_bosses (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id               INTEGER NOT NULL REFERENCES dash_games(id) ON DELETE CASCADE,
            name                  TEXT NOT NULL,
            first_seen_version_id INTEGER REFERENCES dash_game_versions(id) ON DELETE SET NULL,
            UNIQUE(game_id, name)
        );

        CREATE TABLE IF NOT EXISTS dash_boss_encounters (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id           INTEGER NOT NULL REFERENCES dash_runs(id) ON DELETE CASCADE,
            boss_id          INTEGER NOT NULL REFERENCES dash_bosses(id) ON DELETE CASCADE,
            start_time       REAL,
            end_time         REAL,
            duration_seconds REAL,
            player_died      INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_dash_runs_version
            ON dash_runs(version_id, recorded_at DESC);
        CREATE INDEX IF NOT EXISTS idx_dash_item_pickups_run
            ON dash_item_pickups(run_id, item_id);
        CREATE INDEX IF NOT EXISTS idx_dash_boss_enc_run
            ON dash_boss_encounters(run_id, boss_id);
    """)
    conn.commit()
