-- Dashboard module schema initialization
-- Creates a dedicated namespace so these table names do not collide
-- with existing tables in other schemas/namespaces.

BEGIN;

CREATE SCHEMA IF NOT EXISTS dashboard;

-- ---------------------------------------------------------------------------
-- Core entities
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dashboard.users (
    id INTEGER PRIMARY KEY,
    email TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dashboard.games (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT games_user_fk
        FOREIGN KEY (user_id)
        REFERENCES dashboard.users(id)
        ON DELETE CASCADE,
    CONSTRAINT games_user_name_uk
        UNIQUE (user_id, name)
);

CREATE TABLE IF NOT EXISTS dashboard.game_versions (
    id BIGSERIAL PRIMARY KEY,
    game_id BIGINT NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT game_versions_game_fk
        FOREIGN KEY (game_id)
        REFERENCES dashboard.games(id)
        ON DELETE CASCADE,
    CONSTRAINT game_versions_game_name_uk
        UNIQUE (game_id, name)
);

CREATE TABLE IF NOT EXISTS dashboard.runs (
    id BIGSERIAL PRIMARY KEY,
    version_id BIGINT NOT NULL,
    video_filename TEXT,
    start_time DOUBLE PRECISION,
    end_time DOUBLE PRECISION,
    duration_seconds DOUBLE PRECISION,
    outcome TEXT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT runs_version_fk
        FOREIGN KEY (version_id)
        REFERENCES dashboard.game_versions(id)
        ON DELETE CASCADE,
    CONSTRAINT runs_outcome_chk
        CHECK (outcome IS NULL OR outcome IN ('death', 'win'))
);

-- ---------------------------------------------------------------------------
-- Item tracking
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dashboard.items (
    id BIGSERIAL PRIMARY KEY,
    game_id BIGINT NOT NULL,
    name TEXT NOT NULL,
    first_seen_version_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT items_game_fk
        FOREIGN KEY (game_id)
        REFERENCES dashboard.games(id)
        ON DELETE CASCADE,
    CONSTRAINT items_first_seen_version_fk
        FOREIGN KEY (first_seen_version_id)
        REFERENCES dashboard.game_versions(id)
        ON DELETE SET NULL,
    CONSTRAINT items_game_name_uk
        UNIQUE (game_id, name)
);

CREATE TABLE IF NOT EXISTS dashboard.item_pickups (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL,
    item_id BIGINT NOT NULL,
    picked_at_seconds DOUBLE PRECISION,
    options JSONB,
    CONSTRAINT item_pickups_run_fk
        FOREIGN KEY (run_id)
        REFERENCES dashboard.runs(id)
        ON DELETE CASCADE,
    CONSTRAINT item_pickups_item_fk
        FOREIGN KEY (item_id)
        REFERENCES dashboard.items(id)
        ON DELETE CASCADE
);

-- ---------------------------------------------------------------------------
-- Boss tracking
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dashboard.bosses (
    id BIGSERIAL PRIMARY KEY,
    game_id BIGINT NOT NULL,
    name TEXT NOT NULL,
    first_seen_version_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT bosses_game_fk
        FOREIGN KEY (game_id)
        REFERENCES dashboard.games(id)
        ON DELETE CASCADE,
    CONSTRAINT bosses_first_seen_version_fk
        FOREIGN KEY (first_seen_version_id)
        REFERENCES dashboard.game_versions(id)
        ON DELETE SET NULL,
    CONSTRAINT bosses_game_name_uk
        UNIQUE (game_id, name)
);

CREATE TABLE IF NOT EXISTS dashboard.boss_encounters (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL,
    boss_id BIGINT NOT NULL,
    start_time DOUBLE PRECISION,
    end_time DOUBLE PRECISION,
    duration_seconds DOUBLE PRECISION,
    player_died BOOLEAN,
    CONSTRAINT boss_encounters_run_fk
        FOREIGN KEY (run_id)
        REFERENCES dashboard.runs(id)
        ON DELETE CASCADE,
    CONSTRAINT boss_encounters_boss_fk
        FOREIGN KEY (boss_id)
        REFERENCES dashboard.bosses(id)
        ON DELETE CASCADE
);

COMMIT;
