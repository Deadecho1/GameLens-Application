-- Migration 001: add dedup constraint on dashboard.runs
-- Prevents uploading the same video file twice for the same version.
-- Safe to run multiple times (IF NOT EXISTS via DO block).

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'runs_version_filename_uk'
          AND conrelid = 'dashboard.runs'::regclass
    ) THEN
        ALTER TABLE dashboard.runs
            ADD CONSTRAINT runs_version_filename_uk UNIQUE (version_id, video_filename);
    END IF;
END;
$$;
