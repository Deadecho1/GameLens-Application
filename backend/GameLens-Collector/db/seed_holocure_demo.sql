BEGIN;

-- Clear existing Holocure data
DELETE FROM dashboard.boss_encounters WHERE run_id IN (
  SELECT r.id FROM dashboard.runs r
  JOIN dashboard.game_versions gv ON gv.id = r.version_id
  JOIN dashboard.games g ON g.id = gv.game_id
  WHERE g.name = 'Holocure'
);
DELETE FROM dashboard.item_pickups WHERE run_id IN (
  SELECT r.id FROM dashboard.runs r
  JOIN dashboard.game_versions gv ON gv.id = r.version_id
  JOIN dashboard.games g ON g.id = gv.game_id
  WHERE g.name = 'Holocure'
);
DELETE FROM dashboard.runs WHERE version_id IN (
  SELECT gv.id FROM dashboard.game_versions gv
  JOIN dashboard.games g ON g.id = gv.game_id
  WHERE g.name = 'Holocure'
);
DELETE FROM dashboard.items WHERE game_id IN (SELECT id FROM dashboard.games WHERE name='Holocure');
DELETE FROM dashboard.bosses WHERE game_id IN (SELECT id FROM dashboard.games WHERE name='Holocure');
DELETE FROM dashboard.game_versions WHERE game_id IN (SELECT id FROM dashboard.games WHERE name='Holocure');

-- Ensure Holocure game exists (user_id=1)
INSERT INTO dashboard.games (user_id, name) VALUES (1, 'Holocure')
  ON CONFLICT DO NOTHING;

-- Create versions
INSERT INTO dashboard.game_versions (game_id, name)
  SELECT id, 'V0.1' FROM dashboard.games WHERE name='Holocure' AND user_id=1;
INSERT INTO dashboard.game_versions (game_id, name)
  SELECT id, 'V0.2' FROM dashboard.games WHERE name='Holocure' AND user_id=1;

DO $$
DECLARE
  v_game_id INT;
  v_version_id INT;
  v_run_id INT;
  v_item_id INT;
  v_boss_id INT;
BEGIN
  SELECT id INTO v_game_id FROM dashboard.games WHERE name='Holocure' AND user_id=1 LIMIT 1;
  SELECT id INTO v_version_id FROM dashboard.game_versions WHERE game_id=v_game_id AND name='V0.1' LIMIT 1;

  -- run 1: death, 791s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_001.json', 24, 815, 791, 'death', '2026-03-31T17:36:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 405, 225, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 482, 62, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 656, 56, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 779, 59, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 352, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 247, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 541, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 435, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 499, '[]');

  -- run 2: death, 588s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_002.json', 31, 619, 588, 'death', '2026-03-31T16:08:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 415, 235, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 593, 173, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 398, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 164, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 286, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 420, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 121, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Elite Lava Bucket', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 289, '[]');

  -- run 3: death, 906s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_003.json', 77, 983, 906, 'death', '2026-03-31T22:32:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 275, 95, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 504, 84, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 740, 140, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 960, 240, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 931, 31, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 695, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 412, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 478, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 559, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 492, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 153, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Branch', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 283, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Tako''s Breath', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 260, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 95, '[]');

  -- run 4: death, 606s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_004.json', 70, 676, 606, 'death', '2026-03-31T12:37:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 283, 103, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 615, 195, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 631, 31, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 101, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 246, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 123, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 172, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 266, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 157, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 477, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 68, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'GWS', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 256, '[]');

  -- run 5: death, 795s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_005.json', 69, 864, 795, 'death', '2026-03-31T12:41:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 228, 48, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 488, 68, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 837, 237, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 780, 60, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 270, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 443, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 152, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 613, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 282, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 622, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 70, '[]');

  -- run 6: death, 521s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_006.json', 84, 605, 521, 'death', '2026-03-31T13:26:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 374, 194, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 522, 102, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 369, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 83, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 98, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 165, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 89, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 84, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 410, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 313, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'RPG', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 109, '[]');

  -- run 7: death, 699s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_007.json', 87, 786, 699, 'death', '2026-03-31T17:13:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 387, 207, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 532, 112, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 694, 94, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 446, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 188, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 272, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 196, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 211, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 452, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Branch', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 55, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 213, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 370, '[]');

  -- run 8: death, 865s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_008.json', 85, 950, 865, 'death', '2026-03-31T18:51:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 413, 233, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 528, 108, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 713, 113, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 790, 70, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 650, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 553, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 148, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 424, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 620, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 224, '[]');

  -- run 9: death, 714s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_009.json', 68, 782, 714, 'death', '2026-03-31T20:59:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 400, 220, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 649, 229, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 724, 124, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 203, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 116, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 320, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 557, '[]');

  -- run 10: death, 889s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_010.json', 42, 931, 889, 'death', '2026-03-31T22:39:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 248, 68, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 657, 237, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 705, 105, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 829, 109, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 483, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 657, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 546, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 466, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 591, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 486, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 192, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 516, '[]');

  -- run 11: death, 548s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_011.json', 98, 646, 548, 'death', '2026-03-31T22:17:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 358, 178, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 574, 154, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 309, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 338, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 142, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 279, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 142, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 169, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 253, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 278, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 44, '[]');

  -- run 12: death, 775s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_012.json', 86, 861, 775, 'death', '2026-03-31T17:42:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 328, 148, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 650, 230, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 687, 87, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 803, 83, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 50, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 186, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 274, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 159, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 514, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 147, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 607, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 253, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol Concert', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 506, '[]');

  -- run 13: death, 662s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_013.json', 95, 757, 662, 'death', '2026-03-31T21:38:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 408, 228, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 494, 74, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 679, 79, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 392, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 108, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 252, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 120, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 405, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 297, '[]');

  -- run 14: death, 544s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_014.json', 78, 622, 544, 'death', '2026-03-31T22:17:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 362, 182, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 549, 129, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 435, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 38, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 77, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 180, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 143, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 237, '[]');

  -- run 15: death, 686s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_015.json', 70, 756, 686, 'death', '2026-03-31T17:30:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 360, 180, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 553, 133, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 684, 84, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 135, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 42, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 321, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 510, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 520, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'GWS', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 481, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Thunder', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 378, '[]');

  -- run 16: death, 633s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_016.json', 61, 694, 633, 'death', '2026-03-31T12:16:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 254, 74, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 481, 61, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 655, 55, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 382, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 334, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 270, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 179, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 46, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 148, '[]');

  -- run 17: win, 699s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_017.json', 87, 786, 699, 'win', '2026-03-31T19:04:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 284, 104, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 532, 112, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 805, 205, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 305, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 71, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 209, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 511, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 482, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 314, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 215, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 476, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'GWS', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 533, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 123, '[]');

  -- run 18: death, 805s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_018.json', 42, 847, 805, 'death', '2026-03-31T17:26:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 307, 127, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 636, 216, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 671, 71, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 770, 50, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 192, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 348, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 594, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 44, '[]');

  -- run 19: death, 674s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_019.json', 15, 689, 674, 'death', '2026-03-31T13:29:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 390, 210, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 504, 84, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 693, 93, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 475, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 126, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 532, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 387, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 152, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 322, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 226, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 149, '[]');

  -- run 20: win, 704s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_020.json', 40, 744, 704, 'win', '2026-03-31T18:02:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 415, 235, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 586, 166, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 825, 225, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 529, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 66, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 100, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 270, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 324, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 262, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 122, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 474, '[]');

  -- run 21: death, 900s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_021.json', 21, 921, 900, 'death', '2026-03-31T13:28:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 402, 222, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 541, 121, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 652, 52, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 761, 41, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 423, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 524, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 139, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 272, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 420, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 615, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 397, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 618, '[]');

  -- run 22: death, 695s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_022.json', 2, 697, 695, 'death', '2026-03-31T23:18:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 393, 213, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 566, 146, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 665, 65, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 404, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 543, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 309, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 198, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 293, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol Concert', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 523, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 332, '[]');

  -- run 23: death, 717s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_023.json', 18, 735, 717, 'death', '2026-03-31T19:04:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 418, 238, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 522, 102, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 740, 140, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 148, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 129, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 270, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 168, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 427, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 494, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 409, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'RPG', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 459, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Elite Lava Bucket', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 188, '[]');

  -- run 24: death, 621s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_024.json', 12, 633, 621, 'death', '2026-03-31T18:41:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 350, 170, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 622, 202, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 643, 43, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 426, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 105, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 433, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 491, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 335, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 31, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 171, '[]');

  -- run 25: death, 758s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_025.json', 22, 780, 758, 'death', '2026-03-31T20:16:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 253, 73, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 634, 214, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 651, 51, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 758, 38, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 97, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 159, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 311, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 591, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 362, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Tako''s Breath', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 420, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 573, '[]');

  -- run 26: death, 673s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_026.json', 77, 750, 673, 'death', '2026-03-31T19:32:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 335, 155, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 490, 70, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 644, 44, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 474, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 498, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 399, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 56, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 525, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 177, '[]');

  -- run 27: win, 638s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_027.json', 19, 657, 638, 'win', '2026-03-31T20:13:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 393, 213, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 588, 168, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 702, 102, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 110, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 219, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 189, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 399, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 196, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Tako''s Breath', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 427, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 320, '[]');

  -- run 28: death, 660s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_028.json', 6, 666, 660, 'death', '2026-03-31T13:56:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 264, 84, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 505, 85, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 678, 78, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 442, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 211, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 66, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 233, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 471, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 409, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 51, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 253, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 39, '[]');

  -- run 29: death, 677s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_029.json', 73, 750, 677, 'death', '2026-03-31T21:27:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 328, 148, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 646, 226, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 683, 83, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 107, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 211, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 535, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 505, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 489, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 360, '[]');

  -- run 30: death, 730s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_030.json', 19, 749, 730, 'death', '2026-03-31T22:20:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 337, 157, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 482, 62, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 765, 165, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 757, 37, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 70, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 283, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 478, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 479, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 566, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 565, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 192, '[]');

  -- run 31: death, 561s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_031.json', 43, 604, 561, 'death', '2026-03-31T18:49:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 398, 218, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 463, 43, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 215, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 39, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 215, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 188, '[]');

  -- run 32: death, 902s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_032.json', 98, 1000, 902, 'death', '2026-03-31T15:21:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 349, 169, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 514, 94, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 702, 102, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 800, 80, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 930, 30, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 614, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 481, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 502, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 318, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 421, '[]');

  -- run 33: death, 607s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_033.json', 77, 684, 607, 'death', '2026-03-31T15:51:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 260, 80, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 529, 109, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 630, 30, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 386, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 455, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 101, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 418, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 391, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 270, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 259, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 345, '[]');

  -- run 34: death, 717s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_034.json', 32, 749, 717, 'death', '2026-03-31T13:01:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 280, 100, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 503, 83, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 700, 100, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 74, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 426, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 412, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 289, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 46, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Elite Lava Bucket', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 395, '[]');

  -- run 35: death, 743s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_035.json', 13, 756, 743, 'death', '2026-03-31T22:40:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 373, 193, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 653, 233, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 838, 238, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 760, 40, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 439, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 480, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 425, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 377, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 221, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 538, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 539, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Thunder', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 406, '[]');

  -- run 36: death, 623s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_036.json', 10, 633, 623, 'death', '2026-03-31T20:17:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 411, 231, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 573, 153, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 632, 32, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 360, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 217, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 215, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 226, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 320, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 46, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 339, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'GWS', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 108, '[]');

  -- run 37: death, 606s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_037.json', 9, 615, 606, 'death', '2026-03-31T19:48:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 371, 191, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 500, 80, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 634, 34, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 393, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 388, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 356, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 119, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 436, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 215, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 290, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 144, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 92, '[]');

  -- run 38: death, 482s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_038.json', 63, 545, 482, 'death', '2026-03-31T14:15:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 231, 51, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 473, 53, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 145, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 207, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 295, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 175, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 70, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Thunder', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 158, '[]');

  -- run 39: death, 809s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_039.json', 38, 847, 809, 'death', '2026-03-31T14:40:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 382, 202, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 601, 181, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 668, 68, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 814, 94, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 378, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 177, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 350, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 356, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 383, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 438, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 163, '[]');

  -- run 40: death, 975s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_040.json', 71, 1046, 975, 'death', '2026-03-31T17:32:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 252, 72, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 546, 126, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 706, 106, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 884, 164, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 945, 45, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 471, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 167, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 300, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 774, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 404, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 442, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 404, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 76, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 440, '[]');

  -- run 41: death, 851s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_041.json', 9, 860, 851, 'death', '2026-03-31T20:18:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 323, 143, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 594, 174, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 760, 160, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 821, 101, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 516, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 47, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 407, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 369, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 142, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Branch', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 456, '[]');

  -- run 42: death, 763s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_042.json', 92, 855, 763, 'death', '2026-03-31T16:50:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 401, 221, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 626, 206, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 651, 51, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 788, 68, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 568, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 415, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 351, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Branch', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 207, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 500, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 575, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Thunder', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 380, '[]');

  -- run 43: death, 564s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_043.json', 82, 646, 564, 'death', '2026-03-31T23:43:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 402, 222, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 517, 97, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 305, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 168, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 317, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 386, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 169, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 100, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 85, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 345, '[]');

  -- run 44: win, 673s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_044.json', 31, 704, 673, 'win', '2026-03-31T21:15:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 237, 57, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 636, 216, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 780, 180, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 433, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 283, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 125, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 509, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 413, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 522, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 476, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Elite Lava Bucket', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 504, '[]');

  -- run 45: death, 712s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_045.json', 13, 725, 712, 'death', '2026-03-31T21:38:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 310, 130, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 537, 117, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 688, 88, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 367, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 339, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 41, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 297, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 398, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 271, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 93, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 151, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 506, '[]');

  -- run 46: death, 748s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_046.json', 87, 835, 748, 'death', '2026-03-31T14:25:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 353, 173, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 645, 225, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 724, 124, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 772, 52, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 149, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 240, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 530, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 123, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 552, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 486, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 86, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 494, '[]');

  -- run 47: win, 795s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_047.json', 59, 854, 795, 'win', '2026-03-31T12:35:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 397, 217, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 543, 123, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 830, 230, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 770, 50, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 571, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 160, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 258, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 33, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 53, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 335, '[]');

  -- run 48: death, 802s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_048.json', 69, 871, 802, 'death', '2026-03-31T22:46:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 333, 153, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 601, 181, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 742, 142, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 779, 59, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 358, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 272, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 339, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 176, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 563, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 256, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 453, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Branch', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 337, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 311, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 92, '[]');

  -- run 49: death, 584s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_049.json', 86, 670, 584, 'death', '2026-03-31T14:40:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 334, 154, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 592, 172, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 338, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 50, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 67, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 126, '[]');

  -- run 50: death, 669s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_050.json', 85, 754, 669, 'death', '2026-03-31T21:46:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 368, 188, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 520, 100, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 691, 91, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 409, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 104, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 38, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 163, '[]');

  -- run 51: win, 584s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_051.json', 19, 603, 584, 'win', '2026-03-31T18:18:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 275, 95, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 549, 129, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 263, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 221, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 60, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 346, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 356, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 371, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'GWS', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 345, '[]');

  -- run 52: death, 672s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_052.json', 62, 734, 672, 'death', '2026-03-31T19:42:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 229, 49, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 466, 46, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 698, 98, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 312, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 273, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 58, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 356, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 34, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 389, '[]');

  -- run 53: death, 780s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_053.json', 42, 822, 780, 'death', '2026-03-31T22:50:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 369, 189, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 489, 69, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 780, 180, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 808, 88, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 384, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 84, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 130, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 477, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 267, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 107, '[]');

  -- run 54: death, 831s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_054.json', 78, 909, 831, 'death', '2026-03-31T21:49:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 377, 197, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 566, 146, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 728, 128, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 753, 33, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 597, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 125, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 641, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 642, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 362, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 424, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 42, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'GWS', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 328, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 453, '[]');

  -- run 55: death, 685s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_055.json', 31, 716, 685, 'death', '2026-03-31T20:57:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 371, 191, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 598, 178, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 651, 51, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 211, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 437, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 356, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 328, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 532, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 278, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 363, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 415, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 317, '[]');

  -- run 56: death, 749s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_056.json', 14, 763, 749, 'death', '2026-03-31T18:23:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 369, 189, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 516, 96, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 796, 196, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 767, 47, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 117, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 141, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 284, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 393, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 200, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Thunder', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 75, '[]');

  -- run 57: death, 576s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_057.json', 42, 618, 576, 'death', '2026-03-31T18:48:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 335, 155, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 477, 57, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 205, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 361, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 85, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 376, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Branch', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 439, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 114, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 409, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 154, '[]');

  -- run 58: win, 659s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_058.json', 20, 679, 659, 'win', '2026-03-31T14:35:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 309, 129, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 608, 188, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 754, 154, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 346, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 118, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 446, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 374, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 391, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 439, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Tako''s Breath', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 182, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 500, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 344, '[]');

  -- run 59: death, 583s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_059.json', 39, 622, 583, 'death', '2026-03-31T13:15:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 254, 74, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 496, 76, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 142, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 179, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 46, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 170, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 145, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 305, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 177, '[]');

  -- run 60: death, 633s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.1_run_060.json', 95, 728, 633, 'death', '2026-03-31T19:36:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 351, 171, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 605, 185, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 662, 62, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 135, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 442, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 60, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 321, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 209, '[]');

END $$;

DO $$
DECLARE
  v_game_id INT;
  v_version_id INT;
  v_run_id INT;
  v_item_id INT;
  v_boss_id INT;
BEGIN
  SELECT id INTO v_game_id FROM dashboard.games WHERE name='Holocure' AND user_id=1 LIMIT 1;
  SELECT id INTO v_version_id FROM dashboard.game_versions WHERE game_id=v_game_id AND name='V0.2' LIMIT 1;

  -- run 1: death, 894s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_001.json', 62, 956, 894, 'death', '2026-03-24T14:36:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 312, 132, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 639, 219, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 840, 240, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 762, 42, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 153, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 462, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 165, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 519, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 102, '[]');

  -- run 2: death, 739s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_002.json', 93, 832, 739, 'death', '2026-03-13T18:05:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 251, 71, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 491, 71, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 725, 125, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 761, 41, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 338, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 240, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 560, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 551, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 223, '[]');

  -- run 3: win, 838s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_003.json', 49, 887, 838, 'win', '2026-03-07T15:31:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 253, 73, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 526, 106, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 772, 172, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 928, 208, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 97, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 347, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 477, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Branch', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 274, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 89, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 280, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 118, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'GWS', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 474, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 148, '[]');

  -- run 4: death, 734s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_004.json', 7, 741, 734, 'death', '2026-03-20T21:38:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 304, 124, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 635, 215, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 834, 234, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 760, 40, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 42, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 280, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 77, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 513, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 402, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Tako''s Breath', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 422, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 183, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 213, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 66, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 459, '[]');

  -- run 5: death, 855s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_005.json', 34, 889, 855, 'death', '2026-03-31T23:20:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 244, 64, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 611, 191, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 740, 140, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 780, 60, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 344, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 136, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 240, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 572, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 273, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 459, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Thunder', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 533, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 91, '[]');

  -- run 6: death, 933s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_006.json', 11, 944, 933, 'death', '2026-03-10T23:17:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 236, 56, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 524, 104, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 778, 178, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 870, 150, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 953, 53, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 743, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 647, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 71, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 250, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 365, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 181, '[]');

  -- run 7: death, 919s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_007.json', 6, 925, 919, 'death', '2026-03-17T18:32:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 398, 218, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 541, 121, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 711, 111, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 809, 89, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 930, 30, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 466, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 206, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 392, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 249, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 221, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 311, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 316, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 486, '[]');

  -- run 8: death, 891s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_008.json', 78, 969, 891, 'death', '2026-03-31T14:02:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 383, 203, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 622, 202, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 707, 107, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 825, 105, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 87, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 254, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 619, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 427, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 393, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 207, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol Concert', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 210, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 272, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 636, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 355, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 397, '[]');

  -- run 9: death, 865s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_009.json', 72, 937, 865, 'death', '2026-03-28T17:36:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 261, 81, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 609, 189, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 693, 93, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 876, 156, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 605, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 658, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 395, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 258, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 633, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 209, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 417, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 344, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 331, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 160, '[]');

  -- run 10: win, 824s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_010.json', 89, 913, 824, 'win', '2026-03-11T23:00:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 371, 191, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 565, 145, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 790, 190, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 773, 53, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 260, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 196, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 492, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 105, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 64, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 331, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 50, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 356, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 297, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 137, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 105, '[]');

  -- run 11: win, 696s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_011.json', 70, 766, 696, 'win', '2026-03-11T23:04:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 248, 68, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 551, 131, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 797, 197, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 301, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 123, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 426, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 219, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 409, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 420, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 408, '[]');

  -- run 12: win, 1060s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_012.json', 33, 1093, 1060, 'win', '2026-03-11T19:02:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 337, 157, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 584, 164, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 712, 112, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 822, 102, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 1013, 113, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 729, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 475, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 236, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 672, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 181, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol Concert', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 639, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 294, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 357, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 774, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 98, '[]');

  -- run 13: win, 977s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_013.json', 35, 1012, 977, 'win', '2026-03-23T17:54:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 358, 178, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 582, 162, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 648, 48, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 944, 224, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 1104, 204, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 742, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 731, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 130, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 263, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 167, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 337, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Scythe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 483, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 358, '[]');

  -- run 14: death, 923s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_014.json', 77, 1000, 923, 'death', '2026-03-14T23:26:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 390, 210, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 487, 67, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 693, 93, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 878, 158, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 936, 36, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 294, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 735, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 122, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 321, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 58, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 425, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 80, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 196, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 615, '[]');

  -- run 15: death, 856s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_015.json', 86, 942, 856, 'death', '2026-03-13T22:14:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 277, 97, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 533, 113, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 811, 211, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 854, 134, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 313, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 297, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 335, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 222, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 183, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 559, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 276, '[]');

  -- run 16: win, 754s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_016.json', 76, 830, 754, 'win', '2026-03-31T12:51:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 328, 148, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 634, 214, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 726, 126, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 939, 219, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 369, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 410, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 296, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 331, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 504, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 150, '[]');

  -- run 17: death, 591s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_017.json', 9, 600, 591, 'death', '2026-03-24T21:39:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 345, 165, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 499, 79, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 221, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 37, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 194, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 417, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 456, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'MiComet', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 301, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 127, '[]');

  -- run 18: death, 886s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_018.json', 80, 966, 886, 'death', '2026-03-06T22:00:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 412, 232, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 528, 108, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 702, 102, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 838, 118, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 140, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 699, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 50, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 79, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 148, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 456, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 484, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'MiComet', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 419, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 156, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 590, '[]');

  -- run 19: death, 836s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_019.json', 80, 916, 836, 'death', '2026-03-12T23:43:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 227, 47, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 542, 122, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 752, 152, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 834, 114, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 497, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 333, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 300, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 505, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 81, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'GWS', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 132, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 208, '[]');

  -- run 20: win, 1035s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_020.json', 37, 1072, 1035, 'win', '2026-03-31T12:54:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 317, 137, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 638, 218, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 725, 125, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 874, 154, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 1134, 234, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 730, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 774, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 163, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 524, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 529, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Branch', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 145, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 451, '[]');

  -- run 21: death, 797s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_021.json', 90, 887, 797, 'death', '2026-03-26T17:00:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 398, 218, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 599, 179, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 835, 235, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 823, 103, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 540, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 487, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 353, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 151, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 539, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 48, '[]');

  -- run 22: win, 856s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_022.json', 4, 860, 856, 'win', '2026-03-09T18:26:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 367, 187, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 605, 185, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 645, 45, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 788, 68, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 432, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'MiComet', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 250, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 498, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 225, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 376, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 140, '[]');

  -- run 23: death, 841s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_023.json', 15, 856, 841, 'death', '2026-03-26T22:37:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 340, 160, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 582, 162, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 813, 213, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 825, 105, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 596, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 252, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 235, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 244, '[]');

  -- run 24: death, 801s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_024.json', 87, 888, 801, 'death', '2026-03-25T23:18:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 313, 133, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 637, 217, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 678, 78, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 772, 52, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 604, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 428, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 273, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 553, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 324, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 78, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Tako''s Breath', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 420, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 426, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 456, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'RPG', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 350, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 595, '[]');

  -- run 25: death, 1036s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_025.json', 99, 1135, 1036, 'death', '2026-03-12T21:55:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 338, 158, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 522, 102, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 807, 207, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 945, 225, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 1024, 124, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 547, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 193, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 455, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 586, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 579, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 443, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 124, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 387, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 256, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 248, '[]');

  -- run 26: death, 908s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_026.json', 46, 954, 908, 'death', '2026-03-27T17:21:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 300, 120, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 520, 100, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 837, 237, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 923, 203, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 938, 38, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 248, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 464, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 136, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 240, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 580, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 475, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 692, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 103, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 187, '[]');

  -- run 27: win, 1037s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_027.json', 6, 1043, 1037, 'win', '2026-03-07T13:04:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 270, 90, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 529, 109, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 787, 187, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 783, 63, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 1092, 192, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 776, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 693, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 412, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 41, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 596, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 598, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 174, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 156, '[]');

  -- run 28: win, 869s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_028.json', 76, 945, 869, 'win', '2026-03-07T12:36:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 325, 145, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 602, 182, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 785, 185, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 786, 66, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 628, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 186, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 609, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 610, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 403, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 353, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 468, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 202, '[]');

  -- run 29: win, 934s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_029.json', 82, 1016, 934, 'win', '2026-03-13T23:47:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 336, 156, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 535, 115, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 737, 137, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 799, 79, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 1118, 218, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 552, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 122, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 436, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 402, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 544, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Tako''s Breath', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 40, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 708, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 274, '[]');

  -- run 30: win, 842s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_030.json', 21, 863, 842, 'win', '2026-03-31T21:43:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 418, 238, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 596, 176, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 680, 80, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 879, 159, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 258, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 558, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 445, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 391, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 383, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 528, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 523, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 648, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 36, '[]');

  -- run 31: death, 771s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_031.json', 65, 836, 771, 'death', '2026-03-11T15:04:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 389, 209, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 536, 116, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 699, 99, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 759, 39, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 450, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 176, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 216, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 553, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 300, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 201, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 380, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Branch', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 494, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 100, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'MiComet', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 265, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Thunder', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 417, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 424, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 165, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 141, '[]');

  -- run 32: death, 839s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_032.json', 64, 903, 839, 'death', '2026-03-06T15:33:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 400, 220, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 572, 152, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 683, 83, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 827, 107, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 223, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 35, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 37, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 621, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 565, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 150, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 562, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 37, '[]');

  -- run 33: death, 779s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_033.json', 46, 825, 779, 'death', '2026-03-31T17:26:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 252, 72, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 504, 84, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 754, 154, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 791, 71, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 512, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 475, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 120, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 95, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 299, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 257, '[]');

  -- run 34: death, 900s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_034.json', 4, 904, 900, 'death', '2026-03-17T13:38:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 410, 230, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 563, 143, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 747, 147, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 883, 163, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 261, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 650, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 229, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 495, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 443, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 618, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 34, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Thunder', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 297, '[]');

  -- run 35: death, 907s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_035.json', 37, 944, 907, 'death', '2026-03-23T13:32:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 312, 132, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 494, 74, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 776, 176, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 830, 110, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 934, 34, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 540, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 234, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 270, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'MiComet', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 277, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Thunder', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 301, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 241, '[]');

  -- run 36: death, 973s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_036.json', 31, 1004, 973, 'death', '2026-03-13T15:53:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 383, 203, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 556, 136, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 684, 84, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 792, 72, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 945, 45, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 363, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 400, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 657, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 295, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'MiComet', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 221, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 591, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 299, '[]');

  -- run 37: death, 851s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_037.json', 23, 874, 851, 'death', '2026-03-25T17:23:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 352, 172, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 573, 153, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 698, 98, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 857, 137, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 372, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 296, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 399, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 398, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 603, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 600, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 293, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 382, '[]');

  -- run 38: win, 981s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_038.json', 44, 1025, 981, 'win', '2026-03-06T18:07:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 398, 218, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 620, 200, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 695, 95, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 912, 192, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 1092, 192, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 158, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 775, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 270, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 707, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 287, '[]');

  -- run 39: win, 1022s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_039.json', 50, 1072, 1022, 'win', '2026-03-08T15:10:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 251, 71, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 489, 69, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 661, 61, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 886, 166, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 1085, 185, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 523, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 582, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 262, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 121, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 482, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Tako''s Breath', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 345, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 404, '[]');

  -- run 40: death, 768s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_040.json', 38, 806, 768, 'death', '2026-03-19T16:09:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 227, 47, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 465, 45, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 778, 178, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 798, 78, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 496, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 493, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 308, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 533, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 222, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 358, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 49, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 185, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 80, '[]');

  -- run 41: death, 866s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_041.json', 16, 882, 866, 'death', '2026-03-09T16:17:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 408, 228, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 577, 157, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 831, 231, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 752, 32, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 360, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 192, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 674, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 73, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 592, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 563, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Scythe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 155, '[]');

  -- run 42: death, 834s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_042.json', 38, 872, 834, 'death', '2026-03-14T19:40:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 275, 95, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 557, 137, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 658, 58, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 860, 140, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 416, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 396, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 167, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 286, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 70, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 526, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 478, '[]');

  -- run 43: death, 693s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_043.json', 23, 716, 693, 'death', '2026-03-21T15:36:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 364, 184, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 554, 134, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 671, 71, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 236, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 139, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 527, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 226, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 131, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 62, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 427, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 531, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 292, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 80, '[]');

  -- run 44: win, 782s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_044.json', 91, 873, 782, 'win', '2026-03-17T15:38:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 243, 63, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 468, 48, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 698, 98, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 932, 212, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 214, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 593, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 441, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 546, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 375, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Branch', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 191, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 576, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 483, '[]');

  -- run 45: death, 1088s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_045.json', 47, 1135, 1088, 'death', '2026-03-23T17:54:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 330, 150, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 660, 240, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 807, 207, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 875, 155, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 1130, 230, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Mumei''s Flock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 1080, 1115, 35, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 439, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 612, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 252, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 781, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'BL Book', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 109, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Scythe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 770, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 362, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 841, '[]');

  -- run 46: win, 729s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_046.json', 88, 817, 729, 'win', '2026-03-21T14:31:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 404, 224, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 612, 192, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 669, 69, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 917, 197, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 517, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 80, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 438, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Psycho Axe', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 298, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 117, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 438, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 543, '[]');

  -- run 47: win, 661s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_047.json', 71, 732, 661, 'win', '2026-03-30T14:30:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 275, 95, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 589, 169, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 746, 146, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 37, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 205, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 385, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 441, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 86, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Tako''s Breath', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 186, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 260, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 271, '[]');

  -- run 48: death, 917s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_048.json', 94, 1011, 917, 'death', '2026-03-06T17:13:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 362, 182, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 519, 99, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 697, 97, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 904, 184, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 946, 46, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 604, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 52, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 70, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 283, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 572, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 342, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 371, '[]');

  -- run 49: death, 632s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_049.json', 45, 677, 632, 'death', '2026-03-06T17:43:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 315, 135, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 625, 205, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 649, 49, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 99, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 250, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'CEO''s Tears', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 495, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 424, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 104, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 365, '[]');

  -- run 50: death, 853s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_050.json', 20, 873, 853, 'death', '2026-03-31T21:58:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 306, 126, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 482, 62, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 787, 187, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 780, 60, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 448, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 45, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 306, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 452, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 257, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 237, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Tako''s Breath', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 239, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 97, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 211, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 479, '[]');

  -- run 51: death, 938s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_051.json', 31, 969, 938, 'death', '2026-03-16T19:56:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 334, 154, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 568, 148, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 655, 55, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 920, 200, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 954, 54, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 629, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 742, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 589, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 168, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 432, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Tako''s Breath', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 453, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 361, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 491, '[]');

  -- run 52: death, 731s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_052.json', 75, 806, 731, 'death', '2026-03-06T14:29:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 407, 227, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 516, 96, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 761, 161, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 751, 31, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 499, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 539, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 540, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 35, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 535, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 273, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 252, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 517, '[]');

  -- run 53: win, 709s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_053.json', 31, 740, 709, 'win', '2026-03-25T21:00:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 365, 185, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 536, 116, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 803, 203, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 367, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 453, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 567, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 244, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 244, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 89, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 141, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 119, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 525, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 324, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 467, '[]');

  -- run 54: death, 878s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_054.json', 67, 945, 878, 'death', '2026-03-25T17:16:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 372, 192, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 624, 204, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 735, 135, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 783, 63, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 376, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 520, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 515, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 66, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 691, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 169, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 523, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 189, '[]');

  -- run 55: win, 654s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_055.json', 100, 754, 654, 'win', '2026-03-23T17:29:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 330, 150, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 655, 235, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 659, 59, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 497, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 77, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 257, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 289, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bait', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 42, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 439, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 277, '[]');

  -- run 56: win, 649s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_056.json', 24, 673, 649, 'win', '2026-03-31T20:09:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 312, 132, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 588, 168, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 832, 232, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 348, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 316, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 293, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 82, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 135, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 307, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 209, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ninja Headband', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 311, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 193, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 168, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Ancient Sword', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 102, '[]');

  -- run 57: death, 1053s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_057.json', 9, 1062, 1053, 'death', '2026-03-24T15:10:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 304, 124, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 621, 201, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 731, 131, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 903, 183, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 1043, 143, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 217, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 746, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Plug Type Asacoco', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 711, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 368, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Hope Shard', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 837, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bounce Ball', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 362, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'GWS', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 494, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 529, '[]');

  -- run 58: death, 760s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_058.json', 14, 774, 760, 'death', '2026-03-06T13:30:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 314, 134, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 587, 167, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 763, 163, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 760, 40, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 345, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 445, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 101, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'EN''s Curse', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 141, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Face Mask', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 418, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 365, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 536, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 92, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Snake', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 122, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 252, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Anchor', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 172, '[]');

  -- run 59: win, 1046s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_059.json', 18, 1064, 1046, 'win', '2026-03-26T12:48:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 394, 214, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 503, 83, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 760, 160, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Kronii''s Clock', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 720, 901, 181, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'IRyS''s Hope', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 900, 1019, 119, FALSE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Halu', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 516, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 172, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Spider Cooking', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 222, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 598, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Eldritch Horror', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 423, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Gorilla Paw', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 398, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Cutting Board', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 457, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Branch', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 187, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Knightly Milk', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 325, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Trident', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 716, '[]');

  -- run 60: death, 662s
  INSERT INTO dashboard.runs (version_id, video_filename, start_time, end_time, duration_seconds, outcome, recorded_at)
    VALUES (v_version_id, 'holocure_V0.2_run_060.json', 9, 671, 662, 'death', '2026-03-10T14:00:00Z')
    RETURNING id INTO v_run_id;
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Aqua Bot', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 180, 390, 210, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Bae''s Army', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 420, 469, 49, FALSE);
  INSERT INTO dashboard.bosses (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Fauna''s Seal', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_boss_id;
  INSERT INTO dashboard.boss_encounters (run_id, boss_id, start_time, end_time, duration_seconds, player_died)
    VALUES (v_run_id, v_boss_id, 600, 639, 39, TRUE);
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Body Pillow', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 414, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Headphones', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 428, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Idol''s Blue Light', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 173, '[]');
  INSERT INTO dashboard.items (game_id, name, first_seen_version_id)
    VALUES (v_game_id, 'Breastplate', v_version_id)
    ON CONFLICT (game_id, name) DO UPDATE SET name=EXCLUDED.name
    RETURNING id INTO v_item_id;
  INSERT INTO dashboard.item_pickups (run_id, item_id, picked_at_seconds, options)
    VALUES (v_run_id, v_item_id, 268, '[]');

END $$;

COMMIT;
