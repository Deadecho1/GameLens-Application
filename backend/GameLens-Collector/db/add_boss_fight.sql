CREATE TABLE boss_fight (
    id SERIAL PRIMARY KEY,
    run_id INT NOT NULL,
    boss_name VARCHAR(200),
    duration FLOAT NOT NULL,
    player_died BOOLEAN NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    CONSTRAINT boss_fight_run_id FOREIGN KEY (run_id) REFERENCES run(id)
);
