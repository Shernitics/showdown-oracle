import sqlite3
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_PATH = RESULTS_DIR / "results.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    seed            INTEGER NOT NULL,
    commit_sha      TEXT,
    feature_version INTEGER,
    battle_format   TEXT,
    team_id         TEXT,
    hyperparams     TEXT,
    device          TEXT,
    wall_clock_s    REAL
);

CREATE TABLE IF NOT EXISTS battles (
    run_id          TEXT NOT NULL,
    train_steps     INTEGER NOT NULL,
    train_episodes  INTEGER NOT NULL,
    opponent        TEXT NOT NULL,
    deterministic   INTEGER NOT NULL,
    eval_seed       INTEGER NOT NULL,
    battle_idx      INTEGER NOT NULL,
    won             INTEGER NOT NULL,
    turns           INTEGER,
    reward          REAL,
    end_reason      TEXT,
    used_tera       INTEGER,
    mons_left       INTEGER,
    mons_left_opp   INTEGER,
    PRIMARY KEY (run_id, train_steps, opponent, deterministic, battle_idx)
);

CREATE INDEX IF NOT EXISTS ix_curve ON battles(opponent, train_steps);
"""

RUN_COLUMNS = 9
BATTLE_COLUMNS = 14


def connect(path: Path = RESULTS_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def add_run(con: sqlite3.Connection, row: tuple):
    con.execute(f"INSERT OR REPLACE INTO runs VALUES ({','.join('?' * RUN_COLUMNS)})", row)
    con.commit()


def add_battles(con: sqlite3.Connection, rows: list):
    con.executemany(f"INSERT OR IGNORE INTO battles VALUES ({','.join('?' * BATTLE_COLUMNS)})", rows)
    con.commit()


def done_cells(con: sqlite3.Connection) -> dict:
    cur = con.execute(
        "SELECT run_id, train_steps, opponent, deterministic, COUNT(*) FROM battles GROUP BY 1, 2, 3, 4"
    )
    return {(r[0], r[1], r[2], r[3]): r[4] for r in cur}
