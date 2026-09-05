import json
import sqlite3
from contextlib import contextmanager
from typing import Iterable

from config import DATA_DIR, DB_PATH


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS matches (
                fixture_id INTEGER PRIMARY KEY,
                season INTEGER NOT NULL,
                round TEXT,
                kickoff TEXT,
                status TEXT,
                venue TEXT,
                referee TEXT,
                home_team_id INTEGER,
                home_team TEXT,
                away_team_id INTEGER,
                away_team TEXT,
                home_goals INTEGER,
                away_goals INTEGER,
                raw_json TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS team_match_stats (
                fixture_id INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                team_name TEXT,
                corners REAL,
                yellow_cards REAL,
                red_cards REAL,
                fouls REAL,
                shots_total REAL,
                shots_on_goal REAL,
                possession REAL,
                offsides REAL,
                raw_json TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (fixture_id, team_id)
            );

            CREATE TABLE IF NOT EXISTS referees (
                referee_name TEXT PRIMARY KEY,
                matches INTEGER DEFAULT 0,
                yellow_cards REAL DEFAULT 0,
                red_cards REAL DEFAULT 0,
                fouls REAL DEFAULT 0,
                penalties REAL DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS referee_matches (
                fixture_id INTEGER PRIMARY KEY,
                referee_name TEXT,
                season INTEGER,
                kickoff TEXT,
                yellow_cards REAL,
                red_cards REAL,
                fouls REAL,
                penalties REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sync_type TEXT NOT NULL,
                season INTEGER,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_matches_season
                ON matches(season);

            CREATE INDEX IF NOT EXISTS idx_matches_round
                ON matches(round);

            CREATE INDEX IF NOT EXISTS idx_matches_referee
                ON matches(referee);
            """
        )


def upsert_matches(fixtures: Iterable[dict]):
    rows = []

    for item in fixtures:
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})

        rows.append(
            (
                fixture.get("id"),
                league.get("season"),
                league.get("round"),
                fixture.get("date"),
                fixture.get("status", {}).get("short"),
                (fixture.get("venue") or {}).get("name"),
                fixture.get("referee"),
                (teams.get("home") or {}).get("id"),
                (teams.get("home") or {}).get("name"),
                (teams.get("away") or {}).get("id"),
                (teams.get("away") or {}).get("name"),
                goals.get("home"),
                goals.get("away"),
                json.dumps(item, ensure_ascii=False),
            )
        )

    if not rows:
        return 0

    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO matches (
                fixture_id, season, round, kickoff, status, venue, referee,
                home_team_id, home_team, away_team_id, away_team,
                home_goals, away_goals, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fixture_id) DO UPDATE SET
                season=excluded.season,
                round=excluded.round,
                kickoff=excluded.kickoff,
                status=excluded.status,
                venue=excluded.venue,
                referee=excluded.referee,
                home_team_id=excluded.home_team_id,
                home_team=excluded.home_team,
                away_team_id=excluded.away_team_id,
                away_team=excluded.away_team,
                home_goals=excluded.home_goals,
                away_goals=excluded.away_goals,
                raw_json=excluded.raw_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            rows,
        )

    return len(rows)


def log_sync(sync_type: str, season: int | None, details: str):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO sync_log (sync_type, season, details)
            VALUES (?, ?, ?)
            """,
            (sync_type, season, details),
        )


def get_database_summary() -> dict:
    init_db()

    with connect() as conn:
        matches = conn.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"]
        finished = conn.execute(
            "SELECT COUNT(*) AS n FROM matches WHERE status = 'FT'"
        ).fetchone()["n"]
        referees = conn.execute(
            """
            SELECT COUNT(DISTINCT referee) AS n
            FROM matches
            WHERE referee IS NOT NULL AND TRIM(referee) <> ''
            """
        ).fetchone()["n"]

    return {
        "matches": matches,
        "finished": finished,
        "referees": referees,
    }
