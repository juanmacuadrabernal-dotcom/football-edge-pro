from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "laliga.db"
RAW_DIR = BASE_DIR / "data" / "laliga_raw" / "fotmob"

LEAGUE_ID = 87
COUNTRY_CODE = "ESP"
LEAGUE_URL = "https://www.fotmob.com/api/data/leagues"

TIMEOUT = 30
RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.fotmob.com/",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


def normalize(value: Any) -> str:
    s = str(value or "").strip().lower()

    replacements = {
        "ø": "o",
        "å": "a",
        "æ": "ae",
    }

    for a, b in replacements.items():
        s = s.replace(a, b)

    s = unicodedata.normalize("NFKD", s)
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )
    s = re.sub(r"[^a-z0-9]+", "", s)

    return s


def stable_id(text: str) -> int:
    digest = hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:14]
    return int(digest, 16)


def current_season_start() -> int:
    now = datetime.now()
    return now.year if now.month >= 7 else now.year - 1


def make_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def get_json(session, params):
    last_error = None

    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(
                LEAGUE_URL,
                params=params,
                timeout=TIMEOUT,
            )

            if r.status_code == 200:
                return r.json()

            last_error = RuntimeError(
                f"HTTP {r.status_code}: {r.text[:200]}"
            )

        except Exception as exc:
            last_error = exc

        if attempt < RETRIES:
            time.sleep(2 * attempt)

    raise RuntimeError(
        str(last_error)
        if last_error
        else "Error descargando FotMob"
    )


def get_all_matches(payload):
    found = {}

    def walk(node):
        if isinstance(node, dict):
            home = node.get("home")
            away = node.get("away")
            status = node.get("status")
            mid = node.get("id")

            if (
                mid is not None
                and isinstance(home, dict)
                and isinstance(away, dict)
                and isinstance(status, dict)
                and home.get("name")
                and away.get("name")
            ):
                try:
                    found[int(mid)] = node
                except Exception:
                    pass

            for value in node.values():
                walk(value)

        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)

    return list(found.values())


def kickoff_utc(match):
    status = match.get("status", {}) or {}
    value = (
        status.get("utcTime")
        or match.get("utcTime")
    )

    if not value:
        return None

    ts = pd.to_datetime(
        value,
        utc=True,
        errors="coerce",
    )

    if pd.isna(ts):
        return None

    return ts


def is_cancelled(match):
    status = match.get("status", {}) or {}

    if status.get("cancelled"):
        return True

    reason = str(
        status.get("reason") or ""
    ).lower()

    return any(
        word in reason
        for word in (
            "cancel",
            "abandon",
            "postpon",
        )
    )


def discover_current_payload(session, season_start):
    candidates = [
        f"{season_start}/{season_start + 1}",
        f"{season_start}-{season_start + 1}",
        str(season_start),
        str(season_start + 1),
        None,
    ]

    now = pd.Timestamp.now(tz="UTC")
    best = None
    best_future = []

    for season_param in candidates:
        params = {
            "id": LEAGUE_ID,
            "ccode3": COUNTRY_CODE,
        }

        if season_param is not None:
            params["season"] = season_param

        label = (
            season_param
            if season_param is not None
            else "sin season"
        )

        print(
            f"    Probando FotMob season={label}...",
            end=" ",
            flush=True,
        )

        try:
            payload = get_json(
                session,
                params,
            )
        except Exception as exc:
            print(f"ERROR ({exc})")
            continue

        matches = get_all_matches(
            payload
        )

        future = []

        for match in matches:
            ko = kickoff_utc(match)

            if ko is None:
                continue

            status = match.get(
                "status",
                {},
            ) or {}

            if (
                ko >= now - pd.Timedelta(hours=12)
                and not status.get("finished")
                and not is_cancelled(match)
            ):
                future.append(match)

        print(
            f"{len(matches)} partidos | "
            f"{len(future)} futuros"
        )

        if len(future) > len(best_future):
            best = (
                payload,
                season_param,
            )
            best_future = future

        if len(future) >= 5:
            break

    if best is None or not best_future:
        raise RuntimeError(
            "FotMob respondió, pero no encontré próximos "
            "partidos de LaLiga."
        )

    return (
        best[0],
        best[1],
        best_future,
    )


def historical_team_names(conn):
    try:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT team
            FROM team_match_history
            WHERE team IS NOT NULL
            """,
            conn,
        )
    except Exception:
        return []

    return sorted(
        df["team"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def map_team_name(raw_name, historical_names):
    raw = str(raw_name).strip()
    key = normalize(raw)

    exact = {
        normalize(name): name
        for name in historical_names
    }

    if key in exact:
        return exact[key]

    aliases = {
        "atleticomadrid": [
            "Ath Madrid",
            "Atletico Madrid",
        ],
        "atleticodemadrid": [
            "Ath Madrid",
            "Atletico Madrid",
        ],
        "athleticclub": [
            "Ath Bilbao",
            "Athletic Bilbao",
        ],
        "athleticbilbao": [
            "Ath Bilbao",
            "Athletic Bilbao",
        ],
        "deportivoalaves": [
            "Alaves",
        ],
        "alaves": [
            "Alaves",
        ],
        "realbetis": [
            "Betis",
        ],
        "realbetisbalompie": [
            "Betis",
        ],
        "realsociedad": [
            "Sociedad",
        ],
        "rayovallecano": [
            "Vallecano",
            "Rayo Vallecano",
        ],
        "rcdceltadevigo": [
            "Celta",
        ],
        "celtadevigo": [
            "Celta",
        ],
        "celtavigo": [
            "Celta",
        ],
        "rcdespanyol": [
            "Espanol",
            "Espanyol",
        ],
        "espanyol": [
            "Espanol",
            "Espanyol",
        ],
        "deportivoacoruna": [
            "La Coruna",
            "Deportivo",
        ],
        "deportivolacoruna": [
            "La Coruna",
            "Deportivo",
        ],
        "racingsantander": [
            "Santander",
            "Racing Santander",
        ],
        "malaga": [
            "Malaga",
        ],
    }

    for candidate in aliases.get(
        key,
        [],
    ):
        if candidate in historical_names:
            return candidate

    # Los recién ascendidos pueden no existir en team_match_history.
    # OneHotEncoder(handle_unknown='ignore') soporta el nombre nuevo.
    return raw


def score_value(team):
    value = (
        team.get("score")
        if isinstance(team, dict)
        else None
    )

    try:
        return int(value)
    except Exception:
        return None


def fixture_row(match, season_start, historical_names):
    home = match.get("home", {}) or {}
    away = match.get("away", {}) or {}
    status = match.get("status", {}) or {}
    ko = kickoff_utc(match)

    if ko is None:
        return None

    home_raw = home.get("name")
    away_raw = away.get("name")

    home_name = map_team_name(
        home_raw,
        historical_names,
    )
    away_name = map_team_name(
        away_raw,
        historical_names,
    )

    round_value = (
        match.get("round")
        or match.get("roundName")
        or match.get("tournamentStage")
        or ""
    )

    if isinstance(round_value, dict):
        round_value = (
            round_value.get("name")
            or round_value.get("round")
            or str(round_value)
        )

    return {
        "match_id": int(match["id"]),
        "season": int(season_start),
        "round": str(round_value),
        "utc_time": ko.isoformat(),
        "date": ko.strftime("%Y-%m-%d"),
        "home_team_id": (
            home.get("id")
            or stable_id(
                "home|" + str(home_raw)
            )
        ),
        "home_team": home_name,
        "away_team_id": (
            away.get("id")
            or stable_id(
                "away|" + str(away_raw)
            )
        ),
        "away_team": away_name,
        "home_goals": score_value(home),
        "away_goals": score_value(away),
        "odds_home": None,
        "odds_draw": None,
        "odds_away": None,
        "finished": 1 if status.get("finished") else 0,
        "started": 1 if status.get("started") else 0,
        "cancelled": 1 if is_cancelled(match) else 0,
        "valid_for_model": 1,
        "status": (
            "Finished"
            if status.get("finished")
            else "Live"
            if status.get("started")
            else "Scheduled"
        ),
        "status_reason": (
            str(status.get("reason"))
            if status.get("reason") is not None
            else None
        ),
    }


def table_columns(conn, table):
    return [
        r[1]
        for r in conn.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    ]


def upsert_rows(conn, rows):
    if not rows:
        return 0

    cols = table_columns(
        conn,
        "fotmob_matches",
    )

    if not cols:
        raise RuntimeError(
            "No existe la tabla fotmob_matches."
        )

    usable = [
        c for c in cols
        if any(c in row for row in rows)
    ]

    if "match_id" not in usable:
        raise RuntimeError(
            "fotmob_matches no contiene match_id."
        )

    placeholders = ",".join(
        ["?"] * len(usable)
    )

    quoted_cols = ",".join(
        f'"{c}"'
        for c in usable
    )

    sql = (
        f'INSERT OR REPLACE INTO fotmob_matches '
        f'({quoted_cols}) VALUES ({placeholders})'
    )

    values = [
        [
            row.get(c)
            for c in usable
        ]
        for row in rows
    ]

    conn.executemany(
        sql,
        values,
    )

    return len(values)


def main():
    print()
    print("=" * 80)
    print("LALIGA EDGE PRO - ACTUALIZAR PROXIMA JORNADA V3")
    print("=" * 80)

    if not DB_PATH.exists():
        raise SystemExit(
            f"No existe {DB_PATH}"
        )

    season_start = current_season_start()

    print(
        f"Temporada detectada: "
        f"{season_start}/{season_start + 1}"
    )

    session = make_session()

    print()
    print("1/4 Buscando fixtures en FotMob...")

    (
        payload,
        season_param,
        future_matches,
    ) = discover_current_payload(
        session,
        season_start,
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        RAW_DIR
        / f"laliga_{season_start}_{season_start+1}.json"
    )

    raw_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("2/4 Normalizando nombres de equipos...")

    with sqlite3.connect(
        DB_PATH
    ) as conn:
        historical_names = historical_team_names(
            conn
        )

        rows = []

        for match in future_matches:
            row = fixture_row(
                match,
                season_start,
                historical_names,
            )

            if row is not None:
                rows.append(row)

        rows = sorted(
            rows,
            key=lambda x: x["utc_time"],
        )

        print(
            f"    Fixtures futuros detectados: "
            f"{len(rows)}"
        )

        if not rows:
            raise RuntimeError(
                "No quedó ningún fixture tras normalizar."
            )

        print()
        print("3/4 Guardando en laliga.db...")

        # TABLA DEDICADA PARA LA APP.
        # Evita mezclar histórico Football-Data con IDs/fixtures FotMob.
        upcoming_df = pd.DataFrame(rows)

        required_cols = [
            "match_id",
            "season",
            "round",
            "utc_time",
            "home_team",
            "away_team",
            "status",
            "valid_for_model",
        ]

        for col in required_cols:
            if col not in upcoming_df.columns:
                upcoming_df[col] = None

        if "referee" not in upcoming_df.columns:
            upcoming_df["referee"] = None

        upcoming_df = upcoming_df[
            [
                "match_id",
                "season",
                "round",
                "utc_time",
                "home_team",
                "away_team",
                "status",
                "referee",
                "valid_for_model",
            ]
        ].copy()

        upcoming_df.to_sql(
            "laliga_upcoming_fixtures",
            conn,
            if_exists="replace",
            index=False,
        )

        # También mantenemos sincronizada fotmob_matches para compatibilidad,
        # pero la app nueva NO dependerá de esta tabla para los fixtures.
        current_cols = table_columns(
            conn,
            "fotmob_matches",
        )

        if current_cols:
            if "finished" in current_cols:
                conn.execute(
                    """
                    DELETE FROM fotmob_matches
                    WHERE season = ?
                      AND COALESCE(finished, 0) = 0
                    """,
                    (season_start,),
                )
            elif "status" in current_cols:
                conn.execute(
                    """
                    DELETE FROM fotmob_matches
                    WHERE season = ?
                      AND LOWER(COALESCE(status, ''))
                          NOT LIKE '%finish%'
                    """,
                    (season_start,),
                )

            upsert_rows(
                conn,
                rows,
            )

        conn.commit()

        inserted = conn.execute(
            """
            SELECT COUNT(*)
            FROM laliga_upcoming_fixtures
            """
        ).fetchone()[0]

        if inserted <= 0:
            raise RuntimeError(
                "La descarga funcionó, pero la tabla "
                "laliga_upcoming_fixtures quedó vacía."
            )

        check = pd.read_sql_query(
            """
            SELECT match_id, season, round, utc_time,
                   home_team, away_team, status
            FROM laliga_upcoming_fixtures
            ORDER BY utc_time
            LIMIT 5
            """,
            conn,
        )

        print(
            f"    VERIFICADO EN DB: {inserted} fixtures"
        )

        for r in check.itertuples(index=False):
            print(
                f"      DB -> {r.utc_time} | "
                f"{r.home_team} - {r.away_team} | "
                f"Ronda {r.round}"
            )

    print()
    print("4/4 Próximos partidos cargados:")
    print("-" * 80)

    madrid_tz = "Europe/Madrid"

    for row in rows[:20]:
        dt = pd.Timestamp(
            row["utc_time"]
        ).tz_convert(
            madrid_tz
        )

        print(
            f"  {dt.strftime('%d/%m %H:%M')} | "
            f"{row['home_team']} - {row['away_team']} | "
            f"Jornada {row['round'] or '?'}"
        )

    print()
    print("=" * 80)
    print(
        f"OK: {inserted} fixtures guardados en "
        f"data/laliga.db"
    )
    print(
        f"FotMob season utilizado: "
        f"{season_param if season_param is not None else 'actual'}"
    )
    print(
        "Ahora reinicia Streamlit o pulsa Recargar jornada."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
