from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from update_laliga_fixtures_v3 import (
    DB_PATH,
    HEADERS,
    current_season_start,
    discover_current_payload,
    get_all_matches,
    historical_team_names,
    map_team_name,
    kickoff_utc,
    is_cancelled,
    stable_id,
)


BASE_DIR = Path(__file__).resolve().parent
RAW_DETAIL_DIR = (
    BASE_DIR
    / "data"
    / "laliga_raw"
    / "fotmob_referees"
    / "matches"
)

DETAIL_URLS = [
    "https://www.fotmob.com/api/matchDetails",
    "https://www.fotmob.com/api/data/matchDetails",
]

MAX_WORKERS = 4
TIMEOUT = 18
RETRIES = 3


def table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def table_columns(conn, name):
    if not table_exists(conn, name):
        return []
    return [
        row[1]
        for row in conn.execute(
            f'PRAGMA table_info("{name}")'
        ).fetchall()
    ]


def read_table(conn, name):
    if not table_exists(conn, name):
        return pd.DataFrame()
    return pd.read_sql_query(f'SELECT * FROM "{name}"', conn)


def save_status(conn, key, value):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_status (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
        """
    )
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO system_status(key, value, updated_at)
        VALUES(?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
        """,
        (key, str(value), now),
    )


def normalize_team(value):
    s = str(value or "").strip().lower()
    replacements = {
        "á": "a", "é": "e", "í": "i",
        "ó": "o", "ú": "u", "ü": "u",
        "ñ": "n",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)

    aliases = {
        "real betis": "betis",
        "real betis balompie": "betis",
        "atletico madrid": "ath madrid",
        "atletico de madrid": "ath madrid",
        "athletic club": "ath bilbao",
        "athletic bilbao": "ath bilbao",
        "rayo vallecano": "vallecano",
        "real sociedad": "sociedad",
        "rcd espanyol": "espanol",
        "espanyol": "espanol",
        "deportivo la coruna": "la coruna",
        "deportivo a coruna": "la coruna",
        "racing santander": "santander",
        "rc celta": "celta",
        "celta vigo": "celta",
        "deportivo alaves": "alaves",
    }
    s = aliases.get(s, s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def match_key(date_value, home, away):
    dt = pd.to_datetime(
        date_value,
        utc=True,
        errors="coerce",
    )
    day = (
        dt.strftime("%Y-%m-%d")
        if not pd.isna(dt)
        else str(date_value)[:10]
    )
    return (
        day,
        normalize_team(home),
        normalize_team(away),
    )


def finished_matches(payload):
    result = []

    for match in get_all_matches(payload):
        status = match.get("status", {}) or {}

        if (
            status.get("finished")
            and not is_cancelled(match)
        ):
            result.append(match)

    result.sort(
        key=lambda m: str(
            (m.get("status") or {}).get("utcTime")
            or m.get("utcTime")
            or ""
        )
    )
    return result


def _as_int_score(value):
    if value is None:
        return None

    try:
        if isinstance(value, bool):
            return None

        number = float(value)

        if math.isnan(number):
            return None

        return int(number)
    except Exception:
        return None


def _score_pair_from_text(value):
    if value is None:
        return None, None

    if isinstance(value, dict):
        home = (
            value.get("home")
            or value.get("homeScore")
            or value.get("homeTeam")
        )
        away = (
            value.get("away")
            or value.get("awayScore")
            or value.get("awayTeam")
        )

        hg = _as_int_score(home)
        ag = _as_int_score(away)

        if hg is not None and ag is not None:
            return hg, ag

        for key in ("scoreStr", "score", "text", "value"):
            if key in value:
                hg, ag = _score_pair_from_text(value.get(key))
                if hg is not None and ag is not None:
                    return hg, ag

        return None, None

    s = str(value).strip()

    m = re.search(
        r"(?<!\d)(\d{1,2})\s*[-–—:]\s*(\d{1,2})(?!\d)",
        s,
    )

    if not m:
        return None, None

    return int(m.group(1)), int(m.group(2))


def score_value(team):
    try:
        return _as_int_score(
            (team or {}).get("score")
        )
    except Exception:
        return None


def score_pair_from_match(match):
    home = match.get("home", {}) or {}
    away = match.get("away", {}) or {}
    status = match.get("status", {}) or {}

    hg = score_value(home)
    ag = score_value(away)

    if hg is not None and ag is not None:
        return hg, ag

    for h, a in (
        (match.get("homeScore"), match.get("awayScore")),
        (status.get("homeScore"), status.get("awayScore")),
        (status.get("scoreHome"), status.get("scoreAway")),
    ):
        hg = _as_int_score(h)
        ag = _as_int_score(a)

        if hg is not None and ag is not None:
            return hg, ag

    for candidate in (
        status.get("scoreStr"),
        status.get("score"),
        match.get("scoreStr"),
        match.get("score"),
    ):
        hg, ag = _score_pair_from_text(candidate)

        if hg is not None and ag is not None:
            return hg, ag

    return None, None


def score_pair_from_detail(detail):
    header = detail.get("header", {}) or {}
    teams = header.get("teams")

    if isinstance(teams, list) and len(teams) >= 2:
        hg = _as_int_score(
            (teams[0] or {}).get("score")
            if isinstance(teams[0], dict)
            else None
        )
        ag = _as_int_score(
            (teams[1] or {}).get("score")
            if isinstance(teams[1], dict)
            else None
        )

        if hg is not None and ag is not None:
            return hg, ag

    status = header.get("status", {}) or {}

    for candidate in (
        status.get("scoreStr"),
        status.get("score"),
        detail.get("scoreStr"),
        detail.get("score"),
    ):
        hg, ag = _score_pair_from_text(candidate)

        if hg is not None and ag is not None:
            return hg, ag

    found = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in {
                    "scorestr",
                    "score_string",
                    "scoretext",
                }:
                    found.append(value)
                walk(value)

        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(detail)

    for candidate in found:
        hg, ag = _score_pair_from_text(candidate)

        if hg is not None and ag is not None:
            return hg, ag

    return None, None


def format_score(value):
    value = _as_int_score(value)
    return "?" if value is None else str(value)


def finished_row(match, season, historical_names):
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

    home_goals, away_goals = score_pair_from_match(
        match
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
        "season": int(season),
        "round": str(round_value),
        "utc_time": ko.isoformat(),
        "date": ko.strftime("%Y-%m-%d"),
        "home_team_id": (
            home.get("id")
            or stable_id("home|" + str(home_raw))
        ),
        "home_team": home_name,
        "away_team_id": (
            away.get("id")
            or stable_id("away|" + str(away_raw))
        ),
        "away_team": away_name,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "odds_home": np.nan,
        "odds_draw": np.nan,
        "odds_away": np.nan,
        "finished": 1,
        "started": 1,
        "cancelled": 0,
        "valid_for_model": 1,
        "status": "Finished",
        "status_reason": (
            str(status.get("reason"))
            if status.get("reason") is not None
            else None
        ),
    }


# ---------------------------------------------------------------------
# MATCH DETAIL PARSER
# ---------------------------------------------------------------------

def scalar(v: Any) -> bool:
    return v is None or isinstance(
        v,
        (str, int, float, bool),
    )


def numeric(v: Any):
    if v is None:
        return None

    if isinstance(v, bool):
        return float(int(v))

    if isinstance(v, (int, float)):
        try:
            if math.isnan(v):
                return None
        except Exception:
            pass
        return float(v)

    s = str(v).strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)

    if not m:
        return None

    try:
        return float(m.group(0))
    except Exception:
        return None


def normalize_stat_name(value):
    s = str(value or "").strip().lower()
    s = (
        s.replace("(", " ")
        .replace(")", " ")
        .replace("%", " ")
        .replace("-", " ")
        .replace("/", " ")
    )
    s = re.sub(
        r"[^a-z0-9áéíóúüøæå ]+",
        " ",
        s,
    )
    return re.sub(
        r"\s+",
        "_",
        s,
    ).strip("_")


def collect_stat_pairs(node, out):
    if isinstance(node, dict):
        values = node.get("stats")
        name = (
            node.get("key")
            or node.get("title")
            or node.get("name")
        )

        if (
            name
            and isinstance(values, list)
            and len(values) >= 2
            and scalar(values[0])
            and scalar(values[1])
        ):
            out.append(
                (
                    normalize_stat_name(name),
                    values[0],
                    values[1],
                )
            )

        for value in node.values():
            collect_stat_pairs(value, out)

    elif isinstance(node, list):
        for value in node:
            collect_stat_pairs(value, out)


def all_stat_pairs(detail):
    periods = (
        detail.get("content", {})
        .get("stats", {})
        .get("Periods", {})
        .get("All", {})
    )

    pairs = []
    collect_stat_pairs(periods, pairs)

    clean = []
    seen = set()

    for item in pairs:
        if item not in seen:
            seen.add(item)
            clean.append(item)

    return clean


ALIASES = {
    "xg": {
        "expected_goals",
        "expected_goals_xg",
        "expected_goals_xg_",
    },
    "possession": {
        "ball_possession",
        "possession",
        "possession_percentage",
    },
    "shots": {
        "total_shots",
        "shots",
        "total_scoring_att",
    },
    "shots_on_target": {
        "shots_on_target",
        "shotsontarget",
        "ontarget_scoring_att",
        "shots_on_goal",
    },
    "corners": {
        "corner_kicks",
        "corners",
        "corner_taken",
    },
    "fouls": {
        "fouls",
        "fouls_committed",
        "fk_foul_lost",
    },
    "offsides": {
        "offsides",
        "total_offside",
    },
    "yellow_cards": {
        "yellow_cards",
        "yellow_card",
        "total_yel_card",
    },
    "red_cards": {
        "red_cards",
        "red_card",
        "total_red_card",
    },
}


def find_stat(pairs, target):
    aliases = ALIASES[target]

    for name, home, away in pairs:
        if name in aliases:
            return numeric(home), numeric(away)

    for name, home, away in pairs:
        for alias in aliases:
            if alias in name or name in alias:
                return numeric(home), numeric(away)

    return None, None


def extract_infobox(detail):
    return (
        detail.get("content", {})
        .get("matchFacts", {})
        .get("infoBox", {})
        or {}
    )


def text_from_infobox(value):
    if value is None:
        return None

    if isinstance(value, str):
        return value.strip() or None

    if isinstance(value, dict):
        for key in (
            "text",
            "name",
            "value",
            "label",
        ):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return None


def extract_referee(detail):
    info = extract_infobox(detail)

    for key in ("Referee", "referee"):
        if key in info:
            return text_from_infobox(
                info.get(key)
            )

    # Fallback recursivo por si FotMob mueve el campo.
    found = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if "referee" in str(k).lower():
                    text = text_from_infobox(v)
                    if text:
                        found.append(text)
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(detail)
    return found[0] if found else None


def extract_stadium(detail):
    info = extract_infobox(detail)

    for key in (
        "Stadium",
        "stadium",
        "Venue",
        "venue",
    ):
        if key in info:
            return text_from_infobox(
                info.get(key)
            )

    return None


def extract_attendance(detail):
    info = extract_infobox(detail)

    for key in (
        "Attendance",
        "attendance",
    ):
        if key in info:
            v = numeric(info.get(key))
            return (
                int(v)
                if v is not None
                else None
            )

    return None


def event_list(detail):
    content = detail.get("content", {})
    facts_events = (
        content.get(
            "matchFacts",
            {},
        ).get("events", {})
    )

    candidates = []

    if isinstance(facts_events, dict):
        for key in (
            "events",
            "incidents",
        ):
            value = facts_events.get(key)
            if isinstance(value, list):
                candidates = value
                break

    if not candidates:
        header_events = (
            detail.get(
                "header",
                {},
            ).get("events")
        )

        if isinstance(header_events, list):
            candidates = header_events

    return [
        x for x in candidates
        if isinstance(x, dict)
    ]


def card_fallback(detail):
    yh = ya = rh = ra = 0
    found = False

    for ev in event_list(detail):
        typ = str(
            ev.get("type")
            or ev.get("eventType")
            or ""
        ).lower()

        card = str(
            ev.get("card")
            or ev.get("cardType")
            or ev.get("typeStr")
            or ""
        ).lower()

        combined = f"{typ} {card}"

        if not any(
            word in combined
            for word in (
                "card",
                "yellow",
                "red",
            )
        ):
            continue

        is_home = ev.get("isHome")

        if not isinstance(is_home, bool):
            continue

        found = True

        second_yellow = (
            "second" in combined
            and "yellow" in combined
        )
        is_red = (
            "red" in combined
            or second_yellow
        )
        is_yellow = (
            "yellow" in combined
            and not second_yellow
        )

        if is_home:
            if is_red:
                rh += 1
            elif is_yellow:
                yh += 1
        else:
            if is_red:
                ra += 1
            elif is_yellow:
                ya += 1

    if not found:
        return None, None, None, None

    return yh, ya, rh, ra



def shotmap_sot_fallback(detail):
    """
    Fallback robusto para FotMob:
    si la tabla de stats no trae "Shots on target", contamos los disparos
    con isOnTarget=True del shotmap del propio partido.
    """
    general = detail.get("general", {}) or {}
    home_id = (
        (general.get("homeTeam") or {}).get("id")
        if isinstance(general.get("homeTeam"), dict)
        else None
    )
    away_id = (
        (general.get("awayTeam") or {}).get("id")
        if isinstance(general.get("awayTeam"), dict)
        else None
    )

    candidates = []

    content = detail.get("content", {}) or {}

    for node in (
        content.get("shotmap"),
        detail.get("shotmap"),
    ):
        if isinstance(node, dict):
            shots = node.get("shots")
            if isinstance(shots, list):
                candidates = shots
                break

    if not candidates:
        return None, None

    home_sot = 0
    away_sot = 0
    usable = 0

    for shot in candidates:
        if not isinstance(shot, dict):
            continue

        on_target = shot.get("isOnTarget")

        if on_target is not True:
            continue

        team_id = shot.get("teamId")

        if team_id is None:
            continue

        usable += 1

        if str(team_id) == str(home_id):
            home_sot += 1
        elif str(team_id) == str(away_id):
            away_sot += 1

    if usable == 0:
        # 0 tiros a puerta reales es posible, pero si no hemos podido
        # asociar ningún disparo no inventamos un 0-0.
        return None, None

    return float(home_sot), float(away_sot)


def parse_stats(detail, match_id, season):
    pairs = all_stat_pairs(detail)

    result = {
        "match_id": int(match_id),
        "season": int(season),
    }

    for target in ALIASES:
        home, away = find_stat(
            pairs,
            target,
        )
        result[f"home_{target}"] = home
        result[f"away_{target}"] = away

    if (
        result.get("home_shots_on_target") is None
        or result.get("away_shots_on_target") is None
    ):
        sot_home, sot_away = shotmap_sot_fallback(
            detail
        )

        if result.get("home_shots_on_target") is None:
            result["home_shots_on_target"] = sot_home

        if result.get("away_shots_on_target") is None:
            result["away_shots_on_target"] = sot_away

    yh, ya, rh, ra = card_fallback(
        detail
    )

    if (
        result["home_yellow_cards"]
        is None
        and yh is not None
    ):
        result[
            "home_yellow_cards"
        ] = float(yh)

    if (
        result["away_yellow_cards"]
        is None
        and ya is not None
    ):
        result[
            "away_yellow_cards"
        ] = float(ya)

    if (
        result["home_red_cards"]
        is None
        and rh is not None
    ):
        result[
            "home_red_cards"
        ] = float(rh)

    if (
        result["away_red_cards"]
        is None
        and ra is not None
    ):
        result[
            "away_red_cards"
        ] = float(ra)

    result["referee"] = extract_referee(
        detail
    )
    result["stadium"] = extract_stadium(
        detail
    )
    result["attendance"] = (
        extract_attendance(detail)
    )
    result["stat_keys"] = json.dumps(
        sorted(
            {
                name
                for name, _, _ in pairs
            }
        ),
        ensure_ascii=False,
    )

    return result


def cache_path(season, match_id):
    return (
        RAW_DETAIL_DIR
        / str(season)
        / f"{int(match_id)}.json"
    )


def load_cache(season, match_id):
    path = cache_path(
        season,
        match_id,
    )

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None


def save_cache(
    season,
    match_id,
    payload,
):
    path = cache_path(
        season,
        match_id,
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def fetch_detail(
    season,
    match_id,
):
    cached = load_cache(
        season,
        match_id,
    )

    if cached:
        return cached, False

    session = requests.Session()
    session.headers.update(HEADERS)

    last_error = None

    for url in DETAIL_URLS:
        for attempt in range(
            1,
            RETRIES + 1,
        ):
            try:
                r = session.get(
                    url,
                    params={
                        "matchId": int(
                            match_id
                        )
                    },
                    timeout=TIMEOUT,
                )

                if r.status_code == 200:
                    payload = r.json()
                    save_cache(
                        season,
                        match_id,
                        payload,
                    )
                    return payload, True

                last_error = (
                    f"HTTP {r.status_code}"
                )

            except Exception as exc:
                last_error = str(exc)

            if attempt < RETRIES:
                time.sleep(
                    0.8 * attempt
                )

    raise RuntimeError(
        last_error
        or "No se pudo descargar matchDetails"
    )


def process_detail(
    season,
    match,
):
    match_id = int(match["id"])
    detail, downloaded = fetch_detail(
        season,
        match_id,
    )
    stats = parse_stats(
        detail,
        match_id,
        season,
    )

    home_goals, away_goals = (
        score_pair_from_detail(
            detail
        )
    )

    return (
        stats,
        downloaded,
        home_goals,
        away_goals,
    )


def merge_schema(a, b):
    if a.empty:
        return b.copy()

    if b.empty:
        return a.copy()

    cols = list(
        dict.fromkeys(
            list(a.columns)
            + list(b.columns)
        )
    )

    return pd.concat(
        [
            a.reindex(columns=cols),
            b.reindex(columns=cols),
        ],
        ignore_index=True,
        sort=False,
    )


def main():
    print()
    print("=" * 96)
    print("FOOTBALL EDGE PRO - RESULTADOS FOTMOB INCREMENTALES V11.3")
    print("=" * 96)

    season = current_season_start()

    print(
        f"Temporada actual: "
        f"{season}/{season+1}"
    )
    print(
        "Fuente de resultados recientes: FotMob."
    )
    print(
        "Football-Data se mantiene para histórico/cuotas, "
        "pero ya NO decide si hay un partido nuevo."
    )
    print()

    session = requests.Session()
    session.headers.update(HEADERS)

    # La función ya sabe probar los distintos formatos de season.
    payload, season_param, _future = (
        discover_current_payload(
            session,
            season,
        )
    )

    all_matches = get_all_matches(
        payload
    )
    finished = finished_matches(
        payload
    )

    print()
    print(
        f"FotMob temporada: "
        f"{len(all_matches)} partidos | "
        f"{len(finished)} finalizados"
    )

    with sqlite3.connect(DB_PATH) as conn:
        historical_names = (
            historical_team_names(
                conn
            )
        )

        old_matches = read_table(
            conn,
            "fotmob_matches",
        )
        old_matches_alias = read_table(
            conn,
            "matches",
        )
        old_stats = read_table(
            conn,
            "fotmob_match_stats",
        )

        if old_matches.empty:
            old_current = pd.DataFrame()
        else:
            old_current = old_matches[
                pd.to_numeric(
                    old_matches.get(
                        "season",
                        np.nan,
                    ),
                    errors="coerce",
                ) == season
            ].copy()

        old_keys = set()

        for r in old_current.itertuples(
            index=False
        ):
            date_source = (
                getattr(
                    r,
                    "utc_time",
                    None,
                )
                or getattr(
                    r,
                    "date",
                    None,
                )
            )

            old_keys.add(
                match_key(
                    date_source,
                    getattr(
                        r,
                        "home_team",
                        "",
                    ),
                    getattr(
                        r,
                        "away_team",
                        "",
                    ),
                )
            )

        rows = []

        for match in finished:
            row = finished_row(
                match,
                season,
                historical_names,
            )

            if row is not None:
                rows.append(row)

        current_df = pd.DataFrame(
            rows
        ).sort_values(
            ["utc_time", "match_id"]
        ).reset_index(drop=True)

        current_keys = {
            match_key(
                r.utc_time,
                r.home_team,
                r.away_team,
            )
            for r in current_df.itertuples(
                index=False
            )
        }

        new_real_keys = (
            current_keys - old_keys
        )

        # Recuperar las cuotas de Football-Data que ya teníamos en 31
        # partidos haciendo match por fecha + equipos, aunque cambiemos
        # a IDs reales de FotMob.
        old_by_key = {}

        for r in old_current.itertuples(
            index=False
        ):
            key = match_key(
                getattr(
                    r,
                    "utc_time",
                    None,
                )
                or getattr(
                    r,
                    "date",
                    None,
                ),
                getattr(
                    r,
                    "home_team",
                    "",
                ),
                getattr(
                    r,
                    "away_team",
                    "",
                ),
            )
            old_by_key[key] = r

        for idx, row in current_df.iterrows():
            key = match_key(
                row["utc_time"],
                row["home_team"],
                row["away_team"],
            )

            old = old_by_key.get(key)

            if old is not None:
                for col in (
                    "odds_home",
                    "odds_draw",
                    "odds_away",
                ):
                    value = getattr(
                        old,
                        col,
                        np.nan,
                    )

                    if not pd.isna(value):
                        current_df.at[
                            idx,
                            col,
                        ] = value

        # Detectar migración: IDs sintéticos Football-Data -> IDs FotMob.
        old_current_ids = set(
            pd.to_numeric(
                old_current.get(
                    "match_id",
                    pd.Series(
                        dtype=float
                    ),
                ),
                errors="coerce",
            ).dropna().astype(int)
        )

        fotmob_current_ids = set(
            pd.to_numeric(
                current_df["match_id"],
                errors="coerce",
            ).dropna().astype(int)
        )

        migration_needed = bool(
            old_current_ids
            and not old_current_ids.issubset(
                fotmob_current_ids
            )
        )

        if migration_needed:
            print()
            print(
                "🔧 Migración única: la temporada actual "
                "pasa de IDs sintéticos Football-Data "
                "a IDs reales FotMob."
            )
            print(
                "   Esto hace que resultados, estadísticas "
                "y árbitros compartan el mismo match_id."
            )

    print()
    print(
        f"Partidos nuevos REALES detectados: "
        f"{len(new_real_keys)}"
    )

    if new_real_keys:
        print("Nuevos:")
        for r in current_df.itertuples(
            index=False
        ):
            key = match_key(
                r.utc_time,
                r.home_team,
                r.away_team,
            )

            if key in new_real_keys:
                print(
                    f"  + {str(r.date)} · "
                    f"{r.home_team} "
                    f"{format_score(r.home_goals)}-"
                    f"{format_score(r.away_goals)} "
                    f"{r.away_team}"
                )

    # Para una migración inicial necesitamos estadísticas de todos los
    # finalizados de 2026/27. Casi todos ya están en la caché del downloader
    # de árbitros, por lo que esto suele tardar segundos.
    print()
    print(
        "Comprobando estadísticas completas "
        f"de {len(finished)} partidos actuales..."
    )

    stats_rows = []
    detail_scores = {}
    downloads = 0
    cache_hits = 0
    errors = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:
        futures = {
            executor.submit(
                process_detail,
                season,
                match,
            ): match
            for match in finished
        }

        done = 0

        for future in as_completed(
            futures
        ):
            done += 1
            match = futures[future]

            try:
                (
                    stats,
                    downloaded,
                    home_goals_detail,
                    away_goals_detail,
                ) = future.result()

                stats_rows.append(
                    stats
                )

                detail_scores[
                    int(match["id"])
                ] = (
                    home_goals_detail,
                    away_goals_detail,
                )

                if downloaded:
                    downloads += 1
                else:
                    cache_hits += 1

            except Exception as exc:
                errors.append(
                    (
                        int(match["id"]),
                        str(exc),
                    )
                )

            if (
                done % 10 == 0
                or done == len(futures)
            ):
                print(
                    f"  [{done}/{len(futures)}] "
                    f"cache={cache_hits} | "
                    f"nuevas={downloads} | "
                    f"errores={len(errors)}"
                )

    if errors:
        print()
        for mid, error in errors[:10]:
            print(
                f"ERROR match {mid}: {error}"
            )

        raise RuntimeError(
            "No se guardará la migración porque faltan "
            f"{len(errors)} matchDetails."
        )

    # El listado de liga de FotMob puede marcar un partido como
    # finalizado sin incluir el marcador. matchDetails sí suele traerlo.
    current_df = current_df.copy()

    for idx, row in current_df.iterrows():
        mid = int(row["match_id"])

        hg = _as_int_score(
            row.get("home_goals")
        )
        ag = _as_int_score(
            row.get("away_goals")
        )

        if (
            (hg is None or ag is None)
            and mid in detail_scores
        ):
            dh, da = detail_scores[mid]

            if hg is None:
                hg = _as_int_score(dh)

            if ag is None:
                ag = _as_int_score(da)

        current_df.at[idx, "home_goals"] = hg
        current_df.at[idx, "away_goals"] = ag

    missing_scores = current_df[
        current_df["home_goals"].isna()
        | current_df["away_goals"].isna()
    ]

    if not missing_scores.empty:
        print()
        print(
            "ERROR: hay partidos finalizados sin marcador confirmado:"
        )
        print(
            missing_scores[
                [
                    "match_id",
                    "date",
                    "home_team",
                    "away_team",
                ]
            ].to_string(index=False)
        )
        raise RuntimeError(
            "No se guardará ningún partido con marcador vacío."
        )

    current_df["home_goals"] = pd.to_numeric(
        current_df["home_goals"],
        errors="raise",
    ).astype(int)

    current_df["away_goals"] = pd.to_numeric(
        current_df["away_goals"],
        errors="raise",
    ).astype(int)

    if new_real_keys:
        print()
        print("Nuevos con marcador confirmado:")

        for r in current_df.itertuples(index=False):
            key = match_key(
                r.utc_time,
                r.home_team,
                r.away_team,
            )

            if key in new_real_keys:
                print(
                    f"  ✅ {str(r.date)} · "
                    f"{r.home_team} "
                    f"{int(r.home_goals)}-"
                    f"{int(r.away_goals)} "
                    f"{r.away_team}"
                )

    stats_current = pd.DataFrame(
        stats_rows
    )

    # Cobertura informativa.
    for col in (
        "home_shots",
        "home_shots_on_target",
        "home_corners",
        "home_fouls",
        "home_yellow_cards",
    ):
        if col in stats_current.columns:
            print(
                f"  Cobertura {col}: "
                f"{int(stats_current[col].notna().sum())}/"
                f"{len(stats_current)}"
            )

    with sqlite3.connect(DB_PATH) as conn:
        old_matches = read_table(
            conn,
            "fotmob_matches",
        )
        old_matches_alias = read_table(
            conn,
            "matches",
        )
        old_stats = read_table(
            conn,
            "fotmob_match_stats",
        )

        # Conservar histórico de otras temporadas.
        if old_matches.empty:
            hist_matches = pd.DataFrame()
        else:
            hist_matches = old_matches[
                pd.to_numeric(
                    old_matches.get(
                        "season",
                        np.nan,
                    ),
                    errors="coerce",
                ) != season
            ].copy()

        if old_matches_alias.empty:
            hist_alias = pd.DataFrame()
        else:
            hist_alias = old_matches_alias[
                pd.to_numeric(
                    old_matches_alias.get(
                        "season",
                        np.nan,
                    ),
                    errors="coerce",
                ) != season
            ].copy()

        # Stats: borrar los IDs antiguos de la temporada actual.
        old_current_ids = set()

        if not old_matches.empty:
            old_current_ids = set(
                pd.to_numeric(
                    old_matches.loc[
                        pd.to_numeric(
                            old_matches.get(
                                "season",
                                np.nan,
                            ),
                            errors="coerce",
                        ) == season,
                        "match_id",
                    ],
                    errors="coerce",
                ).dropna().astype(int)
            )

        current_fotmob_ids = set(
            current_df[
                "match_id"
            ].astype(int)
        )

        ids_to_remove = (
            old_current_ids
            | current_fotmob_ids
        )

        if old_stats.empty:
            hist_stats = pd.DataFrame()
        elif "match_id" in old_stats.columns:
            hist_stats = old_stats[
                ~pd.to_numeric(
                    old_stats["match_id"],
                    errors="coerce",
                ).isin(
                    ids_to_remove
                )
            ].copy()
        else:
            hist_stats = pd.DataFrame()

        matches_merged = merge_schema(
            hist_matches,
            current_df,
        )
        alias_merged = merge_schema(
            hist_alias,
            current_df,
        )
        stats_merged = merge_schema(
            hist_stats,
            stats_current,
        )

        matches_merged.to_sql(
            "fotmob_matches",
            conn,
            if_exists="replace",
            index=False,
        )

        alias_merged.to_sql(
            "matches",
            conn,
            if_exists="replace",
            index=False,
        )

        stats_merged.to_sql(
            "fotmob_match_stats",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_laliga_matches_time
            ON fotmob_matches(utc_time);

            CREATE INDEX IF NOT EXISTS idx_laliga_matches_season
            ON fotmob_matches(season);

            CREATE INDEX IF NOT EXISTS idx_laliga_stats_match
            ON fotmob_match_stats(match_id);
            """
        )

        save_status(
            conn,
            "laliga_last_new_matches",
            len(new_real_keys),
        )
        save_status(
            conn,
            "laliga_current_season",
            season,
        )
        save_status(
            conn,
            "laliga_current_season_matches",
            len(current_df),
        )
        save_status(
            conn,
            "laliga_results_source",
            "FotMob",
        )

        conn.commit()

    print()
    print("=" * 96)
    print("RESULTADOS FOTMOB ACTUALIZADOS")
    print("=" * 96)
    print(
        f"Partidos finalizados temporada: "
        f"{len(current_df)}"
    )
    print(
        f"Partidos nuevos reales: "
        f"{len(new_real_keys)}"
    )
    print(
        f"Detalles reutilizados de caché: "
        f"{cache_hits}"
    )
    print(
        f"Detalles descargados ahora: "
        f"{downloads}"
    )
    print(
        "✅ Ya no dependemos del retraso de Football-Data "
        "para los resultados del día anterior."
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
