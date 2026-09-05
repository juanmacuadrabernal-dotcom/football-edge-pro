from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import requests


LEAGUE_ID = 59
COUNTRY_CODE = "NOR"
START_SEASON = 2023

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "fotmob_raw"
DB_PATH = DATA_DIR / "eliteserien.db"

LEAGUE_URL = "https://www.fotmob.com/api/data/leagues"
MATCH_DETAIL_URLS = [
    "https://www.fotmob.com/api/data/matchDetails",
    "https://www.fotmob.com/api/matchDetails",
]

REQUEST_DELAY = 0.85
TIMEOUT = 25
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.fotmob.com/",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------
# HTTP
# ---------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)

            if r.status_code == 200:
                return r.json()

            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(3 * attempt, 10)
                print(f"  HTTP {r.status_code}. Reintentando en {wait}s...")
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"HTTP {r.status_code} en {r.url}: {r.text[:250]}"
            )

        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                wait = min(3 * attempt, 10)
                time.sleep(wait)

    raise RuntimeError(str(last_error) if last_error else "Error HTTP desconocido")


def get_league_season(
    session: requests.Session,
    season: int,
) -> dict[str, Any]:
    return get_json(
        session,
        LEAGUE_URL,
        {
            "id": LEAGUE_ID,
            "ccode3": COUNTRY_CODE,
            "season": str(season),
        },
    )


def get_match_detail(
    session: requests.Session,
    match_id: int,
) -> dict[str, Any]:
    errors = []

    for url in MATCH_DETAIL_URLS:
        try:
            return get_json(
                session,
                url,
                {"matchId": str(match_id)},
                retries=2,
            )
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(" | ".join(errors))


# ---------------------------------------------------------
# JSON / STATS
# ---------------------------------------------------------

def scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def numeric(v: Any) -> float | None:
    if v is None:
        return None

    if isinstance(v, bool):
        return float(int(v))

    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)

    s = str(v).strip().replace(",", ".")

    # "54%", "12", "1.47", "435/510 (85%)" -> primer número
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None

    try:
        return float(m.group(0))
    except ValueError:
        return None


def normalize_name(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = (
        s.replace("(", " ")
        .replace(")", " ")
        .replace("%", " ")
        .replace("-", " ")
        .replace("/", " ")
    )
    s = re.sub(r"[^a-z0-9áéíóúüøæå ]+", " ", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s


def collect_stat_pairs(node: Any, out: list[tuple[str, Any, Any]]) -> None:
    """
    Busca recursivamente objetos con:
      key/title + stats:[home, away]

    Esto soporta tanto la estructura agrupada como la estructura plana
    que FotMob ha usado en distintas versiones.
    """
    if isinstance(node, dict):
        values = node.get("stats")
        name = node.get("key") or node.get("title") or node.get("name")

        if (
            name
            and isinstance(values, list)
            and len(values) >= 2
            and scalar(values[0])
            and scalar(values[1])
        ):
            out.append((normalize_name(name), values[0], values[1]))

        for value in node.values():
            collect_stat_pairs(value, out)

    elif isinstance(node, list):
        for value in node:
            collect_stat_pairs(value, out)


def all_stat_pairs(detail: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    periods = (
        detail.get("content", {})
        .get("stats", {})
        .get("Periods", {})
        .get("All", {})
    )

    pairs: list[tuple[str, Any, Any]] = []
    collect_stat_pairs(periods, pairs)

    # Quitar duplicados exactos conservando orden.
    seen = set()
    clean = []
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
    "big_chances": {
        "big_chances",
        "big_chance",
    },
    "big_chances_missed": {
        "big_chances_missed",
        "big_chance_missed",
    },
    "saves": {
        "saves",
        "total_saves",
    },
    "passes": {
        "passes",
        "total_pass",
    },
    "accurate_passes": {
        "accurate_passes",
        "accurate_pass",
    },
    "tackles": {
        "tackles",
        "total_tackle",
    },
}


def find_stat(
    pairs: list[tuple[str, Any, Any]],
    target: str,
) -> tuple[float | None, float | None]:
    aliases = ALIASES[target]

    # Coincidencia exacta.
    for name, home, away in pairs:
        if name in aliases:
            return numeric(home), numeric(away)

    # Coincidencia parcial para cambios menores de etiqueta.
    for name, home, away in pairs:
        for alias in aliases:
            if alias in name or name in alias:
                return numeric(home), numeric(away)

    return None, None


def extract_infobox(detail: dict[str, Any]) -> dict[str, Any]:
    return (
        detail.get("content", {})
        .get("matchFacts", {})
        .get("infoBox", {})
        or {}
    )


def text_from_infobox_value(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value.strip() or None

    if isinstance(value, dict):
        for key in ("text", "name", "value", "label"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return None


def extract_referee(detail: dict[str, Any]) -> str | None:
    info = extract_infobox(detail)

    for key in ("Referee", "referee"):
        if key in info:
            return text_from_infobox_value(info.get(key))

    return None


def extract_stadium(detail: dict[str, Any]) -> str | None:
    info = extract_infobox(detail)

    for key in ("Stadium", "stadium", "Venue", "venue"):
        if key in info:
            return text_from_infobox_value(info.get(key))

    return None


def extract_attendance(detail: dict[str, Any]) -> int | None:
    info = extract_infobox(detail)

    for key in ("Attendance", "attendance"):
        if key in info:
            v = numeric(info.get(key))
            return int(v) if v is not None else None

    return None


def event_list(detail: dict[str, Any]) -> list[dict[str, Any]]:
    content = detail.get("content", {})
    facts_events = content.get("matchFacts", {}).get("events", {})

    candidates = []

    if isinstance(facts_events, dict):
        for key in ("events", "incidents"):
            value = facts_events.get(key)
            if isinstance(value, list):
                candidates = value
                break

    if not candidates:
        header_events = detail.get("header", {}).get("events")
        if isinstance(header_events, list):
            candidates = header_events

    return [x for x in candidates if isinstance(x, dict)]


def card_fallback(
    detail: dict[str, Any],
) -> tuple[int | None, int | None, int | None, int | None]:
    yh = ya = rh = ra = 0
    found = False

    for ev in event_list(detail):
        typ = str(ev.get("type") or ev.get("eventType") or "").lower()
        card = str(
            ev.get("card")
            or ev.get("cardType")
            or ev.get("typeStr")
            or ""
        ).lower()

        combined = f"{typ} {card}"

        if "card" not in combined and "yellow" not in combined and "red" not in combined:
            continue

        is_home = ev.get("isHome")
        if not isinstance(is_home, bool):
            continue

        found = True

        second_yellow = "second" in combined and "yellow" in combined
        is_red = "red" in combined or second_yellow
        is_yellow = "yellow" in combined and not second_yellow

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


def extract_match_stats(detail: dict[str, Any]) -> dict[str, Any]:
    pairs = all_stat_pairs(detail)

    result: dict[str, Any] = {}

    for target in ALIASES:
        home, away = find_stat(pairs, target)
        result[f"home_{target}"] = home
        result[f"away_{target}"] = away

    # Fallback de tarjetas mediante eventos.
    yh, ya, rh, ra = card_fallback(detail)

    if result["home_yellow_cards"] is None and yh is not None:
        result["home_yellow_cards"] = float(yh)
    if result["away_yellow_cards"] is None and ya is not None:
        result["away_yellow_cards"] = float(ya)
    if result["home_red_cards"] is None and rh is not None:
        result["home_red_cards"] = float(rh)
    if result["away_red_cards"] is None and ra is not None:
        result["away_red_cards"] = float(ra)

    result["referee"] = extract_referee(detail)
    result["stadium"] = extract_stadium(detail)
    result["attendance"] = extract_attendance(detail)
    result["stat_keys"] = json.dumps(
        sorted({name for name, _, _ in pairs}),
        ensure_ascii=False,
    )

    return result


# ---------------------------------------------------------
# MATCH LIST
# ---------------------------------------------------------

def get_all_matches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extrae partidos aunque FotMob cambie la posición exacta dentro del JSON.

    Históricamente la API ha devuelto estructuras distintas, por ejemplo:
      - matches.allMatches
      - matches como lista
      - fixtures dentro de bloques de la página

    Para no depender de una ruta rígida, buscamos recursivamente objetos que
    tengan la forma de un partido: id + home + away + status.
    """
    found: dict[int, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            home = node.get("home")
            away = node.get("away")
            status = node.get("status")
            match_id = node.get("id")

            if (
                match_id is not None
                and isinstance(home, dict)
                and isinstance(away, dict)
                and isinstance(status, dict)
                and home.get("name")
                and away.get("name")
            ):
                try:
                    found[int(match_id)] = node
                except Exception:
                    pass

            for value in node.values():
                walk(value)

        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)

    matches = list(found.values())

    def sort_key(match: dict[str, Any]) -> str:
        status = match.get("status", {}) or {}
        return str(status.get("utcTime") or match.get("utcTime") or "")

    return sorted(matches, key=sort_key)


def is_finished(match: dict[str, Any]) -> bool:
    status = match.get("status", {}) or {}
    return bool(status.get("finished"))


def score_value(team: dict[str, Any]) -> int | None:
    value = team.get("score")
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def match_row(match: dict[str, Any], season: int) -> dict[str, Any]:
    home = match.get("home", {}) or {}
    away = match.get("away", {}) or {}
    status = match.get("status", {}) or {}

    return {
        "match_id": int(match["id"]),
        "season": int(season),
        "round": str(
            match.get("round")
            or match.get("roundName")
            or match.get("tournamentStage")
            or ""
        ),
        "utc_time": status.get("utcTime") or match.get("utcTime"),
        "home_team_id": home.get("id"),
        "home_team": home.get("name"),
        "away_team_id": away.get("id"),
        "away_team": away.get("name"),
        "home_goals": score_value(home),
        "away_goals": score_value(away),
        "finished": 1 if status.get("finished") else 0,
        "started": 1 if status.get("started") else 0,
        "cancelled": 1 if status.get("cancelled") else 0,
        "status_reason": (
            status.get("reason")
            if isinstance(status.get("reason"), str)
            else json.dumps(status.get("reason"), ensure_ascii=False)
            if status.get("reason") is not None
            else None
        ),
    }


# ---------------------------------------------------------
# SQLITE
# ---------------------------------------------------------

MATCH_COLS = [
    "match_id", "season", "round", "utc_time",
    "home_team_id", "home_team", "away_team_id", "away_team",
    "home_goals", "away_goals",
    "finished", "started", "cancelled", "status_reason",
]

STAT_COLS = [
    "match_id", "season",
    "home_xg", "away_xg",
    "home_possession", "away_possession",
    "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners",
    "home_fouls", "away_fouls",
    "home_offsides", "away_offsides",
    "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards",
    "home_big_chances", "away_big_chances",
    "home_big_chances_missed", "away_big_chances_missed",
    "home_saves", "away_saves",
    "home_passes", "away_passes",
    "home_accurate_passes", "away_accurate_passes",
    "home_tackles", "away_tackles",
    "referee", "stadium", "attendance", "stat_keys",
]


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS fotmob_matches (
            match_id INTEGER PRIMARY KEY,
            season INTEGER,
            round TEXT,
            utc_time TEXT,
            home_team_id INTEGER,
            home_team TEXT,
            away_team_id INTEGER,
            away_team TEXT,
            home_goals INTEGER,
            away_goals INTEGER,
            finished INTEGER,
            started INTEGER,
            cancelled INTEGER,
            status_reason TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS fotmob_match_stats (
            match_id INTEGER PRIMARY KEY,
            season INTEGER,
            home_xg REAL,
            away_xg REAL,
            home_possession REAL,
            away_possession REAL,
            home_shots REAL,
            away_shots REAL,
            home_shots_on_target REAL,
            away_shots_on_target REAL,
            home_corners REAL,
            away_corners REAL,
            home_fouls REAL,
            away_fouls REAL,
            home_offsides REAL,
            away_offsides REAL,
            home_yellow_cards REAL,
            away_yellow_cards REAL,
            home_red_cards REAL,
            away_red_cards REAL,
            home_big_chances REAL,
            away_big_chances REAL,
            home_big_chances_missed REAL,
            away_big_chances_missed REAL,
            home_saves REAL,
            away_saves REAL,
            home_passes REAL,
            away_passes REAL,
            home_accurate_passes REAL,
            away_accurate_passes REAL,
            home_tackles REAL,
            away_tackles REAL,
            referee TEXT,
            stadium TEXT,
            attendance INTEGER,
            stat_keys TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS fotmob_referees (
            referee TEXT PRIMARY KEY,
            matches INTEGER,
            avg_yellow_cards REAL,
            avg_red_cards REAL,
            avg_fouls REAL,
            avg_corners REAL,
            first_season INTEGER,
            last_season INTEGER,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_fm_matches_season
            ON fotmob_matches(season);

        CREATE INDEX IF NOT EXISTS idx_fm_matches_time
            ON fotmob_matches(utc_time);

        CREATE INDEX IF NOT EXISTS idx_fm_stats_referee
            ON fotmob_match_stats(referee);
        """
    )


def upsert_match(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    values = [row.get(c) for c in MATCH_COLS]
    placeholders = ",".join("?" for _ in MATCH_COLS)

    updates = ",".join(
        f"{c}=excluded.{c}" for c in MATCH_COLS if c != "match_id"
    )

    conn.execute(
        f"""
        INSERT INTO fotmob_matches ({",".join(MATCH_COLS)})
        VALUES ({placeholders})
        ON CONFLICT(match_id) DO UPDATE SET
            {updates},
            updated_at=CURRENT_TIMESTAMP
        """,
        values,
    )


def upsert_stats(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    values = [row.get(c) for c in STAT_COLS]
    placeholders = ",".join("?" for _ in STAT_COLS)

    updates = ",".join(
        f"{c}=excluded.{c}" for c in STAT_COLS if c != "match_id"
    )

    conn.execute(
        f"""
        INSERT INTO fotmob_match_stats ({",".join(STAT_COLS)})
        VALUES ({placeholders})
        ON CONFLICT(match_id) DO UPDATE SET
            {updates},
            updated_at=CURRENT_TIMESTAMP
        """,
        values,
    )


def rebuild_referees(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM fotmob_referees")

    conn.execute(
        """
        INSERT INTO fotmob_referees (
            referee,
            matches,
            avg_yellow_cards,
            avg_red_cards,
            avg_fouls,
            avg_corners,
            first_season,
            last_season
        )
        SELECT
            referee,
            COUNT(*) AS matches,
            AVG(
                COALESCE(home_yellow_cards, 0)
                + COALESCE(away_yellow_cards, 0)
            ) AS avg_yellow_cards,
            AVG(
                COALESCE(home_red_cards, 0)
                + COALESCE(away_red_cards, 0)
            ) AS avg_red_cards,
            AVG(
                CASE
                    WHEN home_fouls IS NOT NULL AND away_fouls IS NOT NULL
                    THEN home_fouls + away_fouls
                    ELSE NULL
                END
            ) AS avg_fouls,
            AVG(
                CASE
                    WHEN home_corners IS NOT NULL AND away_corners IS NOT NULL
                    THEN home_corners + away_corners
                    ELSE NULL
                END
            ) AS avg_corners,
            MIN(season),
            MAX(season)
        FROM fotmob_match_stats
        WHERE referee IS NOT NULL
          AND TRIM(referee) <> ''
        GROUP BY referee
        """
    )


# ---------------------------------------------------------
# CACHE
# ---------------------------------------------------------

def season_cache_path(season: int) -> Path:
    return RAW_DIR / "seasons" / f"eliteserien_{season}.json"


def detail_cache_path(season: int, match_id: int) -> Path:
    return RAW_DIR / "matches" / str(season) / f"{match_id}.json"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def discover_seasons(session: requests.Session) -> list[int]:
    payload = get_league_season(session, START_SEASON)

    raw = payload.get("allAvailableSeasons", [])
    seasons = []

    for value in raw:
        try:
            year = int(str(value)[:4])
            if year >= START_SEASON:
                seasons.append(year)
        except Exception:
            pass

    return sorted(set(seasons))


def coverage(conn: sqlite3.Connection) -> dict[str, int]:
    fields = {
        "xG": "home_xg",
        "Tiros": "home_shots",
        "Tiros a puerta": "home_shots_on_target",
        "Corners": "home_corners",
        "Faltas": "home_fouls",
        "Offsides": "home_offsides",
        "Amarillas": "home_yellow_cards",
        "Rojas": "home_red_cards",
        "Arbitro": "referee",
    }

    result = {}

    for label, col in fields.items():
        row = conn.execute(
            f"""
            SELECT COUNT(*) FROM fotmob_match_stats
            WHERE {col} IS NOT NULL
            """
        ).fetchone()
        result[label] = int(row[0])

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Vuelve a descargar detalles aunque ya estén en caché.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = make_session()

    print()
    print("=" * 68)
    print("ELITESERIEN EDGE PRO - FOTMOB STATS")
    print("=" * 68)
    print("Fuente: FotMob")
    print(f"Liga: Eliteserien (ID {LEAGUE_ID})")
    print(f"Datos completos desde: {START_SEASON}")
    print()

    try:
        seasons = discover_seasons(session)
    except Exception as exc:
        print("ERROR conectando con FotMob:")
        print(exc)
        return 1

    if not seasons:
        print("No se detectaron temporadas desde 2023.")
        return 1

    print("Temporadas detectadas:", seasons)
    print()

    with sqlite3.connect(DB_PATH) as conn:
        init_db(conn)

        all_finished: list[tuple[int, dict[str, Any]]] = []

        # 1) Calendario y resultados
        for season in seasons:
            print(f"[Temporada {season}] descargando calendario...")

            try:
                payload = get_league_season(session, season)
            except Exception as exc:
                print(f"  ERROR temporada {season}: {exc}")
                continue

            save_json(season_cache_path(season), payload)

            matches = get_all_matches(payload)
            finished = 0

            for match in matches:
                if "id" not in match:
                    continue

                row = match_row(match, season)
                upsert_match(conn, row)

                if is_finished(match):
                    all_finished.append((season, match))
                    finished += 1

            conn.commit()

            print(
                f"  Partidos encontrados: {len(matches)} | "
                f"finalizados: {finished}"
            )

            time.sleep(REQUEST_DELAY)

        print()
        print(f"Partidos finalizados a procesar: {len(all_finished)}")
        print("Los partidos ya descargados se reutilizan automáticamente.")
        print()

        # 2) Detalles y estadísticas
        ok = 0
        cache_hits = 0
        errors = 0
        total = len(all_finished)
        start_time = time.time()

        for index, (season, match) in enumerate(all_finished, start=1):
            match_id = int(match["id"])
            home = (match.get("home") or {}).get("name", "?")
            away = (match.get("away") or {}).get("name", "?")
            cache_path = detail_cache_path(season, match_id)

            detail = None

            if not args.refresh:
                detail = load_json(cache_path)
                if detail:
                    cache_hits += 1

            if detail is None:
                try:
                    detail = get_match_detail(session, match_id)
                    save_json(cache_path, detail)
                    time.sleep(REQUEST_DELAY)
                except Exception as exc:
                    errors += 1
                    print(
                        f"[{index}/{total}] ERROR {season} "
                        f"{home} - {away} ({match_id}): {exc}"
                    )

                    # Si el primer partido ya falla por 403, paramos para no
                    # hacer cientos de peticiones inútiles.
                    if index <= 2 and "403" in str(exc):
                        print()
                        print("FotMob permite la liga pero bloquea matchDetails.")
                        print("No se continuará para evitar peticiones innecesarias.")
                        return 2

                    continue

            try:
                stats = extract_match_stats(detail)
                stats["match_id"] = match_id
                stats["season"] = season
                upsert_stats(conn, stats)
                conn.commit()
                ok += 1

            except Exception as exc:
                errors += 1
                print(
                    f"[{index}/{total}] ERROR parseando "
                    f"{home} - {away}: {exc}"
                )
                continue

            elapsed = time.time() - start_time
            avg = elapsed / max(index, 1)
            remaining = avg * (total - index)

            if index == 1 or index % 10 == 0 or index == total:
                print(
                    f"[{index}/{total}] OK | "
                    f"{home} - {away} | "
                    f"faltan aprox. {remaining/60:.1f} min"
                )

        rebuild_referees(conn)
        conn.commit()

        cov = coverage(conn)

        total_matches = conn.execute(
            "SELECT COUNT(*) FROM fotmob_matches"
        ).fetchone()[0]

        total_stats = conn.execute(
            "SELECT COUNT(*) FROM fotmob_match_stats"
        ).fetchone()[0]

        refs = conn.execute(
            "SELECT COUNT(*) FROM fotmob_referees"
        ).fetchone()[0]

    print()
    print("=" * 68)
    print("DESCARGA FOTMOB COMPLETADA")
    print("=" * 68)
    print(f"Calendario/partidos guardados: {total_matches}")
    print(f"Partidos con detalle guardado: {total_stats}")
    print(f"Detalles reutilizados de cache: {cache_hits}")
    print(f"Errores: {errors}")
    print(f"Arbitros detectados: {refs}")
    print()
    print("Cobertura:")
    for label, count in cov.items():
        print(f"  {label:<17} {count} partidos")
    print()
    print(f"Base de datos: {DB_PATH}")
    print(f"JSON crudos: {RAW_DIR}")
    print("=" * 68)
    print()
    print("IMPORTANTE:")
    print(
        "Este archivo descarga estadísticas completas de equipos desde 2023. "
        "El histórico anterior de árbitros se añadirá en un módulo separado "
        "para no convertir esta primera descarga en miles de peticiones extra."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
