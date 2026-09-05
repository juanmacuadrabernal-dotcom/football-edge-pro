from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import requests


LEAGUE_ID = 59
COUNTRY_CODE = "NOR"
HISTORICAL_END_SEASON = 2022

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "eliteserien.db"
CACHE_DIR = DATA_DIR / "fotmob_referee_raw"

LEAGUE_URL = "https://www.fotmob.com/api/data/leagues"
MATCH_DETAIL_URLS = [
    "https://www.fotmob.com/api/data/matchDetails",
    "https://www.fotmob.com/api/matchDetails",
]

TIMEOUT = 25
REQUEST_DELAY = 0.80

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


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    retries: int = 3,
) -> dict[str, Any]:
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)

            if r.status_code == 200:
                return r.json()

            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(3 * attempt, 10)
                print(f"  HTTP {r.status_code}; reintento en {wait}s...")
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"HTTP {r.status_code}: {r.text[:250]}"
            )

        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(3 * attempt, 10))

    raise RuntimeError(str(last_error))


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
# PARTIDOS ANTIGUOS: RUTA CORRECTA
# ---------------------------------------------------------

def get_matches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    FotMob antiguo:
      fixtures.allMatches
      overview.matches.allMatches

    Priorizamos fixtures.allMatches.
    """

    fixtures = as_dict(payload.get("fixtures"))
    matches = fixtures.get("allMatches")

    if isinstance(matches, list) and matches:
        return [m for m in matches if isinstance(m, dict)]

    overview = as_dict(payload.get("overview"))
    overview_matches = as_dict(overview.get("matches"))
    matches = overview_matches.get("allMatches")

    if isinstance(matches, list) and matches:
        return [m for m in matches if isinstance(m, dict)]

    return []


def is_finished(match: dict[str, Any]) -> bool:
    status = as_dict(match.get("status"))

    if status.get("cancelled") is True:
        return False

    if status.get("finished") is True:
        return True

    reason = as_dict(status.get("reason"))

    short = str(
        reason.get("short")
        or status.get("short")
        or ""
    ).upper()

    long = str(
        reason.get("long")
        or ""
    ).lower()

    if short in {"FT", "AET", "AP"}:
        return True

    if "full-time" in long or "finished" in long:
        return True

    # Todas las temporadas <= 2022 ya han terminado.
    # Si tiene ID + equipos y no está cancelado/no iniciado,
    # aceptamos el partido como histórico.
    if (
        match.get("id") is not None
        and isinstance(match.get("home"), dict)
        and isinstance(match.get("away"), dict)
        and not status.get("notStarted")
    ):
        return True

    return False


# ---------------------------------------------------------
# STATS / ÁRBITRO
# ---------------------------------------------------------

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


def numeric(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return float(int(value))

    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)

    m = re.search(
        r"-?\d+(?:[.,]\d+)?",
        str(value).strip(),
    )

    if not m:
        return None

    try:
        return float(m.group(0).replace(",", "."))
    except Exception:
        return None


def collect_stat_pairs(
    node: Any,
    out: list[tuple[str, Any, Any]],
) -> None:
    if isinstance(node, dict):
        values = node.get("stats")
        name = node.get("key") or node.get("title") or node.get("name")

        if (
            name
            and isinstance(values, list)
            and len(values) >= 2
        ):
            out.append(
                (
                    normalize_name(name),
                    values[0],
                    values[1],
                )
            )

        for value in node.values():
            collect_stat_pairs(value, out)

    elif isinstance(node, list):
        for value in node:
            collect_stat_pairs(value, out)


def all_stat_pairs(
    detail: dict[str, Any],
) -> list[tuple[str, Any, Any]]:
    content = as_dict(detail.get("content"))
    stats = as_dict(content.get("stats"))
    periods = as_dict(stats.get("Periods"))
    all_period = as_dict(periods.get("All"))

    pairs: list[tuple[str, Any, Any]] = []
    collect_stat_pairs(all_period, pairs)

    if not pairs:
        collect_stat_pairs(detail, pairs)

    return pairs


def find_stat(
    pairs: list[tuple[str, Any, Any]],
    aliases: set[str],
) -> tuple[float | None, float | None]:
    for name, home, away in pairs:
        if name in aliases:
            return numeric(home), numeric(away)

    for name, home, away in pairs:
        for alias in aliases:
            if alias in name or name in alias:
                return numeric(home), numeric(away)

    return None, None


def extract_referee(
    detail: dict[str, Any],
) -> str | None:
    content = as_dict(detail.get("content"))
    match_facts = as_dict(content.get("matchFacts"))
    info = as_dict(match_facts.get("infoBox"))

    referee = info.get("Referee") or info.get("referee")

    if isinstance(referee, str):
        return referee.strip() or None

    if isinstance(referee, dict):
        text = referee.get("text") or referee.get("name")
        if isinstance(text, str):
            return text.strip() or None

    return None


def extract_match_date(
    detail: dict[str, Any],
) -> str | None:
    header = as_dict(detail.get("header"))
    status = as_dict(header.get("status"))

    utc = status.get("utcTime")
    if isinstance(utc, str):
        return utc

    general = as_dict(detail.get("general"))
    value = general.get("matchTimeUTCDate")

    return value if isinstance(value, str) else None


def event_cards(
    detail: dict[str, Any],
) -> tuple[int | None, int | None, int | None, int | None]:
    content = as_dict(detail.get("content"))
    facts = as_dict(content.get("matchFacts"))
    events_box = as_dict(facts.get("events"))
    events = events_box.get("events")

    if not isinstance(events, list):
        return None, None, None, None

    yh = ya = rh = ra = 0
    found = False

    for event in events:
        if not isinstance(event, dict):
            continue

        combined = " ".join(
            str(event.get(k) or "")
            for k in (
                "type",
                "eventType",
                "card",
                "cardType",
                "typeStr",
            )
        ).lower()

        if "yellow" not in combined and "red" not in combined:
            continue

        is_home = event.get("isHome")
        if not isinstance(is_home, bool):
            continue

        found = True

        second_yellow = (
            "second" in combined and "yellow" in combined
        )

        red = "red" in combined or second_yellow
        yellow = "yellow" in combined and not second_yellow

        if is_home:
            if red:
                rh += 1
            elif yellow:
                yh += 1
        else:
            if red:
                ra += 1
            elif yellow:
                ya += 1

    if not found:
        return None, None, None, None

    return yh, ya, rh, ra


def extract_referee_row(
    detail: dict[str, Any],
    match_id: int,
    season: int,
    home_team: str,
    away_team: str,
) -> dict[str, Any] | None:
    referee = extract_referee(detail)

    if not referee:
        return None

    pairs = all_stat_pairs(detail)

    home_fouls, away_fouls = find_stat(
        pairs,
        {"fouls", "fouls_committed", "fk_foul_lost"},
    )

    home_yellow, away_yellow = find_stat(
        pairs,
        {"yellow_cards", "yellow_card", "total_yel_card"},
    )

    home_red, away_red = find_stat(
        pairs,
        {"red_cards", "red_card", "total_red_card"},
    )

    if home_yellow is None and away_yellow is None:
        yh, ya, rh, ra = event_cards(detail)

        if yh is not None:
            home_yellow = float(yh)
            away_yellow = float(ya)
            home_red = float(rh)
            away_red = float(ra)

    return {
        "match_id": match_id,
        "season": season,
        "utc_time": extract_match_date(detail),
        "home_team": home_team,
        "away_team": away_team,
        "referee": referee,

        "home_yellow": home_yellow,
        "away_yellow": away_yellow,
        "total_yellow": (
            home_yellow + away_yellow
            if home_yellow is not None
            and away_yellow is not None
            else None
        ),

        "home_red": home_red,
        "away_red": away_red,
        "total_red": (
            home_red + away_red
            if home_red is not None
            and away_red is not None
            else None
        ),

        "home_fouls": home_fouls,
        "away_fouls": away_fouls,
        "total_fouls": (
            home_fouls + away_fouls
            if home_fouls is not None
            and away_fouls is not None
            else None
        ),
    }


# ---------------------------------------------------------
# SQLITE
# ---------------------------------------------------------

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS referee_match_history (
            match_id INTEGER PRIMARY KEY,
            season INTEGER,
            utc_time TEXT,
            home_team TEXT,
            away_team TEXT,
            referee TEXT,
            home_yellow REAL,
            away_yellow REAL,
            total_yellow REAL,
            home_red REAL,
            away_red REAL,
            total_red REAL,
            home_fouls REAL,
            away_fouls REAL,
            total_fouls REAL,
            source TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS referee_summary (
            referee TEXT PRIMARY KEY,
            matches INTEGER,
            first_season INTEGER,
            last_season INTEGER,

            avg_yellow REAL,
            avg_red REAL,
            avg_fouls REAL,

            last10_yellow REAL,
            last20_yellow REAL,
            last10_red REAL,
            last20_red REAL,
            last10_fouls REAL,
            last20_fouls REAL,

            since2023_matches INTEGER,
            since2023_avg_yellow REAL,
            since2023_avg_red REAL,
            since2023_avg_fouls REAL,

            home_yellow_avg REAL,
            away_yellow_avg REAL,

            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def upsert_referee_match(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    source: str,
) -> None:
    conn.execute(
        """
        INSERT INTO referee_match_history (
            match_id, season, utc_time,
            home_team, away_team, referee,
            home_yellow, away_yellow, total_yellow,
            home_red, away_red, total_red,
            home_fouls, away_fouls, total_fouls,
            source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(match_id) DO UPDATE SET
            season=excluded.season,
            utc_time=excluded.utc_time,
            home_team=excluded.home_team,
            away_team=excluded.away_team,
            referee=excluded.referee,
            home_yellow=excluded.home_yellow,
            away_yellow=excluded.away_yellow,
            total_yellow=excluded.total_yellow,
            home_red=excluded.home_red,
            away_red=excluded.away_red,
            total_red=excluded.total_red,
            home_fouls=excluded.home_fouls,
            away_fouls=excluded.away_fouls,
            total_fouls=excluded.total_fouls,
            source=excluded.source,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            row["match_id"],
            row["season"],
            row["utc_time"],
            row["home_team"],
            row["away_team"],
            row["referee"],
            row["home_yellow"],
            row["away_yellow"],
            row["total_yellow"],
            row["home_red"],
            row["away_red"],
            row["total_red"],
            row["home_fouls"],
            row["away_fouls"],
            row["total_fouls"],
            source,
        ),
    )


def seed_2023_plus(
    conn: sqlite3.Connection,
) -> int:
    rows = conn.execute(
        """
        SELECT
            m.match_id,
            m.season,
            m.utc_time,
            m.home_team,
            m.away_team,
            s.referee,
            s.home_yellow_cards,
            s.away_yellow_cards,
            s.home_red_cards,
            s.away_red_cards,
            s.home_fouls,
            s.away_fouls
        FROM fotmob_matches m
        JOIN fotmob_match_stats s
            ON s.match_id = m.match_id
        WHERE m.season >= 2023
          AND m.finished = 1
          AND m.cancelled = 0
          AND COALESCE(m.valid_for_model, 1) = 1
          AND s.referee IS NOT NULL
        """
    ).fetchall()

    for row in rows:
        (
            match_id,
            season,
            utc_time,
            home_team,
            away_team,
            referee,
            home_yellow,
            away_yellow,
            home_red,
            away_red,
            home_fouls,
            away_fouls,
        ) = row

        data = {
            "match_id": match_id,
            "season": season,
            "utc_time": utc_time,
            "home_team": home_team,
            "away_team": away_team,
            "referee": referee,

            "home_yellow": home_yellow,
            "away_yellow": away_yellow,
            "total_yellow": (
                home_yellow + away_yellow
                if home_yellow is not None
                and away_yellow is not None
                else None
            ),

            "home_red": home_red,
            "away_red": away_red,
            "total_red": (
                home_red + away_red
                if home_red is not None
                and away_red is not None
                else None
            ),

            "home_fouls": home_fouls,
            "away_fouls": away_fouls,
            "total_fouls": (
                home_fouls + away_fouls
                if home_fouls is not None
                and away_fouls is not None
                else None
            ),
        }

        upsert_referee_match(
            conn,
            data,
            "fotmob_2023_plus",
        )

    conn.commit()
    return len(rows)


def avg(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def rebuild_summary(
    conn: sqlite3.Connection,
) -> None:
    conn.execute("DELETE FROM referee_summary")

    referees = [
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT referee
            FROM referee_match_history
            WHERE referee IS NOT NULL
              AND TRIM(referee) <> ''
            """
        ).fetchall()
    ]

    for referee in referees:
        rows = conn.execute(
            """
            SELECT
                season,
                utc_time,
                home_yellow,
                away_yellow,
                total_yellow,
                total_red,
                total_fouls
            FROM referee_match_history
            WHERE referee = ?
            ORDER BY
                CASE WHEN utc_time IS NULL THEN 1 ELSE 0 END,
                utc_time,
                match_id
            """,
            (referee,),
        ).fetchall()

        if not rows:
            continue

        seasons = [
            int(r[0])
            for r in rows
            if r[0] is not None
        ]

        yellow = [r[4] for r in rows]
        red = [r[5] for r in rows]
        fouls = [r[6] for r in rows]

        since2023 = [
            r for r in rows
            if r[0] is not None
            and int(r[0]) >= 2023
        ]

        conn.execute(
            """
            INSERT INTO referee_summary (
                referee,
                matches,
                first_season,
                last_season,
                avg_yellow,
                avg_red,
                avg_fouls,
                last10_yellow,
                last20_yellow,
                last10_red,
                last20_red,
                last10_fouls,
                last20_fouls,
                since2023_matches,
                since2023_avg_yellow,
                since2023_avg_red,
                since2023_avg_fouls,
                home_yellow_avg,
                away_yellow_avg
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                referee,
                len(rows),
                min(seasons) if seasons else None,
                max(seasons) if seasons else None,

                avg(yellow),
                avg(red),
                avg(fouls),

                avg(yellow[-10:]),
                avg(yellow[-20:]),
                avg(red[-10:]),
                avg(red[-20:]),
                avg(fouls[-10:]),
                avg(fouls[-20:]),

                len(since2023),
                avg([r[4] for r in since2023]),
                avg([r[5] for r in since2023]),
                avg([r[6] for r in since2023]),

                avg([r[2] for r in rows]),
                avg([r[3] for r in rows]),
            ),
        )

    conn.commit()


def cache_path(
    season: int,
    match_id: int,
) -> Path:
    return (
        CACHE_DIR
        / str(season)
        / f"{match_id}.json"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Vuelve a descargar incluso la caché existente.",
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Descarga solo una temporada histórica.",
    )

    args = parser.parse_args()

    print()
    print("=" * 72)
    print("ELITESERIEN EDGE PRO - HISTORICO DE ARBITROS V2")
    print("=" * 72)

    session = make_session()

    try:
        base = get_league_season(session, 2022)
        available = []

        for value in base.get("allAvailableSeasons", []):
            try:
                available.append(int(str(value)[:4]))
            except Exception:
                pass

        historical = sorted(
            s for s in set(available)
            if s <= HISTORICAL_END_SEASON
        )

        if args.season is not None:
            historical = [
                s for s in historical
                if s == args.season
            ]

    except Exception as exc:
        print("ERROR detectando temporadas:")
        print(exc)
        return 1

    print("Temporadas a procesar:")
    print(historical)
    print()

    with sqlite3.connect(DB_PATH) as conn:
        init_db(conn)

        seeded = seed_2023_plus(conn)
        print(f"Partidos 2023+ reutilizados: {seeded}")
        print()

        total_new = 0
        cache_hits = 0
        errors = 0
        without_referee = 0

        for season in historical:
            print(f"[{season}] Descargando calendario...")

            try:
                payload = get_league_season(
                    session,
                    season,
                )
            except Exception as exc:
                print(f"  ERROR calendario: {exc}")
                continue

            matches_all = get_matches(payload)

            matches = [
                m
                for m in matches_all
                if is_finished(m)
            ]

            print(
                f"  Partidos encontrados: {len(matches_all)} | "
                f"válidos/finalizados: {len(matches)}"
            )

            for i, match in enumerate(matches, start=1):
                match_id = int(match["id"])

                home = as_dict(
                    match.get("home")
                ).get("name") or "?"

                away = as_dict(
                    match.get("away")
                ).get("name") or "?"

                path = cache_path(
                    season,
                    match_id,
                )

                detail = None

                if not args.refresh and path.exists():
                    try:
                        detail = json.loads(
                            path.read_text(
                                encoding="utf-8"
                            )
                        )
                        cache_hits += 1
                    except Exception:
                        detail = None

                if detail is None:
                    try:
                        detail = get_match_detail(
                            session,
                            match_id,
                        )

                        path.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        path.write_text(
                            json.dumps(
                                detail,
                                ensure_ascii=False,
                            ),
                            encoding="utf-8",
                        )

                        total_new += 1
                        time.sleep(REQUEST_DELAY)

                    except Exception as exc:
                        errors += 1

                        print(
                            f"  ERROR {match_id} "
                            f"{home}-{away}: {exc}"
                        )

                        continue

                try:
                    row = extract_referee_row(
                        detail,
                        match_id,
                        season,
                        home,
                        away,
                    )

                    if row is None:
                        without_referee += 1
                    else:
                        upsert_referee_match(
                            conn,
                            row,
                            "fotmob_historical",
                        )

                except Exception as exc:
                    errors += 1

                    print(
                        f"  ERROR parseando "
                        f"{match_id} {home}-{away}: {exc}"
                    )

                if i % 25 == 0:
                    conn.commit()

                    print(
                        f"  [{i}/{len(matches)}] "
                        f"{home} - {away}"
                    )

            conn.commit()

        print()
        print("Construyendo resumen de árbitros...")
        rebuild_summary(conn)

        total_history = conn.execute(
            """
            SELECT COUNT(*)
            FROM referee_match_history
            """
        ).fetchone()[0]

        total_refs = conn.execute(
            """
            SELECT COUNT(*)
            FROM referee_summary
            """
        ).fetchone()[0]

        min_season, max_season = conn.execute(
            """
            SELECT MIN(season), MAX(season)
            FROM referee_match_history
            """
        ).fetchone()

        cards_cov = conn.execute(
            """
            SELECT COUNT(*)
            FROM referee_match_history
            WHERE total_yellow IS NOT NULL
            """
        ).fetchone()[0]

        fouls_cov = conn.execute(
            """
            SELECT COUNT(*)
            FROM referee_match_history
            WHERE total_fouls IS NOT NULL
            """
        ).fetchone()[0]

        top_refs = conn.execute(
            """
            SELECT
                referee,
                matches,
                ROUND(avg_yellow, 2),
                ROUND(last20_yellow, 2),
                ROUND(avg_fouls, 2)
            FROM referee_summary
            ORDER BY matches DESC
            LIMIT 10
            """
        ).fetchall()

    print()
    print("=" * 72)
    print("HISTORICO DE ARBITROS V2 COMPLETADO")
    print("=" * 72)

    print(f"Partidos árbitro guardados: {total_history}")
    print(f"Árbitros distintos: {total_refs}")
    print(f"Rango temporadas: {min_season} -> {max_season}")
    print(f"Partidos con amarillas: {cards_cov}")
    print(f"Partidos con faltas: {fouls_cov}")
    print(f"Descargas nuevas: {total_new}")
    print(f"Reutilizados de caché: {cache_hits}")
    print(f"Sin árbitro disponible: {without_referee}")
    print(f"Errores: {errors}")

    print()
    print("Árbitros con más partidos:")

    for referee, matches, avg_y, last20_y, avg_f in top_refs:
        print(
            f"  {referee:<28} "
            f"{matches:>3} partidos | "
            f"amarillas {avg_y} | "
            f"últ20 {last20_y} | "
            f"faltas {avg_f}"
        )

    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
