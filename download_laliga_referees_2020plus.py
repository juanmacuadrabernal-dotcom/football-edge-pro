from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "laliga.db"
RAW_DIR = BASE_DIR / "data" / "laliga_raw" / "fotmob_referees"

LEAGUE_ID = 87
COUNTRY_CODE = "ESP"

LEAGUE_URL = "https://www.fotmob.com/api/data/leagues"
DETAIL_URLS = [
    "https://www.fotmob.com/api/matchDetails",
    "https://www.fotmob.com/api/data/matchDetails",
]

SEASONS = list(range(2020, 2027))

REQUEST_DELAY = 0.35
TIMEOUT = 30
RETRIES = 4

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


def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def get_json(session, url, params):
    last_error = None

    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(
                url,
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
            time.sleep(attempt * 1.5)

    raise RuntimeError(
        str(last_error) if last_error else "Error HTTP desconocido"
    )


def season_candidates(start_year):
    return [
        f"{start_year}/{start_year + 1}",
        f"{start_year}-{start_year + 1}",
        str(start_year),
    ]


def extract_matches(payload):
    found = {}

    def walk(node):
        if isinstance(node, dict):
            mid = node.get("id")
            home = node.get("home")
            away = node.get("away")
            status = node.get("status")

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


def finished_matches_for_season(session, start_year):
    best = []
    best_label = None
    best_payload = None

    for season_label in season_candidates(start_year):
        try:
            payload = get_json(
                session,
                LEAGUE_URL,
                {
                    "id": LEAGUE_ID,
                    "ccode3": COUNTRY_CODE,
                    "season": season_label,
                },
            )
        except Exception:
            continue

        matches = extract_matches(payload)

        finished = []
        for m in matches:
            status = m.get("status", {}) or {}

            if status.get("finished"):
                reason = str(
                    status.get("reason") or ""
                ).lower()

                if "abandon" in reason or "cancel" in reason:
                    continue

                finished.append(m)

        if len(finished) > len(best):
            best = finished
            best_label = season_label
            best_payload = payload

        if len(finished) >= 300:
            break

    if not best:
        raise RuntimeError(
            f"No encontré partidos finalizados para {start_year}/{start_year+1}"
        )

    season_dir = RAW_DIR / "seasons"
    season_dir.mkdir(parents=True, exist_ok=True)

    (season_dir / f"laliga_{start_year}_{start_year+1}.json").write_text(
        json.dumps(
            best_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return best_label, best


def detail_path(season, match_id):
    p = RAW_DIR / "matches" / str(season)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{match_id}.json"


def fetch_match_detail(session, season, match_id):
    path = detail_path(season, match_id)

    if path.exists():
        try:
            return json.loads(
                path.read_text(encoding="utf-8")
            ), False
        except Exception:
            pass

    last_error = None

    for url in DETAIL_URLS:
        try:
            payload = get_json(
                session,
                url,
                {"matchId": int(match_id)},
            )

            path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            time.sleep(REQUEST_DELAY)
            return payload, True

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"matchId={match_id}: {last_error}"
    )


def clean_text(value):
    if value is None:
        return None

    if isinstance(value, dict):
        for key in ("text", "name", "title", "value"):
            if key in value and value[key]:
                return str(value[key]).strip()
        return None

    s = str(value).strip()
    return s or None


def find_referee(payload):
    candidates = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                key_l = str(key).lower()
                new_path = f"{path}.{key_l}"

                if "referee" in key_l or "referee" in new_path:
                    text = clean_text(value)
                    if text:
                        candidates.append(text)

                walk(value, new_path)

        elif isinstance(node, list):
            for value in node:
                walk(value, path)

    walk(payload)

    for candidate in candidates:
        # Evita serializaciones de objetos enteros.
        if (
            candidate
            and len(candidate) <= 100
            and not candidate.startswith("{")
            and not candidate.startswith("[")
        ):
            return candidate

    return None


def normalize_stat_key(value):
    s = str(value or "").strip().lower()
    s = (
        s.replace("å", "a")
        .replace("ø", "o")
        .replace("æ", "ae")
    )
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def numeric_pair(value):
    if isinstance(value, dict):
        # Some FotMob variants use home/away dicts.
        home = value.get("home")
        away = value.get("away")

        if home is not None or away is not None:
            return to_num(home), to_num(away)

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return to_num(value[0]), to_num(value[1])

    return None, None


def to_num(value):
    if value is None:
        return np.nan

    if isinstance(value, dict):
        for key in ("value", "stat", "text"):
            if key in value:
                return to_num(value[key])
        return np.nan

    if isinstance(value, str):
        value = value.replace("%", "").strip()

    try:
        return float(value)
    except Exception:
        return np.nan


def find_stat_pair(payload, aliases):
    aliases = {
        normalize_stat_key(x)
        for x in aliases
    }

    matches = []

    def walk(node):
        if isinstance(node, dict):
            title = (
                node.get("title")
                or node.get("name")
                or node.get("key")
                or node.get("label")
            )

            key = normalize_stat_key(title)

            if key in aliases:
                for value_key in ("stats", "values", "value"):
                    if value_key in node:
                        h, a = numeric_pair(node[value_key])

                        if not (
                            pd.isna(h)
                            and pd.isna(a)
                        ):
                            matches.append((h, a))

            # Direct key/value variants.
            for raw_key, value in node.items():
                key2 = normalize_stat_key(raw_key)

                if key2 in aliases:
                    h, a = numeric_pair(value)

                    if not (
                        pd.isna(h)
                        and pd.isna(a)
                    ):
                        matches.append((h, a))

            for value in node.values():
                walk(value)

        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)

    if matches:
        return matches[0]

    return np.nan, np.nan


def header_teams(payload):
    header = payload.get("header", {}) or {}
    teams = header.get("teams")

    if isinstance(teams, list) and len(teams) >= 2:
        return teams[0], teams[1]

    general = payload.get("general", {}) or {}

    home = general.get("homeTeam", {}) or {}
    away = general.get("awayTeam", {}) or {}

    return home, away


def kickoff(payload, fallback_match):
    general = payload.get("general", {}) or {}
    header = payload.get("header", {}) or {}
    status = header.get("status", {}) or {}
    fallback_status = fallback_match.get("status", {}) or {}

    value = (
        general.get("matchTimeUTCDate")
        or status.get("utcTime")
        or fallback_status.get("utcTime")
    )

    ts = pd.to_datetime(
        value,
        utc=True,
        errors="coerce",
    )

    if pd.isna(ts):
        return None

    return ts


def parse_match(season, fixture, payload):
    match_id = int(fixture["id"])

    header_home, header_away = header_teams(payload)

    fixture_home = fixture.get("home", {}) or {}
    fixture_away = fixture.get("away", {}) or {}

    home_team = (
        header_home.get("name")
        or fixture_home.get("name")
    )

    away_team = (
        header_away.get("name")
        or fixture_away.get("name")
    )

    ref = find_referee(payload)

    hy, ay = find_stat_pair(
        payload,
        [
            "yellow cards",
            "yellowcards",
            "amarillas",
            "tarjetas amarillas",
        ],
    )

    hr, ar = find_stat_pair(
        payload,
        [
            "red cards",
            "redcards",
            "rojas",
            "tarjetas rojas",
        ],
    )

    hf, af = find_stat_pair(
        payload,
        [
            "fouls committed",
            "fouls",
            "foulscommitted",
            "faltas",
            "faltas cometidas",
        ],
    )

    ko = kickoff(payload, fixture)

    return {
        "match_id": match_id,
        "season": int(season),
        "utc_time": (
            ko.isoformat()
            if ko is not None
            else None
        ),
        "home_team": home_team,
        "away_team": away_team,
        "referee": ref,
        "home_yellow": hy,
        "away_yellow": ay,
        "total_yellow": (
            hy + ay
            if pd.notna(hy) and pd.notna(ay)
            else np.nan
        ),
        "home_red": hr,
        "away_red": ar,
        "total_red": (
            hr + ar
            if pd.notna(hr) and pd.notna(ar)
            else np.nan
        ),
        "home_fouls": hf,
        "away_fouls": af,
        "total_fouls": (
            hf + af
            if pd.notna(hf) and pd.notna(af)
            else np.nan
        ),
    }


def build_summary(history):
    valid = history.dropna(
        subset=["referee"]
    ).copy()

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "referee",
                "matches",
                "avg_yellow",
                "avg_home_yellow",
                "avg_away_yellow",
                "avg_red",
                "avg_fouls",
                "last10_yellow",
                "last20_yellow",
                "first_season",
                "last_season",
                "league_avg_yellow",
                "yellow_delta_vs_league",
                "profile",
            ]
        )

    valid["utc_time"] = pd.to_datetime(
        valid["utc_time"],
        utc=True,
        errors="coerce",
    )

    league_avg = float(
        pd.to_numeric(
            valid["total_yellow"],
            errors="coerce",
        ).mean()
    )

    rows = []

    for referee, g in valid.groupby("referee"):
        g = g.sort_values("utc_time")

        total_y = pd.to_numeric(
            g["total_yellow"],
            errors="coerce",
        )

        avg_y = float(total_y.mean())
        delta = avg_y - league_avg

        if delta >= 0.60:
            profile = "MUY TARJETERO"
        elif delta >= 0.25:
            profile = "TARJETERO"
        elif delta <= -0.60:
            profile = "MUY PERMISIVO"
        elif delta <= -0.25:
            profile = "PERMISIVO"
        else:
            profile = "NEUTRO"

        rows.append(
            {
                "referee": referee,
                "matches": int(len(g)),
                "avg_yellow": avg_y,
                "avg_home_yellow": float(
                    pd.to_numeric(
                        g["home_yellow"],
                        errors="coerce",
                    ).mean()
                ),
                "avg_away_yellow": float(
                    pd.to_numeric(
                        g["away_yellow"],
                        errors="coerce",
                    ).mean()
                ),
                "avg_red": float(
                    pd.to_numeric(
                        g["total_red"],
                        errors="coerce",
                    ).mean()
                ),
                "avg_fouls": float(
                    pd.to_numeric(
                        g["total_fouls"],
                        errors="coerce",
                    ).mean()
                ),
                "last10_yellow": float(
                    total_y.tail(10).mean()
                ),
                "last20_yellow": float(
                    total_y.tail(20).mean()
                ),
                "first_season": int(g["season"].min()),
                "last_season": int(g["season"].max()),
                "league_avg_yellow": league_avg,
                "yellow_delta_vs_league": delta,
                "profile": profile,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["matches", "avg_yellow"],
        ascending=[False, False],
    )


def save_progress(rows):
    if not rows:
        return

    df = pd.DataFrame(rows)

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(
            "referee_match_history",
            conn,
            if_exists="replace",
            index=False,
        )

        summary = build_summary(df)

        summary.to_sql(
            "referee_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.commit()


def main():
    print()
    print("=" * 84)
    print("LALIGA EDGE PRO - DESCARGA HISTORICA DE ARBITROS 2020+")
    print("=" * 84)
    print(
        "Puedes dejarlo ejecutando. Usa caché y se puede reanudar."
    )

    if not DB_PATH.exists():
        raise SystemExit(
            f"No existe {DB_PATH}. "
            "Ejecuta esto desde la carpeta del proyecto."
        )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = make_session()
    rows = []

    total_downloaded = 0
    total_cached = 0
    total_errors = 0
    total_no_ref = 0

    for season in SEASONS:
        print()
        print("=" * 84)
        print(
            f"TEMPORADA {season}/{season + 1}"
        )
        print("=" * 84)

        season_label, fixtures = finished_matches_for_season(
            session,
            season,
        )

        print(
            f"FotMob season={season_label} | "
            f"finalizados={len(fixtures)}"
        )

        for i, fixture in enumerate(
            fixtures,
            start=1,
        ):
            mid = int(fixture["id"])

            try:
                payload, downloaded = fetch_match_detail(
                    session,
                    season,
                    mid,
                )

                if downloaded:
                    total_downloaded += 1
                else:
                    total_cached += 1

                row = parse_match(
                    season,
                    fixture,
                    payload,
                )

                if not row["referee"]:
                    total_no_ref += 1

                rows.append(row)

            except Exception as exc:
                total_errors += 1
                print(
                    f"  ERROR match {mid}: {exc}"
                )

            if i % 25 == 0 or i == len(fixtures):
                refs_so_far = sum(
                    1
                    for r in rows
                    if r.get("referee")
                )

                print(
                    f"  [{i:3d}/{len(fixtures)}] "
                    f"refs={refs_so_far} | "
                    f"nuevas={total_downloaded} | "
                    f"cache={total_cached} | "
                    f"errores={total_errors}"
                )

        # Guardado tras cada temporada para que un corte no pierda trabajo.
        save_progress(rows)

    history = pd.DataFrame(rows)

    if history.empty:
        raise RuntimeError(
            "No se pudo construir referee_match_history."
        )

    summary = build_summary(history)

    with sqlite3.connect(DB_PATH) as conn:
        history.to_sql(
            "referee_match_history",
            conn,
            if_exists="replace",
            index=False,
        )

        summary.to_sql(
            "referee_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.commit()

    print()
    print("=" * 84)
    print("DESCARGA DE ARBITROS COMPLETADA")
    print("=" * 84)
    print(
        f"Partidos guardados: {len(history)}"
    )
    print(
        f"Árbitros distintos: "
        f"{history['referee'].nunique(dropna=True)}"
    )
    print(
        f"Partidos con árbitro: "
        f"{history['referee'].notna().sum()}"
    )
    print(
        f"Partidos con amarillas: "
        f"{history['total_yellow'].notna().sum()}"
    )
    print(
        f"Partidos con faltas: "
        f"{history['total_fouls'].notna().sum()}"
    )
    print(
        f"Descargas nuevas: {total_downloaded}"
    )
    print(
        f"Leídos desde caché: {total_cached}"
    )
    print(
        f"Sin árbitro: {total_no_ref}"
    )
    print(
        f"Errores: {total_errors}"
    )

    if not summary.empty:
        print()
        print("TOP ÁRBITROS POR PARTIDOS:")
        print("-" * 84)

        for r in summary.head(15).itertuples(index=False):
            print(
                f"{r.referee:<30} | "
                f"N={int(r.matches):3d} | "
                f"amarillas={r.avg_yellow:4.2f} | "
                f"last20={r.last20_yellow:4.2f} | "
                f"delta={r.yellow_delta_vs_league:+4.2f} | "
                f"{r.profile}"
            )

    print()
    print(
        "Mañana entrenaremos Cards V3 usando SOLO el historial "
        "del árbitro anterior al kickoff de cada partido."
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
