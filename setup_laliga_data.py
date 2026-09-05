from __future__ import annotations

import hashlib
import io
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "laliga.db"
CACHE_DIR = DATA_DIR / "laliga_raw"

HISTORICAL_SEASONS = {
    2020: "2021",
    2021: "2122",
    2022: "2223",
    2023: "2324",
    2024: "2425",
    2025: "2526",
    2026: "2627",
}

SP1_URL = "https://www.football-data.co.uk/mmz4281/{folder}/SP1.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def stable_int(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:15]
    return int(digest, 16)


def team_id(name: str) -> int:
    return stable_int("team|" + str(name).strip().lower())


def match_id(season: int, date, home: str, away: str) -> int:
    key = f"{season}|{date}|{home}|{away}"
    return stable_int("match|" + key)


def num(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def first_existing(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def combine_datetime(df):
    date_col = first_existing(df, ["Date", "date"])
    time_col = first_existing(df, ["Time", "time"])

    if date_col is None:
        return pd.Series(pd.NaT, index=df.index)

    date_text = df[date_col].astype(str)

    if time_col is not None:
        time_text = df[time_col].fillna("12:00").astype(str)
        combined = date_text + " " + time_text
    else:
        combined = date_text + " 12:00"

    try:
        dt = pd.to_datetime(
            combined,
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
    except TypeError:
        dt = pd.to_datetime(
            combined,
            dayfirst=True,
            errors="coerce",
        )

    # Football-Data gives local kickoff times for the competition.
    try:
        dt = dt.dt.tz_localize(
            "Europe/Madrid",
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    except TypeError:
        pass

    try:
        dt = dt.dt.tz_convert("UTC")
    except (TypeError, AttributeError):
        pass

    return dt


def download_csv(url, cache_path):
    headers = {
        "User-Agent": "Mozilla/5.0 EliteserienEdgePro-MultiLeague/1.0"
    }

    r = requests.get(
        url,
        headers=headers,
        timeout=40,
    )
    r.raise_for_status()

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_path.write_bytes(r.content)

    return pd.read_csv(
        io.BytesIO(r.content),
        low_memory=False,
    )


def download_history():
    frames = []

    for season, folder in HISTORICAL_SEASONS.items():
        url = SP1_URL.format(folder=folder)
        cache = CACHE_DIR / f"SP1_{season}_{season+1}.csv"

        print(
            f"    {season}/{season+1}: ",
            end="",
            flush=True,
        )

        try:
            df = download_csv(url, cache)
            df["season"] = season
            frames.append(df)
            print(f"{len(df)} partidos")
        except Exception as exc:
            print(f"ERROR -> {exc}")

    if not frames:
        raise RuntimeError(
            "No se pudo descargar ninguna temporada de LaLiga."
        )

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def prepare_history(raw):
    home_col = first_existing(
        raw,
        ["HomeTeam", "Home"],
    )
    away_col = first_existing(
        raw,
        ["AwayTeam", "Away"],
    )
    hg_col = first_existing(
        raw,
        ["FTHG", "HG"],
    )
    ag_col = first_existing(
        raw,
        ["FTAG", "AG"],
    )

    if not all([home_col, away_col, hg_col, ag_col]):
        raise RuntimeError(
            "El formato SP1 ha cambiado y faltan columnas básicas."
        )

    df = raw.copy()

    df["utc_dt"] = combine_datetime(df)
    df["home_team"] = df[home_col].astype(str).str.strip()
    df["away_team"] = df[away_col].astype(str).str.strip()
    df["home_goals"] = pd.to_numeric(
        df[hg_col],
        errors="coerce",
    )
    df["away_goals"] = pd.to_numeric(
        df[ag_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "season",
            "utc_dt",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    df["season"] = df["season"].astype(int)
    df["match_id"] = [
        match_id(s, d, h, a)
        for s, d, h, a in zip(
            df["season"],
            df["utc_dt"].astype(str),
            df["home_team"],
            df["away_team"],
        )
    ]

    df["home_team_id"] = df["home_team"].map(team_id)
    df["away_team_id"] = df["away_team"].map(team_id)

    return df


def odds_series(df, candidates):
    for col in candidates:
        if col in df.columns:
            s = pd.to_numeric(
                df[col],
                errors="coerce",
            )
            if int((s > 1.0).sum()) > 0:
                return s, col

    return (
        pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        ),
        None,
    )


def build_tables(hist):
    referee_col = first_existing(
        hist,
        ["Referee", "referee"],
    )

    odds_home, odds_home_src = odds_series(
        hist,
        ["AvgCH", "B365CH", "PSCH", "MaxCH", "AvgH", "B365H", "PSH", "MaxH"],
    )
    odds_draw, odds_draw_src = odds_series(
        hist,
        ["AvgCD", "B365CD", "PSCD", "MaxCD", "AvgD", "B365D", "PSD", "MaxD"],
    )
    odds_away, odds_away_src = odds_series(
        hist,
        ["AvgCA", "B365CA", "PSCA", "MaxCA", "AvgA", "B365A", "PSA", "MaxA"],
    )

    matches = pd.DataFrame(
        {
            "match_id": hist["match_id"],
            "season": hist["season"],
            "round": np.nan,
            "utc_time": hist["utc_dt"].astype(str),
            "date": hist["utc_dt"].dt.strftime("%Y-%m-%d"),
            "home_team_id": hist["home_team_id"],
            "home_team": hist["home_team"],
            "away_team_id": hist["away_team_id"],
            "away_team": hist["away_team"],
            "home_goals": hist["home_goals"],
            "away_goals": hist["away_goals"],
            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,
            "finished": 1,
            "cancelled": 0,
            "valid_for_model": 1,
            "status": "Finished",
        }
    )

    stats = pd.DataFrame(
        {
            "match_id": hist["match_id"],
            "home_xg": np.nan,
            "away_xg": np.nan,
            "home_possession": np.nan,
            "away_possession": np.nan,
            "home_shots": num(hist, "HS"),
            "away_shots": num(hist, "AS"),
            "home_shots_on_target": num(hist, "HST"),
            "away_shots_on_target": num(hist, "AST"),
            "home_corners": num(hist, "HC"),
            "away_corners": num(hist, "AC"),
            "home_fouls": num(hist, "HF"),
            "away_fouls": num(hist, "AF"),
            "home_offsides": np.nan,
            "away_offsides": np.nan,
            "home_yellow_cards": num(hist, "HY"),
            "away_yellow_cards": num(hist, "AY"),
            "home_red_cards": num(hist, "HR"),
            "away_red_cards": num(hist, "AR"),
            "referee": (
                hist[referee_col].astype(object)
                if referee_col
                else np.nan
            ),
            "stadium": np.nan,
            "attendance": np.nan,
        }
    )

    refs = pd.DataFrame(
        {
            "match_id": hist["match_id"],
            "season": hist["season"],
            "utc_time": hist["utc_dt"].astype(str),
            "referee": (
                hist[referee_col].astype(object)
                if referee_col
                else np.nan
            ),
            "home_yellow": num(hist, "HY"),
            "away_yellow": num(hist, "AY"),
            "total_yellow": num(hist, "HY") + num(hist, "AY"),
            "home_red": num(hist, "HR"),
            "away_red": num(hist, "AR"),
            "total_red": num(hist, "HR") + num(hist, "AR"),
            "home_fouls": num(hist, "HF"),
            "away_fouls": num(hist, "AF"),
            "total_fouls": num(hist, "HF") + num(hist, "AF"),
        }
    )

    refs = refs.dropna(
        subset=["referee"]
    ).copy()

    print()
    print("    Cuotas históricas elegidas:")
    print(
        f"      1: {odds_home_src or 'sin datos'} | "
        f"X: {odds_draw_src or 'sin datos'} | "
        f"2: {odds_away_src or 'sin datos'}"
    )

    return matches, stats, refs


def download_upcoming():
    cache = CACHE_DIR / "fixtures_latest.csv"

    try:
        raw = download_csv(
            FIXTURES_URL,
            cache,
        )
    except Exception as exc:
        print(
            f"    Fixtures: no disponibles ahora ({exc})"
        )
        return pd.DataFrame()

    div_col = first_existing(
        raw,
        ["Div", "League", "league"],
    )

    if div_col:
        mask = raw[div_col].astype(str).str.upper().isin(
            ["SP1", "LA LIGA", "LALIGA"]
        )
        raw = raw[mask].copy()

    home_col = first_existing(
        raw,
        ["HomeTeam", "Home"],
    )
    away_col = first_existing(
        raw,
        ["AwayTeam", "Away"],
    )

    if not home_col or not away_col or raw.empty:
        print(
            "    Fixtures: el CSV no contiene partidos SP1 detectables."
        )
        return pd.DataFrame()

    raw["utc_dt"] = combine_datetime(raw)
    raw["home_team"] = raw[home_col].astype(str).str.strip()
    raw["away_team"] = raw[away_col].astype(str).str.strip()

    now = pd.Timestamp.now(tz="UTC")
    raw = raw[
        raw["utc_dt"] >= now - pd.Timedelta(hours=6)
    ].copy()

    if raw.empty:
        print("    Fixtures futuros SP1: 0")
        return pd.DataFrame()

    # season = año de inicio de temporada (julio-diciembre -> mismo año,
    # enero-junio -> año anterior).
    local_dt = raw["utc_dt"].dt.tz_convert(
        "Europe/Madrid"
    )
    raw["season"] = np.where(
        local_dt.dt.month >= 7,
        local_dt.dt.year,
        local_dt.dt.year - 1,
    ).astype(int)

    raw["match_id"] = [
        match_id(s, d, h, a)
        for s, d, h, a in zip(
            raw["season"],
            raw["utc_dt"].astype(str),
            raw["home_team"],
            raw["away_team"],
        )
    ]

    raw["home_team_id"] = raw["home_team"].map(team_id)
    raw["away_team_id"] = raw["away_team"].map(team_id)

    fixtures = pd.DataFrame(
        {
            "match_id": raw["match_id"],
            "season": raw["season"],
            "round": np.nan,
            "utc_time": raw["utc_dt"].astype(str),
            "date": raw["utc_dt"].dt.strftime("%Y-%m-%d"),
            "home_team_id": raw["home_team_id"],
            "home_team": raw["home_team"],
            "away_team_id": raw["away_team_id"],
            "away_team": raw["away_team"],
            "home_goals": np.nan,
            "away_goals": np.nan,
            "odds_home": np.nan,
            "odds_draw": np.nan,
            "odds_away": np.nan,
            "finished": 0,
            "cancelled": 0,
            "valid_for_model": 1,
            "status": "Scheduled",
        }
    )

    print(
        f"    Fixtures futuros SP1: {len(fixtures)}"
    )

    return fixtures


def referee_summary(refs):
    if refs.empty:
        return pd.DataFrame()

    grp = refs.groupby("referee", dropna=True)

    summary = grp.agg(
        matches=("match_id", "count"),
        avg_yellow=("total_yellow", "mean"),
        avg_home_yellow=("home_yellow", "mean"),
        avg_away_yellow=("away_yellow", "mean"),
        avg_red=("total_red", "mean"),
        avg_fouls=("total_fouls", "mean"),
        first_season=("season", "min"),
        last_season=("season", "max"),
    ).reset_index()

    return summary.sort_values(
        ["matches", "avg_yellow"],
        ascending=[False, False],
    )


def main():
    print()
    print("=" * 80)
    print("LALIGA EDGE PRO - PHASE 1 DATA SETUP")
    print("=" * 80)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("1/5 Descargando LaLiga 2020/21 -> 2026/27...")
    raw = download_history()

    print()
    print("2/5 Normalizando histórico...")
    hist = prepare_history(raw)

    print(
        f"    Partidos históricos válidos: {len(hist)}"
    )
    print(
        f"    Rango: {hist['season'].min()} -> {hist['season'].max()}"
    )
    print(
        f"    Equipos distintos: "
        f"{len(set(hist['home_team']) | set(hist['away_team']))}"
    )

    print()
    print("3/5 Creando tablas compatibles con nuestros modelos...")
    matches, stats, refs = build_tables(hist)

    print()
    print("4/5 Buscando próximos partidos...")
    fixtures = download_upcoming()

    all_matches = pd.concat(
        [matches, fixtures],
        ignore_index=True,
    ).drop_duplicates(
        subset=["match_id"],
        keep="first",
    )

    print()
    print("5/5 Guardando laliga.db...")

    with sqlite3.connect(DB_PATH) as conn:
        raw.to_sql(
            "football_data_raw",
            conn,
            if_exists="replace",
            index=False,
        )

        matches.to_sql(
            "matches",
            conn,
            if_exists="replace",
            index=False,
        )

        all_matches.to_sql(
            "fotmob_matches",
            conn,
            if_exists="replace",
            index=False,
        )

        stats.to_sql(
            "fotmob_match_stats",
            conn,
            if_exists="replace",
            index=False,
        )

        refs.to_sql(
            "referee_match_history",
            conn,
            if_exists="replace",
            index=False,
        )

        referee_summary(refs).to_sql(
            "referee_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.commit()

    print()
    print("=" * 80)
    print("LALIGA DATA READY")
    print("=" * 80)
    print(f"DB: {DB_PATH}")
    print(
        f"Histórico total: {len(matches)} | "
        f"Modelado equipos: se usará 2023+"
    )
    print(
        f"Árbitros 2020+: {refs['referee'].nunique() if not refs.empty else 0}"
    )
    print(
        f"Corners con datos: "
        f"{int(stats['home_corners'].notna().sum())}/{len(stats)}"
    )
    print(
        f"Amarillas con datos: "
        f"{int(stats['home_yellow_cards'].notna().sum())}/{len(stats)}"
    )
    print(
        f"Tiros a puerta con datos: "
        f"{int(stats['home_shots_on_target'].notna().sum())}/{len(stats)}"
    )
    print(
        f"Próximos fixtures guardados: {len(fixtures)}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
