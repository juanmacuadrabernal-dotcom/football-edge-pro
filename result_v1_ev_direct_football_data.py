from __future__ import annotations

import io
import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "eliteserien.db"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/new/NOR.csv"
OUTCOMES = ["H", "D", "A"]


def normalize_team(value):
    if pd.isna(value):
        return ""

    s = str(value).strip().lower()

    replacements = {
        "ø": "o",
        "å": "a",
        "æ": "ae",
        "ö": "o",
        "ä": "a",
        "ü": "u",
    }

    for a, b in replacements.items():
        s = s.replace(a, b)

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "", s)

    aliases = {
        "bodoglimt": "bodoglimt",
        "bodoeglimt": "bodoglimt",

        "stromsgodset": "stromsgodset",

        "lillestrom": "lillestrom",

        "valerenga": "valerenga",

        "tromso": "tromso",

        "stabaek": "stabaek",

        "aalesund": "aalesund",
        "alesund": "aalesund",

        "hamkam": "hamkam",
        "hamarkameratene": "hamkam",

        "odd": "odd",
        "odds": "odd",
        "oddsballklubb": "odd",

        "haugesund": "haugesund",
        "fkhaugesund": "haugesund",

        "sarpsborg08": "sarpsborg08",
        "sarpsborg": "sarpsborg08",

        "kfum": "kfumoslo",
        "kfumoslo": "kfumoslo",

        "kristiansund": "kristiansund",

        "sandefjord": "sandefjord",

        "fredrikstad": "fredrikstad",

        "mjoendalen": "mjondalen",
        "mjondalen": "mjondalen",

        "rosenborg": "rosenborg",

        "brann": "brann",

        "molde": "molde",

        "viking": "viking",

        "start": "start",

        "jerv": "jerv",

        "bryne": "bryne",
    }

    return aliases.get(s, s)


def season_to_int(value):
    if pd.isna(value):
        return np.nan

    m = re.search(r"(20\d{2})", str(value))

    if not m:
        return np.nan

    return int(m.group(1))


def load_predictions(conn):
    df = pd.read_sql_query(
        """
        SELECT *
        FROM result_v1_backtest_predictions
        ORDER BY season, match_day, match_id
        """,
        conn,
    )

    if df.empty:
        raise RuntimeError(
            "No existe result_v1_backtest_predictions. "
            "Ejecuta primero result_model_v1_fixed.py."
        )

    df["season_key"] = pd.to_numeric(
        df["season"],
        errors="coerce",
    ).astype("Int64")

    df["home_key"] = df["home_team"].map(normalize_team)
    df["away_key"] = df["away_team"].map(normalize_team)

    return df


def download_football_data():
    print("    Descargando NOR.csv...")

    headers = {
        "User-Agent": "Mozilla/5.0 EliteserienEdgePro/1.0"
    }

    response = requests.get(
        FOOTBALL_DATA_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "football_data_NOR_latest.csv"
    raw_path.write_bytes(response.content)

    df = pd.read_csv(
        io.BytesIO(response.content),
        low_memory=False,
    )

    print(
        f"    Filas descargadas: {len(df)} | "
        f"Columnas: {len(df.columns)}"
    )

    print(
        "    Columnas detectadas: "
        + ", ".join(df.columns)
    )

    return df


def choose_odds_columns(df):
    candidates = [
        ("AvgCH", "AvgCD", "AvgCA", "AVG cierre"),
        ("B365CH", "B365CD", "B365CA", "Bet365 cierre"),
        ("PSCH", "PSCD", "PSCA", "Pinnacle cierre"),
        ("MaxCH", "MaxCD", "MaxCA", "Máxima cierre"),
        ("BFECH", "BFECD", "BFECA", "Betfair Exchange cierre"),
    ]

    best = None
    best_count = -1

    for h, d, a, label in candidates:
        if not all(c in df.columns for c in (h, d, a)):
            continue

        temp = df[[h, d, a]].apply(
            pd.to_numeric,
            errors="coerce",
        )

        valid = (
            (temp[h] > 1.0)
            & (temp[d] > 1.0)
            & (temp[a] > 1.0)
        )

        count = int(valid.sum())

        print(
            f"    {label:<28} -> "
            f"{count} filas válidas"
        )

        if count > best_count:
            best_count = count
            best = (h, d, a, label)

    if best is None or best_count <= 0:
        raise RuntimeError(
            "Football-Data se descargó, pero no encuentro "
            "ningún trío de cuotas 1X2 válido."
        )

    return best


def prepare_odds(raw):
    required = [
        "Season",
        "Home",
        "Away",
    ]

    missing = [
        c for c in required
        if c not in raw.columns
    ]

    if missing:
        raise RuntimeError(
            "Faltan columnas en NOR.csv: "
            + ", ".join(missing)
        )

    hcol, dcol, acol, label = choose_odds_columns(raw)

    df = raw.copy()

    df["season_key"] = df["Season"].map(season_to_int)

    df["home_key"] = df["Home"].map(normalize_team)
    df["away_key"] = df["Away"].map(normalize_team)

    df["odds_home"] = pd.to_numeric(
        df[hcol],
        errors="coerce",
    )

    df["odds_draw"] = pd.to_numeric(
        df[dcol],
        errors="coerce",
    )

    df["odds_away"] = pd.to_numeric(
        df[acol],
        errors="coerce",
    )

    if "Date" in df.columns:
        try:
            df["odds_date"] = pd.to_datetime(
                df["Date"],
                format="mixed",
                dayfirst=True,
                errors="coerce",
            )
        except TypeError:
            df["odds_date"] = pd.to_datetime(
                df["Date"],
                dayfirst=True,
                errors="coerce",
            )
    else:
        df["odds_date"] = pd.NaT

    if "HG" in df.columns:
        df["fd_home_goals"] = pd.to_numeric(
            df["HG"],
            errors="coerce",
        )
    else:
        df["fd_home_goals"] = np.nan

    if "AG" in df.columns:
        df["fd_away_goals"] = pd.to_numeric(
            df["AG"],
            errors="coerce",
        )
    else:
        df["fd_away_goals"] = np.nan

    df = df.dropna(
        subset=[
            "season_key",
            "odds_home",
            "odds_draw",
            "odds_away",
        ]
    ).copy()

    df["season_key"] = df["season_key"].astype(int)

    df = df[
        (df["season_key"] >= 2023)
        & (df["odds_home"] > 1.0)
        & (df["odds_draw"] > 1.0)
        & (df["odds_away"] > 1.0)
    ].copy()

    print()
    print(
        f"    Fuente elegida: {label} "
        f"({hcol}/{dcol}/{acol})"
    )

    print(
        f"    Partidos 2023+ con cuotas: {len(df)}"
    )

    return df, label


def merge_data(pred, odds):
    merged = pred.merge(
        odds[
            [
                "season_key",
                "home_key",
                "away_key",
                "Home",
                "Away",
                "odds_date",
                "odds_home",
                "odds_draw",
                "odds_away",
                "fd_home_goals",
                "fd_away_goals",
            ]
        ],
        on=[
            "season_key",
            "home_key",
            "away_key",
        ],
        how="left",
    )

    if merged.duplicated("match_id", keep=False).any():
        # En caso de duplicado, priorizar coincidencia de marcador.
        merged["score_bonus"] = 0

        score_match = (
            pd.to_numeric(
                merged["home_goals"],
                errors="coerce",
            )
            == merged["fd_home_goals"]
        ) & (
            pd.to_numeric(
                merged["away_goals"],
                errors="coerce",
            )
            == merged["fd_away_goals"]
        )

        merged.loc[
            score_match.fillna(False),
            "score_bonus",
        ] = 1

        merged = (
            merged
            .sort_values(
                ["match_id", "score_bonus"],
                ascending=[True, False],
            )
            .drop_duplicates(
                subset=["match_id"],
                keep="first",
            )
        )

    return merged


def market_no_vig(df):
    h = 1.0 / df["odds_home"].to_numpy(dtype=float)
    d = 1.0 / df["odds_draw"].to_numpy(dtype=float)
    a = 1.0 / df["odds_away"].to_numpy(dtype=float)

    total = h + d + a

    return h / total, d / total, a / total


def calculate_ev(merged):
    df = merged.dropna(
        subset=[
            "odds_home",
            "odds_draw",
            "odds_away",
            "final_home",
            "final_draw",
            "final_away",
        ]
    ).copy()

    if df.empty:
        raise RuntimeError(
            "No se cruzó ningún partido con cuota."
        )

    market_h, market_d, market_a = market_no_vig(df)

    df["market_home_novig"] = market_h
    df["market_draw_novig"] = market_d
    df["market_away_novig"] = market_a

    df["ev_home"] = (
        df["final_home"] * df["odds_home"] - 1.0
    )

    df["ev_draw"] = (
        df["final_draw"] * df["odds_draw"] - 1.0
    )

    df["ev_away"] = (
        df["final_away"] * df["odds_away"] - 1.0
    )

    ev_matrix = df[
        ["ev_home", "ev_draw", "ev_away"]
    ].to_numpy(dtype=float)

    prob_matrix = df[
        ["final_home", "final_draw", "final_away"]
    ].to_numpy(dtype=float)

    odds_matrix = df[
        ["odds_home", "odds_draw", "odds_away"]
    ].to_numpy(dtype=float)

    market_matrix = df[
        [
            "market_home_novig",
            "market_draw_novig",
            "market_away_novig",
        ]
    ].to_numpy(dtype=float)

    best = np.argmax(
        ev_matrix,
        axis=1,
    )

    outcome_arr = np.asarray(OUTCOMES)

    df["pick"] = outcome_arr[best]

    df["best_ev"] = ev_matrix[
        np.arange(len(df)),
        best,
    ]

    df["pick_prob"] = prob_matrix[
        np.arange(len(df)),
        best,
    ]

    df["pick_odds"] = odds_matrix[
        np.arange(len(df)),
        best,
    ]

    df["market_prob"] = market_matrix[
        np.arange(len(df)),
        best,
    ]

    df["prob_edge"] = (
        df["pick_prob"] - df["market_prob"]
    )

    df["won"] = (
        df["pick"] == df["actual"]
    ).astype(int)

    df["profit_1u"] = np.where(
        df["won"] == 1,
        df["pick_odds"] - 1.0,
        -1.0,
    )

    return df


def make_global_summary(df):
    rows = []

    for ev_threshold in (
        0.00,
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
    ):
        bets = df[
            df["best_ev"] >= ev_threshold
        ].copy()

        if bets.empty:
            continue

        rows.append(
            {
                "ev_threshold": ev_threshold,
                "bets": len(bets),
                "hit_rate": float(
                    bets["won"].mean()
                ),
                "avg_odds": float(
                    bets["pick_odds"].mean()
                ),
                "avg_model_prob": float(
                    bets["pick_prob"].mean()
                ),
                "avg_market_prob": float(
                    bets["market_prob"].mean()
                ),
                "avg_ev": float(
                    bets["best_ev"].mean()
                ),
                "profit_units": float(
                    bets["profit_1u"].sum()
                ),
                "roi": float(
                    bets["profit_1u"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def make_season_summary(df):
    rows = []

    for season, season_df in df.groupby("season"):
        for ev_threshold in (
            0.03,
            0.05,
            0.08,
            0.10,
            0.15,
        ):
            bets = season_df[
                season_df["best_ev"] >= ev_threshold
            ].copy()

            if len(bets) < 5:
                continue

            rows.append(
                {
                    "season": int(season),
                    "ev_threshold": ev_threshold,
                    "bets": len(bets),
                    "hit_rate": float(
                        bets["won"].mean()
                    ),
                    "avg_odds": float(
                        bets["pick_odds"].mean()
                    ),
                    "profit_units": float(
                        bets["profit_1u"].sum()
                    ),
                    "roi": float(
                        bets["profit_1u"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_outcome_summary(df):
    rows = []

    names = {
        "H": "LOCAL",
        "D": "EMPATE",
        "A": "VISITANTE",
    }

    for pick, pick_df in df.groupby("pick"):
        bets = pick_df[
            pick_df["best_ev"] >= 0.05
        ].copy()

        if len(bets) < 5:
            continue

        rows.append(
            {
                "pick": pick,
                "pick_name": names.get(pick, pick),
                "bets": len(bets),
                "hit_rate": float(
                    bets["won"].mean()
                ),
                "avg_odds": float(
                    bets["pick_odds"].mean()
                ),
                "profit_units": float(
                    bets["profit_1u"].sum()
                ),
                "roi": float(
                    bets["profit_1u"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print()
    print("=" * 80)
    print("ELITESERIEN EDGE PRO - FOOTBALL-DATA DIRECT EV 1X2")
    print("=" * 80)
    print()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DB_PATH) as conn:
        print("1/6 Cargando predicciones 1X2...")
        pred = load_predictions(conn)

        print(
            f"    Predicciones: {len(pred)}"
        )

        print("2/6 Descargando Football-Data original...")
        raw = download_football_data()

        print("3/6 Detectando mejor fuente de cuotas...")
        odds, odds_source = prepare_odds(raw)

        print("4/6 Cruzando partidos...")
        merged = merge_data(
            pred,
            odds,
        )

        covered = merged[
            ["odds_home", "odds_draw", "odds_away"]
        ].notna().all(axis=1)

        n_covered = int(
            covered.sum()
        )

        print(
            f"    Cobertura: "
            f"{n_covered}/{len(merged)} "
            f"({n_covered/len(merged)*100:.1f}%)"
        )

        if n_covered < len(merged):
            unmatched = merged[
                ~covered
            ][
                [
                    "season",
                    "home_team",
                    "away_team",
                    "home_key",
                    "away_key",
                ]
            ].drop_duplicates()

            print()
            print(
                "    No emparejados "
                f"({len(unmatched)} combinaciones):"
            )

            for r in unmatched.head(20).itertuples(
                index=False
            ):
                print(
                    f"      {int(r.season)} | "
                    f"{r.home_team} - {r.away_team} | "
                    f"{r.home_key} - {r.away_key}"
                )

        print("5/6 Calculando EV + ROI...")
        detail = calculate_ev(
            merged
        )

        global_summary = make_global_summary(
            detail
        )

        season_summary = make_season_summary(
            detail
        )

        outcome_summary = make_outcome_summary(
            detail
        )

        print("6/6 Guardando resultados...")

        detail.to_sql(
            "result_v1_ev_predictions",
            conn,
            if_exists="replace",
            index=False,
        )

        global_summary.to_sql(
            "result_v1_ev_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        season_summary.to_sql(
            "result_v1_ev_by_season",
            conn,
            if_exists="replace",
            index=False,
        )

        outcome_summary.to_sql(
            "result_v1_ev_by_outcome",
            conn,
            if_exists="replace",
            index=False,
        )

        # Reparamos además las cuotas del matches existente
        # cuando es posible cruzarlo por temporada/equipos.
        conn.commit()

    detail.to_csv(
        REPORTS_DIR
        / "result_v1_ev_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    global_summary.to_csv(
        REPORTS_DIR
        / "result_v1_ev_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    season_summary.to_csv(
        REPORTS_DIR
        / "result_v1_ev_by_season.csv",
        index=False,
        encoding="utf-8-sig",
    )

    outcome_summary.to_csv(
        REPORTS_DIR
        / "result_v1_ev_by_outcome.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("BACKTEST EV 1X2")
    print("=" * 80)

    for r in global_summary.itertuples(
        index=False
    ):
        print(
            f"EV >= {r.ev_threshold*100:4.0f}% | "
            f"N={int(r.bets):3d} | "
            f"Hit={r.hit_rate*100:5.1f}% | "
            f"Cuota={r.avg_odds:.2f} | "
            f"EV medio={r.avg_ev*100:5.1f}% | "
            f"Profit={r.profit_units:+7.2f}u | "
            f"ROI={r.roi*100:+6.2f}%"
        )

    print()
    print("=" * 80)
    print("ROI POR TEMPORADA")
    print("=" * 80)

    if season_summary.empty:
        print("Sin muestra suficiente.")
    else:
        for r in season_summary.itertuples(
            index=False
        ):
            print(
                f"{int(r.season)} | "
                f"EV>={r.ev_threshold*100:2.0f}% | "
                f"N={int(r.bets):3d} | "
                f"Hit={r.hit_rate*100:5.1f}% | "
                f"Cuota={r.avg_odds:.2f} | "
                f"Profit={r.profit_units:+7.2f}u | "
                f"ROI={r.roi*100:+6.2f}%"
            )

    print()
    print("=" * 80)
    print("ROI POR TIPO DE PICK - EV >= 5%")
    print("=" * 80)

    if outcome_summary.empty:
        print("Sin muestra suficiente.")
    else:
        for r in outcome_summary.itertuples(
            index=False
        ):
            print(
                f"{r.pick_name:<10} | "
                f"N={int(r.bets):3d} | "
                f"Hit={r.hit_rate*100:5.1f}% | "
                f"Cuota={r.avg_odds:.2f} | "
                f"Profit={r.profit_units:+7.2f}u | "
                f"ROI={r.roi*100:+6.2f}%"
            )

    print()
    print("=" * 80)
    print("FUENTE DE CUOTAS")
    print("=" * 80)
    print(f"{odds_source}")
    print(
        "Backtest con stake fijo de 1 unidad por pick. "
        "El ROI histórico no garantiza rentabilidad futura."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
