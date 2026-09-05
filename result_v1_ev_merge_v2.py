from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "eliteserien.db"
REPORTS_DIR = BASE_DIR / "reports"

OUTCOMES = ["H", "D", "A"]


def table_exists(conn, table):
    return conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table,),
    ).fetchone() is not None


def columns_of(conn, table):
    return [
        r[1]
        for r in conn.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    ]


def first_col(cols, candidates):
    lookup = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def normalize_team(value):
    if pd.isna(value):
        return ""

    s = str(value).strip().lower()

    # Caracteres noruegos antes de quitar diacríticos.
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
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )

    s = s.replace("&", "and")
    s = re.sub(r"\bfk\b", " ", s)
    s = re.sub(r"\bil\b", " ", s)
    s = re.sub(r"\bsk\b", " ", s)
    s = re.sub(r"\bbk\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)

    aliases = {
        "bodoglimt": "bodoglimt",
        "bodoeglimt": "bodoglimt",
        "bodo": "bodoglimt",

        "stromsgodset": "stromsgodset",
        "stromsgodsetif": "stromsgodset",

        "lillestrom": "lillestrom",
        "lillestromsk": "lillestrom",

        "valerenga": "valerenga",

        "tromso": "tromso",
        "tromsoil": "tromso",

        "stabaek": "stabaek",
        "stabaekif": "stabaek",

        "aalesund": "aalesund",
        "aalesundsfk": "aalesund",

        "hamkam": "hamkam",
        "hamarkameratene": "hamkam",

        "odd": "odd",
        "odds": "odd",
        "oddsballklubb": "odd",

        "haugesund": "haugesund",
        "fkhaugesund": "haugesund",

        "sarpsborg08": "sarpsborg08",
        "sarpsborg": "sarpsborg08",

        "kfumoslo": "kfumoslo",
        "kfum": "kfumoslo",

        "kristiansund": "kristiansund",
        "kristiansundbk": "kristiansund",

        "sandefjord": "sandefjord",
        "sandefjordfotball": "sandefjord",

        "fredrikstad": "fredrikstad",
        "fredrikstadfk": "fredrikstad",

        "mjoendalen": "mjondalen",
        "mjondalen": "mjondalen",

        "rosenborg": "rosenborg",
        "rosenborgbk": "rosenborg",

        "brann": "brann",
        "skbrann": "brann",

        "molde": "molde",
        "moldefk": "molde",

        "viking": "viking",
        "vikingfk": "viking",

        "start": "start",
        "ikstart": "start",

        "jerv": "jerv",
        "fkjerv": "jerv",
    }

    return aliases.get(s, s)


def season_value(value, date_value=None):
    if not pd.isna(value):
        m = re.search(r"(20\d{2})", str(value))
        if m:
            return int(m.group(1))

    if date_value is not None and not pd.isna(date_value):
        return int(pd.Timestamp(date_value).year)

    return np.nan


def load_predictions(conn):
    if not table_exists(conn, "result_v1_backtest_predictions"):
        raise RuntimeError(
            "Falta result_v1_backtest_predictions."
        )

    df = pd.read_sql_query(
        """
        SELECT *
        FROM result_v1_backtest_predictions
        ORDER BY season, match_day, match_id
        """,
        conn,
    )

    df["match_day"] = pd.to_datetime(
        df["match_day"],
        errors="coerce",
    ).dt.floor("D")

    df["season_key"] = df["season"].apply(
        lambda x: season_value(x)
    )

    df["home_key"] = df["home_team"].map(normalize_team)
    df["away_key"] = df["away_team"].map(normalize_team)

    return df


def detect_matches_schema(conn):
    table = "matches"

    if not table_exists(conn, table):
        raise RuntimeError("No existe la tabla matches.")

    cols = columns_of(conn, table)

    spec = {
        "table": table,
        "season": first_col(
            cols,
            ["season", "Season"],
        ),
        "date": first_col(
            cols,
            [
                "date",
                "Date",
                "match_date",
                "utc_time",
                "datetime",
            ],
        ),
        "home": first_col(
            cols,
            [
                "home_team",
                "home",
                "Home",
                "HomeTeam",
            ],
        ),
        "away": first_col(
            cols,
            [
                "away_team",
                "away",
                "Away",
                "AwayTeam",
            ],
        ),
        "hg": first_col(
            cols,
            [
                "home_goals",
                "hg",
                "HG",
                "FTHG",
            ],
        ),
        "ag": first_col(
            cols,
            [
                "away_goals",
                "ag",
                "AG",
                "FTAG",
            ],
        ),
        "oh": first_col(
            cols,
            [
                "odds_home",
                "AvgCH",
                "avgch",
                "B365CH",
                "PSCH",
                "MaxCH",
            ],
        ),
        "od": first_col(
            cols,
            [
                "odds_draw",
                "AvgCD",
                "avgcd",
                "B365CD",
                "PSCD",
                "MaxCD",
            ],
        ),
        "oa": first_col(
            cols,
            [
                "odds_away",
                "AvgCA",
                "avgca",
                "B365CA",
                "PSCA",
                "MaxCA",
            ],
        ),
    }

    required = ["home", "away", "oh", "od", "oa"]

    missing = [
        key for key in required
        if not spec[key]
    ]

    if missing:
        raise RuntimeError(
            "No pude detectar columnas: "
            + ", ".join(missing)
            + "\nColumnas disponibles: "
            + ", ".join(cols)
        )

    return spec


def load_odds(conn, spec):
    select_parts = []

    def add(alias, col):
        if col:
            select_parts.append(
                f'"{col}" AS "{alias}"'
            )
        else:
            select_parts.append(
                f'NULL AS "{alias}"'
            )

    add("season_raw", spec["season"])
    add("odds_date_raw", spec["date"])
    add("odds_home_team", spec["home"])
    add("odds_away_team", spec["away"])
    add("odds_hg", spec["hg"])
    add("odds_ag", spec["ag"])
    add("odds_home", spec["oh"])
    add("odds_draw", spec["od"])
    add("odds_away", spec["oa"])

    query = (
        "SELECT "
        + ", ".join(select_parts)
        + f' FROM "{spec["table"]}"'
    )

    df = pd.read_sql_query(query, conn)

    # Parsing de fecha tolerante a DD/MM/YYYY e ISO.
    if df["odds_date_raw"].notna().any():
        try:
            df["odds_date"] = pd.to_datetime(
                df["odds_date_raw"],
                format="mixed",
                dayfirst=True,
                errors="coerce",
            ).dt.floor("D")
        except TypeError:
            df["odds_date"] = pd.to_datetime(
                df["odds_date_raw"],
                dayfirst=True,
                errors="coerce",
            ).dt.floor("D")
    else:
        df["odds_date"] = pd.NaT

    df["season_key"] = [
        season_value(s, d)
        for s, d in zip(
            df["season_raw"],
            df["odds_date"],
        )
    ]

    df["home_key"] = df["odds_home_team"].map(normalize_team)
    df["away_key"] = df["odds_away_team"].map(normalize_team)

    for col in (
        "odds_home",
        "odds_draw",
        "odds_away",
        "odds_hg",
        "odds_ag",
    ):
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

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
        (df["odds_home"] > 1.0)
        & (df["odds_draw"] > 1.0)
        & (df["odds_away"] > 1.0)
    ].copy()

    return df


def print_team_diagnostics(pred, odds):
    pred_names = sorted(
        set(
            zip(
                pred["home_team"].astype(str),
                pred["home_key"],
            )
        )
    )

    odds_names = sorted(
        set(
            zip(
                odds["odds_home_team"].astype(str),
                odds["home_key"],
            )
        )
    )

    print()
    print("    Ejemplos nombres modelo:")
    for raw, key in pred_names[:12]:
        print(f"      {raw:<25} -> {key}")

    print()
    print("    Ejemplos nombres cuotas:")
    for raw, key in odds_names[:12]:
        print(f"      {raw:<25} -> {key}")


def robust_merge(pred, odds):
    # En Eliteserien cada combinación local/visitante es única
    # dentro de una temporada, por lo que esta es la clave principal.
    merged = pred.merge(
        odds,
        on=[
            "season_key",
            "home_key",
            "away_key",
        ],
        how="left",
        suffixes=("", "_odds"),
    )

    # Si por algún motivo hay duplicados, puntuar candidatos.
    if merged.duplicated("match_id", keep=False).any():
        merged["candidate_score"] = 0.0

        # Fecha cercana suma calidad.
        valid_dates = (
            merged["match_day"].notna()
            & merged["odds_date"].notna()
        )

        delta = (
            merged.loc[valid_dates, "match_day"]
            - merged.loc[valid_dates, "odds_date"]
        ).abs().dt.days

        merged.loc[
            valid_dates,
            "candidate_score",
        ] += np.maximum(0, 10 - delta)

        # Marcador coincidente suma mucho.
        if "home_goals" in merged.columns:
            score_match = (
                pd.to_numeric(
                    merged["home_goals"],
                    errors="coerce",
                )
                == merged["odds_hg"]
            ) & (
                pd.to_numeric(
                    merged["away_goals"],
                    errors="coerce",
                )
                == merged["odds_ag"]
            )

            merged.loc[
                score_match.fillna(False),
                "candidate_score",
            ] += 50

        merged = (
            merged
            .sort_values(
                ["match_id", "candidate_score"],
                ascending=[True, False],
            )
            .drop_duplicates(
                subset=["match_id"],
                keep="first",
            )
        )

    return merged


def market_no_vig(df):
    ih = 1.0 / df["odds_home"].to_numpy(dtype=float)
    id_ = 1.0 / df["odds_draw"].to_numpy(dtype=float)
    ia = 1.0 / df["odds_away"].to_numpy(dtype=float)

    total = ih + id_ + ia

    return ih / total, id_ / total, ia / total


def calculate_ev(detail):
    df = detail.dropna(
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
        return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    mh, md, ma = market_no_vig(df)

    df["market_home_novig"] = mh
    df["market_draw_novig"] = md
    df["market_away_novig"] = ma

    df["ev_home"] = (
        df["final_home"] * df["odds_home"] - 1
    )
    df["ev_draw"] = (
        df["final_draw"] * df["odds_draw"] - 1
    )
    df["ev_away"] = (
        df["final_away"] * df["odds_away"] - 1
    )

    evs = df[
        ["ev_home", "ev_draw", "ev_away"]
    ].to_numpy(dtype=float)

    probs = df[
        ["final_home", "final_draw", "final_away"]
    ].to_numpy(dtype=float)

    odds_matrix = df[
        ["odds_home", "odds_draw", "odds_away"]
    ].to_numpy(dtype=float)

    best = np.argmax(evs, axis=1)

    outcome_arr = np.asarray(OUTCOMES)

    df["pick"] = outcome_arr[best]
    df["best_ev"] = evs[np.arange(len(df)), best]
    df["pick_prob"] = probs[np.arange(len(df)), best]
    df["pick_odds"] = odds_matrix[np.arange(len(df)), best]

    df["won"] = (
        df["pick"] == df["actual"]
    ).astype(int)

    df["profit_1u"] = np.where(
        df["won"] == 1,
        df["pick_odds"] - 1.0,
        -1.0,
    )

    # Edge también contra probabilidad no-vig del mercado.
    market_matrix = df[
        [
            "market_home_novig",
            "market_draw_novig",
            "market_away_novig",
        ]
    ].to_numpy(dtype=float)

    df["prob_edge"] = (
        probs[np.arange(len(df)), best]
        - market_matrix[np.arange(len(df)), best]
    )

    global_rows = []

    for threshold in (
        0.00,
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
    ):
        bets = df[
            df["best_ev"] >= threshold
        ]

        if bets.empty:
            continue

        global_rows.append(
            {
                "ev_threshold": threshold,
                "bets": len(bets),
                "hit_rate": bets["won"].mean(),
                "avg_odds": bets["pick_odds"].mean(),
                "avg_probability": bets["pick_prob"].mean(),
                "avg_ev": bets["best_ev"].mean(),
                "profit_units": bets["profit_1u"].sum(),
                "roi": bets["profit_1u"].mean(),
            }
        )

    by_season_rows = []

    for season, g in df.groupby("season"):
        for threshold in (
            0.03,
            0.05,
            0.08,
            0.10,
            0.15,
        ):
            bets = g[
                g["best_ev"] >= threshold
            ]

            if len(bets) < 5:
                continue

            by_season_rows.append(
                {
                    "season": int(season),
                    "ev_threshold": threshold,
                    "bets": len(bets),
                    "hit_rate": bets["won"].mean(),
                    "avg_odds": bets["pick_odds"].mean(),
                    "profit_units": bets["profit_1u"].sum(),
                    "roi": bets["profit_1u"].mean(),
                }
            )

    # Separar resultado elegido: H / D / A.
    outcome_rows = []

    for pick, g in df.groupby("pick"):
        bets = g[g["best_ev"] >= 0.05]

        if len(bets) < 5:
            continue

        outcome_rows.append(
            {
                "pick": pick,
                "bets_ev5": len(bets),
                "hit_rate": bets["won"].mean(),
                "avg_odds": bets["pick_odds"].mean(),
                "profit_units": bets["profit_1u"].sum(),
                "roi": bets["profit_1u"].mean(),
            }
        )

    return (
        df,
        pd.DataFrame(global_rows),
        pd.DataFrame(by_season_rows),
        pd.DataFrame(outcome_rows),
    )


def main():
    print()
    print("=" * 80)
    print("ELITESERIEN EDGE PRO - FIX ODDS MERGE + ROI 1X2")
    print("=" * 80)
    print()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DB_PATH) as conn:
        print("1/5 Cargando predicciones 1X2...")
        pred = load_predictions(conn)

        print("2/5 Cargando cuotas Football-Data...")
        spec = detect_matches_schema(conn)

        print(
            f"    home={spec['home']} | away={spec['away']} | "
            f"season={spec['season']} | date={spec['date']}"
        )
        print(
            f"    odds={spec['oh']} / {spec['od']} / {spec['oa']}"
        )

        odds = load_odds(conn, spec)

        print(
            f"    Filas con cuotas válidas: {len(odds)}"
        )

        print_team_diagnostics(pred, odds)

        print()
        print("3/5 Emparejando por temporada + local + visitante...")
        merged = robust_merge(pred, odds)

        covered = merged[
            ["odds_home", "odds_draw", "odds_away"]
        ].notna().all(axis=1)

        n_covered = int(covered.sum())

        print(
            f"    Cobertura: {n_covered}/{len(merged)} "
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
            print("    Primeros no emparejados:")
            for r in unmatched.head(20).itertuples(index=False):
                print(
                    f"      {int(r.season)} | "
                    f"{r.home_team} - {r.away_team} | "
                    f"{r.home_key} - {r.away_key}"
                )

        print()
        print("4/5 Calculando EV y ROI...")
        detail, global_summary, by_season, by_outcome = (
            calculate_ev(merged)
        )

        if detail.empty:
            raise RuntimeError(
                "Seguimos sin partidos con cuotas. "
                "Mira los nombres no emparejados impresos arriba."
            )

        print("5/5 Guardando resultados...")

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

        by_season.to_sql(
            "result_v1_ev_by_season",
            conn,
            if_exists="replace",
            index=False,
        )

        by_outcome.to_sql(
            "result_v1_ev_by_outcome",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.commit()

    detail.to_csv(
        REPORTS_DIR / "result_v1_ev_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    global_summary.to_csv(
        REPORTS_DIR / "result_v1_ev_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_season.to_csv(
        REPORTS_DIR / "result_v1_ev_by_season.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_outcome.to_csv(
        REPORTS_DIR / "result_v1_ev_by_outcome.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("BACKTEST EV 1X2")
    print("=" * 80)

    for r in global_summary.itertuples(index=False):
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

    for r in by_season.itertuples(index=False):
        print(
            f"{int(r.season)} | "
            f"EV>={r.ev_threshold*100:2.0f}% | "
            f"N={int(r.bets):3d} | "
            f"Hit={r.hit_rate*100:5.1f}% | "
            f"Profit={r.profit_units:+7.2f}u | "
            f"ROI={r.roi*100:+6.2f}%"
        )

    print()
    print("=" * 80)
    print("ROI POR TIPO DE PICK (EV >= 5%)")
    print("=" * 80)

    if by_outcome.empty:
        print("No hay muestra suficiente.")
    else:
        names = {
            "H": "LOCAL",
            "D": "EMPATE",
            "A": "VISITANTE",
        }

        for r in by_outcome.itertuples(index=False):
            print(
                f"{names.get(r.pick, r.pick):<10} | "
                f"N={int(r.bets_ev5):3d} | "
                f"Hit={r.hit_rate*100:5.1f}% | "
                f"Cuota={r.avg_odds:.2f} | "
                f"Profit={r.profit_units:+7.2f}u | "
                f"ROI={r.roi*100:+6.2f}%"
            )

    print()
    print("=" * 80)
    print("NOTA")
    print("=" * 80)
    print(
        "El ROI es retrospectivo y usa 1 unidad fija por apuesta. "
        "No prueba beneficio futuro; buscamos estabilidad entre temporadas."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
