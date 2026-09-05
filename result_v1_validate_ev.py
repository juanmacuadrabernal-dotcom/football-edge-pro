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

CLASSES = ["H", "D", "A"]


def manual_logloss(y_true, probs):
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, 1 - 1e-12)

    class_to_idx = {"H": 0, "D": 1, "A": 2}
    idx = np.array([class_to_idx[y] for y in y_true], dtype=int)

    chosen = probs[np.arange(len(probs)), idx]
    return float(-np.mean(np.log(chosen)))


def multiclass_brier(y_true, probs):
    mapping = {
        "H": np.array([1.0, 0.0, 0.0]),
        "D": np.array([0.0, 1.0, 0.0]),
        "A": np.array([0.0, 0.0, 1.0]),
    }

    ys = np.vstack([mapping[y] for y in y_true])
    probs = np.asarray(probs, dtype=float)

    return float(
        np.mean(
            np.sum((probs - ys) ** 2, axis=1) / 3.0
        )
    )


def accuracy(y_true, probs):
    pred = np.asarray(CLASSES)[np.argmax(probs, axis=1)]
    return float(np.mean(pred == np.asarray(y_true)))


def table_exists(conn, table):
    q = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table,),
    ).fetchone()

    return q is not None


def columns_of(conn, table):
    rows = conn.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()

    return [r[1] for r in rows]


def first_existing(columns, candidates):
    lower_map = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def load_predictions(conn):
    if not table_exists(conn, "result_v1_backtest_predictions"):
        raise RuntimeError(
            "No existe result_v1_backtest_predictions. "
            "Ejecuta result_model_v1_fixed.py primero."
        )

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
            "result_v1_backtest_predictions está vacía."
        )

    df["match_day"] = pd.to_datetime(
        df["match_day"],
        errors="coerce",
    ).dt.floor("D")

    return df.dropna(subset=["match_day"]).copy()


def corrected_metrics(pred):
    rows = []

    groups = [("GLOBAL", pred)]

    for season, g in pred.groupby("season"):
        groups.append((str(int(season)), g))

    for period, g in groups:
        y = g["actual"].to_numpy()

        final = g[
            ["final_home", "final_draw", "final_away"]
        ].to_numpy(dtype=float)

        raw = g[
            ["raw_home", "raw_draw", "raw_away"]
        ].to_numpy(dtype=float)

        base = g[
            ["base_home", "base_draw", "base_away"]
        ].to_numpy(dtype=float)

        rows.append(
            {
                "period": period,
                "matches": len(g),
                "accuracy_final": accuracy(y, final),
                "accuracy_raw": accuracy(y, raw),
                "accuracy_base": accuracy(y, base),
                "brier_final": multiclass_brier(y, final),
                "brier_raw": multiclass_brier(y, raw),
                "brier_base": multiclass_brier(y, base),
                "logloss_final": manual_logloss(y, final),
                "logloss_raw": manual_logloss(y, raw),
                "logloss_base": manual_logloss(y, base),
            }
        )

    return pd.DataFrame(rows)


def normalize_team(value):
    if pd.isna(value):
        return ""

    s = str(value).strip().lower()

    s = unicodedata.normalize("NFKD", s)
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )

    replacements = {
        "ø": "o",
        "æ": "ae",
        "å": "a",
        "ö": "o",
    }

    for a, b in replacements.items():
        s = s.replace(a, b)

    s = re.sub(r"\bfk\b", "", s)
    s = re.sub(r"\bil\b", "", s)
    s = re.sub(r"\bsk\b", "", s)
    s = re.sub(r"\bbk\b", "", s)
    s = re.sub(r"\bodds ballklubb\b", "odd", s)
    s = re.sub(r"\bodds\b", "odd", s)
    s = re.sub(r"\bstromsgodset\b", "stromsgodset", s)
    s = re.sub(r"\bbodo/glimt\b", "bodo glimt", s)

    s = re.sub(r"[^a-z0-9]+", "", s)

    aliases = {
        "bodoeglimt": "bodoglimt",
        "bodoglimt": "bodoglimt",
        "lillestrom": "lillestrom",
        "stromsgodset": "stromsgodset",
        "haugesund": "haugesund",
        "fkhaugesund": "haugesund",
        "odd": "odd",
        "odds": "odd",
    }

    return aliases.get(s, s)


def detect_odds_table(conn):
    preferred = ["matches"]

    all_tables = [
        r[0]
        for r in conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table'
            ORDER BY name
            """
        ).fetchall()
    ]

    candidates = preferred + [
        t for t in all_tables
        if t not in preferred
    ]

    for table in candidates:
        if not table_exists(conn, table):
            continue

        cols = columns_of(conn, table)

        home_col = first_existing(
            cols,
            [
                "home_team", "home", "Home",
            ],
        )

        away_col = first_existing(
            cols,
            [
                "away_team", "away", "Away",
            ],
        )

        date_col = first_existing(
            cols,
            [
                "date", "Date", "match_date",
                "utc_time", "datetime",
            ],
        )

        if not home_col or not away_col or not date_col:
            continue

        # Preferimos cuota media de cierre.
        home_odds = first_existing(
            cols,
            [
                "AvgCH", "avgch",
                "B365CH", "b365ch",
                "PSCH", "psch",
                "MaxCH", "maxch",
                "home_odds", "odds_home",
            ],
        )

        draw_odds = first_existing(
            cols,
            [
                "AvgCD", "avgcd",
                "B365CD", "b365cd",
                "PSCD", "pscd",
                "MaxCD", "maxcd",
                "draw_odds", "odds_draw",
            ],
        )

        away_odds = first_existing(
            cols,
            [
                "AvgCA", "avgca",
                "B365CA", "b365ca",
                "PSCA", "psca",
                "MaxCA", "maxca",
                "away_odds", "odds_away",
            ],
        )

        if home_odds and draw_odds and away_odds:
            return {
                "table": table,
                "home": home_col,
                "away": away_col,
                "date": date_col,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
            }

    return None


def load_odds(conn, spec):
    table = spec["table"]

    query = f"""
        SELECT
            "{spec['date']}" AS odds_date,
            "{spec['home']}" AS odds_home_team,
            "{spec['away']}" AS odds_away_team,
            "{spec['home_odds']}" AS odds_home,
            "{spec['draw_odds']}" AS odds_draw,
            "{spec['away_odds']}" AS odds_away
        FROM "{table}"
    """

    odds = pd.read_sql_query(query, conn)

    odds["odds_date"] = pd.to_datetime(
        odds["odds_date"],
        errors="coerce",
        dayfirst=True,
    ).dt.floor("D")

    for col in ("odds_home", "odds_draw", "odds_away"):
        odds[col] = pd.to_numeric(
            odds[col],
            errors="coerce",
        )

    odds["home_key"] = odds["odds_home_team"].map(normalize_team)
    odds["away_key"] = odds["odds_away_team"].map(normalize_team)

    odds = odds.dropna(
        subset=[
            "odds_date",
            "odds_home",
            "odds_draw",
            "odds_away",
        ]
    ).copy()

    odds = odds[
        (odds["odds_home"] > 1.0)
        & (odds["odds_draw"] > 1.0)
        & (odds["odds_away"] > 1.0)
    ].copy()

    return odds


def merge_predictions_odds(pred, odds):
    x = pred.copy()

    x["home_key"] = x["home_team"].map(normalize_team)
    x["away_key"] = x["away_team"].map(normalize_team)

    merged = x.merge(
        odds,
        left_on=[
            "match_day",
            "home_key",
            "away_key",
        ],
        right_on=[
            "odds_date",
            "home_key",
            "away_key",
        ],
        how="left",
    )

    # Si hubiera duplicados del proveedor, elegimos una fila.
    merged = merged.sort_values(
        ["match_id"]
    ).drop_duplicates(
        subset=["match_id"],
        keep="first",
    )

    return merged


def no_vig_probs(frame):
    inv_h = 1.0 / frame["odds_home"].to_numpy(dtype=float)
    inv_d = 1.0 / frame["odds_draw"].to_numpy(dtype=float)
    inv_a = 1.0 / frame["odds_away"].to_numpy(dtype=float)

    total = inv_h + inv_d + inv_a

    return (
        inv_h / total,
        inv_d / total,
        inv_a / total,
    )


def ev_backtest(merged):
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
        return pd.DataFrame(), pd.DataFrame(), df

    fair_h, fair_d, fair_a = no_vig_probs(df)

    df["market_prob_home"] = fair_h
    df["market_prob_draw"] = fair_d
    df["market_prob_away"] = fair_a

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

    best_idx = np.argmax(ev_matrix, axis=1)

    outcomes = np.asarray(CLASSES)
    df["best_pick"] = outcomes[best_idx]

    df["best_ev"] = ev_matrix[
        np.arange(len(df)),
        best_idx,
    ]

    odds_matrix = df[
        ["odds_home", "odds_draw", "odds_away"]
    ].to_numpy(dtype=float)

    df["best_odds"] = odds_matrix[
        np.arange(len(df)),
        best_idx,
    ]

    prob_matrix = df[
        ["final_home", "final_draw", "final_away"]
    ].to_numpy(dtype=float)

    df["best_prob"] = prob_matrix[
        np.arange(len(df)),
        best_idx,
    ]

    df["won"] = (
        df["best_pick"] == df["actual"]
    ).astype(int)

    df["profit_1u"] = np.where(
        df["won"] == 1,
        df["best_odds"] - 1.0,
        -1.0,
    )

    rows = []

    thresholds = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]

    for threshold in thresholds:
        bets = df[
            df["best_ev"] >= threshold
        ].copy()

        if bets.empty:
            continue

        rows.append(
            {
                "ev_threshold": threshold,
                "bets": len(bets),
                "hit_rate": float(bets["won"].mean()),
                "avg_odds": float(bets["best_odds"].mean()),
                "avg_model_prob": float(bets["best_prob"].mean()),
                "avg_ev": float(bets["best_ev"].mean()),
                "profit_units": float(bets["profit_1u"].sum()),
                "roi": float(bets["profit_1u"].mean()),
            }
        )

    global_summary = pd.DataFrame(rows)

    season_rows = []

    for season, g in df.groupby("season"):
        for threshold in (0.03, 0.05, 0.08, 0.10):
            bets = g[
                g["best_ev"] >= threshold
            ].copy()

            if len(bets) < 5:
                continue

            season_rows.append(
                {
                    "season": int(season),
                    "ev_threshold": threshold,
                    "bets": len(bets),
                    "hit_rate": float(bets["won"].mean()),
                    "avg_odds": float(bets["best_odds"].mean()),
                    "profit_units": float(bets["profit_1u"].sum()),
                    "roi": float(bets["profit_1u"].mean()),
                }
            )

    season_summary = pd.DataFrame(season_rows)

    return global_summary, season_summary, df


def main():
    print()
    print("=" * 78)
    print("ELITESERIEN EDGE PRO - VALIDACION 1X2 + EV")
    print("=" * 78)
    print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        print("1/5 Corrigiendo métricas 1X2...")
        pred = load_predictions(conn)
        metrics = corrected_metrics(pred)

        print("2/5 Buscando cuotas históricas...")
        spec = detect_odds_table(conn)

        if spec is None:
            print()
            print("No he podido detectar automáticamente las columnas de cuotas.")
            print("Las métricas corregidas sí se guardarán.")
            odds = None
        else:
            print(
                f"    Tabla: {spec['table']} | "
                f"Cuotas: {spec['home_odds']} / "
                f"{spec['draw_odds']} / {spec['away_odds']}"
            )
            odds = load_odds(conn, spec)

        print("3/5 Guardando métricas corregidas...")
        metrics.to_sql(
            "result_v1_corrected_metrics",
            conn,
            if_exists="replace",
            index=False,
        )

        metrics.to_csv(
            REPORTS_DIR / "result_v1_corrected_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if odds is None:
            conn.commit()
            print_metrics(metrics)
            return

        print("4/5 Cruzando modelo con cuotas...")
        merged = merge_predictions_odds(pred, odds)

        covered = merged[
            ["odds_home", "odds_draw", "odds_away"]
        ].notna().all(axis=1).sum()

        print(
            f"    Cobertura cuotas: {covered}/{len(merged)} "
            f"({covered/len(merged)*100:.1f}%)"
        )

        print("5/5 Backtest EV + ROI...")
        global_ev, season_ev, detail = ev_backtest(merged)

        if not global_ev.empty:
            global_ev.to_sql(
                "result_v1_ev_summary",
                conn,
                if_exists="replace",
                index=False,
            )

        if not season_ev.empty:
            season_ev.to_sql(
                "result_v1_ev_by_season",
                conn,
                if_exists="replace",
                index=False,
            )

        if not detail.empty:
            detail.to_sql(
                "result_v1_ev_predictions",
                conn,
                if_exists="replace",
                index=False,
            )

        conn.commit()

    metrics.to_csv(
        REPORTS_DIR / "result_v1_corrected_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not global_ev.empty:
        global_ev.to_csv(
            REPORTS_DIR / "result_v1_ev_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not season_ev.empty:
        season_ev.to_csv(
            REPORTS_DIR / "result_v1_ev_by_season.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not detail.empty:
        detail.to_csv(
            REPORTS_DIR / "result_v1_ev_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print_metrics(metrics)

    print()
    print("=" * 78)
    print("BACKTEST EV 1X2")
    print("=" * 78)

    if global_ev.empty:
        print("No hay suficientes partidos con cuotas cruzadas.")
    else:
        for r in global_ev.itertuples(index=False):
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
    print("=" * 78)
    print("ROI POR TEMPORADA")
    print("=" * 78)

    if season_ev.empty:
        print("Sin datos suficientes.")
    else:
        for r in season_ev.itertuples(index=False):
            print(
                f"{int(r.season)} | "
                f"EV>={r.ev_threshold*100:2.0f}% | "
                f"N={int(r.bets):3d} | "
                f"Profit={r.profit_units:+7.2f}u | "
                f"ROI={r.roi*100:+6.2f}%"
            )

    print()
    print("=" * 78)
    print("IMPORTANTE")
    print("=" * 78)
    print(
        "Este ROI usa stake fijo de 1 unidad. "
        "Todavía no aplicamos Kelly ni gestión de banca."
    )
    print(
        "El objetivo es comprobar si el edge sobrevive a cuotas reales, "
        "no maximizar beneficio retrospectivo."
    )
    print("=" * 78)


def print_metrics(metrics):
    print()
    print("=" * 78)
    print("METRICAS 1X2 CORREGIDAS")
    print("=" * 78)

    for r in metrics.itertuples(index=False):
        brier_ok = (
            "MEJORA"
            if r.brier_final < r.brier_base
            else "NO MEJORA"
        )

        log_ok = (
            "MEJORA"
            if r.logloss_final < r.logloss_base
            else "NO MEJORA"
        )

        print()
        print(f"{r.period:>6} | N={int(r.matches):3d}")
        print(
            f"  Accuracy: modelo={r.accuracy_final*100:5.1f}% | "
            f"raw={r.accuracy_raw*100:5.1f}% | "
            f"base={r.accuracy_base*100:5.1f}%"
        )
        print(
            f"  Brier:    modelo={r.brier_final:.3f} | "
            f"raw={r.brier_raw:.3f} | "
            f"base={r.brier_base:.3f} | {brier_ok}"
        )
        print(
            f"  LogLoss:  modelo={r.logloss_final:.3f} | "
            f"raw={r.logloss_raw:.3f} | "
            f"base={r.logloss_base:.3f} | {log_ok}"
        )


if __name__ == "__main__":
    main()
