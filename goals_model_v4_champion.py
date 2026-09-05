from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "eliteserien.db"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TEST_START_SEASON = 2024
MIN_TRAIN_MATCHES = 180
HALF_LIFE_DAYS = 420.0
POISSON_ALPHA = 2.0

MARKETS = {
    "over_15": "Over 1.5",
    "over_25": "Over 2.5",
    "over_35": "Over 3.5",
    "btts": "BTTS",
}

# Mucho más compacto que V1/V2/V3.
ATTACK_METRICS = (
    "goals_for",
    "xg_for",
    "shots_on_target_for",
    "points",
)

DEFENCE_METRICS = (
    "goals_against",
    "xg_against",
    "shots_on_target_against",
)


def load_team_features(conn):
    df = pd.read_sql_query(
        """
        SELECT *
        FROM team_pre_match_features
        ORDER BY utc_time, match_id, venue
        """,
        conn,
    )

    if df.empty:
        raise RuntimeError("team_pre_match_features está vacía.")

    df["utc_time"] = pd.to_datetime(
        df["utc_time"],
        utc=True,
        errors="coerce",
    )

    return df.dropna(subset=["utc_time"]).copy()


def pair_matches(team_df):
    home = team_df[team_df["venue"] == "home"].copy()
    away = team_df[team_df["venue"] == "away"].copy()

    core = [
        "match_id",
        "season",
        "round",
        "utc_time",
        "team",
        "opponent",
        "goals_for",
        "goals_against",
    ]

    selected = []

    for prefix in ("pre_last5_", "pre_last10_", "pre_season_"):
        for metric in set(ATTACK_METRICS + DEFENCE_METRICS):
            col = prefix + metric
            if col in team_df.columns:
                selected.append(col)

    # Split local/visitante: solo las variables realmente útiles.
    home_specific = [
        c for c in (
            "pre_home_last5_goals_for",
            "pre_home_last5_goals_against",
            "pre_home_last5_xg_for",
            "pre_home_last5_xg_against",
            "pre_home_last5_shots_on_target_for",
            "pre_home_last5_shots_on_target_against",
            "pre_home_last5_points",
        )
        if c in team_df.columns
    ]

    away_specific = [
        c for c in (
            "pre_away_last5_goals_for",
            "pre_away_last5_goals_against",
            "pre_away_last5_xg_for",
            "pre_away_last5_xg_against",
            "pre_away_last5_shots_on_target_for",
            "pre_away_last5_shots_on_target_against",
            "pre_away_last5_points",
        )
        if c in team_df.columns
    ]

    hcols = list(dict.fromkeys(core + selected + home_specific))
    acols = list(dict.fromkeys(core + selected + away_specific))

    home = home[hcols].copy()
    away = away[acols].copy()

    home = home.rename(
        columns={
            c: f"home_{c}"
            for c in home.columns
            if c not in ("match_id", "season", "round", "utc_time")
        }
    )

    away = away.rename(
        columns={
            c: f"away_{c}"
            for c in away.columns
            if c not in ("match_id", "season", "round", "utc_time")
        }
    )

    away = away.drop(
        columns=["season", "round", "utc_time"],
        errors="ignore",
    )

    df = home.merge(
        away,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    df["home_goals"] = pd.to_numeric(
        df["home_goals_for"], errors="coerce"
    )
    df["away_goals"] = pd.to_numeric(
        df["away_goals_for"], errors="coerce"
    )

    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    df["match_day"] = df["utc_time"].dt.floor("D")

    return df.sort_values(
        ["utc_time", "match_id"]
    ).reset_index(drop=True)


def build_long_rows(match_df):
    rows = []

    def val(row, col):
        return row.get(col, np.nan)

    for _, r in match_df.iterrows():
        # HOME TEAM GOALS
        home_row = {
            "match_id": r["match_id"],
            "season": int(r["season"]),
            "utc_time": r["utc_time"],
            "match_day": r["match_day"],
            "attack_team": r["home_team"],
            "defence_team": r["away_team"],
            "is_home": 1,
            "target_goals": float(r["home_goals"]),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
        }

        # AWAY TEAM GOALS
        away_row = {
            "match_id": r["match_id"],
            "season": int(r["season"]),
            "utc_time": r["utc_time"],
            "match_day": r["match_day"],
            "attack_team": r["away_team"],
            "defence_team": r["home_team"],
            "is_home": 0,
            "target_goals": float(r["away_goals"]),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
        }

        # Ataque propio y defensa rival.
        for w in ("last5", "last10", "season"):
            for metric in ATTACK_METRICS:
                home_row[f"att_{w}_{metric}"] = val(
                    r, f"home_pre_{w}_{metric}"
                )
                away_row[f"att_{w}_{metric}"] = val(
                    r, f"away_pre_{w}_{metric}"
                )

            for metric in DEFENCE_METRICS:
                home_row[f"oppdef_{w}_{metric}"] = val(
                    r, f"away_pre_{w}_{metric}"
                )
                away_row[f"oppdef_{w}_{metric}"] = val(
                    r, f"home_pre_{w}_{metric}"
                )

        # Venue split.
        for metric in (
            "goals_for",
            "goals_against",
            "xg_for",
            "xg_against",
            "shots_on_target_for",
            "shots_on_target_against",
            "points",
        ):
            home_row[f"venue_{metric}"] = val(
                r, f"home_pre_home_last5_{metric}"
            )
            away_row[f"venue_{metric}"] = val(
                r, f"away_pre_away_last5_{metric}"
            )

        # Interacciones futbolísticas directas.
        for w in ("last5", "last10", "season"):
            home_row[f"expected_xg_{w}"] = np.nanmean([
                home_row.get(f"att_{w}_xg_for", np.nan),
                home_row.get(f"oppdef_{w}_xg_against", np.nan),
            ])
            away_row[f"expected_xg_{w}"] = np.nanmean([
                away_row.get(f"att_{w}_xg_for", np.nan),
                away_row.get(f"oppdef_{w}_xg_against", np.nan),
            ])

            home_row[f"expected_goals_{w}"] = np.nanmean([
                home_row.get(f"att_{w}_goals_for", np.nan),
                home_row.get(f"oppdef_{w}_goals_against", np.nan),
            ])
            away_row[f"expected_goals_{w}"] = np.nanmean([
                away_row.get(f"att_{w}_goals_for", np.nan),
                away_row.get(f"oppdef_{w}_goals_against", np.nan),
            ])

        rows.extend([home_row, away_row])

    long_df = pd.DataFrame(rows)

    categorical = [
        "attack_team",
        "defence_team",
    ]

    exclude = {
        "match_id",
        "season",
        "utc_time",
        "match_day",
        "target_goals",
        "home_team",
        "away_team",
        *categorical,
    }

    numeric = [
        c for c in long_df.columns
        if c not in exclude
    ]

    numeric = [
        c for c in numeric
        if pd.to_numeric(
            long_df[c], errors="coerce"
        ).notna().mean() >= 0.40
    ]

    return long_df, numeric, categorical


def make_model(numeric, categorical):
    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("prep", prep),
            (
                "model",
                PoissonRegressor(
                    alpha=POISSON_ALPHA,
                    max_iter=3000,
                ),
            ),
        ]
    )


def recency_weights(frame, cutoff_day):
    days = (
        cutoff_day - frame["match_day"]
    ).dt.days.clip(lower=0)

    weights = np.power(
        0.5,
        days.to_numpy(dtype=float) / HALF_LIFE_DAYS,
    )

    return np.clip(weights, 0.15, 1.0)


def poisson_cdf(k, lam):
    lam = max(float(lam), 1e-6)
    term = math.exp(-lam)
    total = term

    for i in range(1, k + 1):
        term *= lam / i
        total += term

    return float(np.clip(total, 0, 1))


def market_probs(lh, la):
    total = max(float(lh + la), 1e-6)

    p0h = math.exp(-max(float(lh), 1e-6))
    p0a = math.exp(-max(float(la), 1e-6))

    return {
        "over_15": 1 - poisson_cdf(1, total),
        "over_25": 1 - poisson_cdf(2, total),
        "over_35": 1 - poisson_cdf(3, total),
        "btts": 1 - p0h - p0a + math.exp(-total),
    }


def actual_market(home_goals, away_goals):
    total = home_goals + away_goals

    return {
        "over_15": int(total > 1.5),
        "over_25": int(total > 2.5),
        "over_35": int(total > 3.5),
        "btts": int(home_goals > 0 and away_goals > 0),
    }


def walkforward_predictions(match_df, long_df, numeric, categorical):
    feature_cols = numeric + categorical
    out = []

    test_days = sorted(
        match_df.loc[
            match_df["season"] >= TEST_START_SEASON,
            "match_day"
        ].unique()
    )

    print(f"    Días OOS: {len(test_days)}")

    for i, test_day_raw in enumerate(test_days, start=1):
        test_day = pd.Timestamp(test_day_raw)

        train_long = long_df[
            long_df["match_day"] < test_day
        ].copy()

        test_matches = match_df[
            match_df["match_day"] == test_day
        ].copy()

        if (
            len(train_long) < MIN_TRAIN_MATCHES * 2
            or test_matches.empty
        ):
            continue

        model = make_model(numeric, categorical)

        weights = recency_weights(
            train_long,
            test_day,
        )

        model.fit(
            train_long[feature_cols],
            train_long["target_goals"],
            model__sample_weight=weights,
        )

        # Baseline dinámico de goles local/visitante.
        train_matches = match_df[
            match_df["match_day"] < test_day
        ].copy()

        match_weights = recency_weights(
            pd.DataFrame({
                "match_day": train_matches["match_day"]
            }),
            test_day,
        )

        base_home = float(
            np.average(
                train_matches["home_goals"],
                weights=match_weights,
            )
        )
        base_away = float(
            np.average(
                train_matches["away_goals"],
                weights=match_weights,
            )
        )

        base_probs = market_probs(
            base_home,
            base_away,
        )

        test_long = long_df[
            long_df["match_day"] == test_day
        ].copy()

        pred_goals = np.clip(
            model.predict(test_long[feature_cols]),
            0.05,
            5.5,
        )

        test_long = test_long.copy()
        test_long["pred_goals"] = pred_goals

        for _, match in test_matches.iterrows():
            side_rows = test_long[
                test_long["match_id"] == match["match_id"]
            ]

            home_side = side_rows[
                side_rows["is_home"] == 1
            ]

            away_side = side_rows[
                side_rows["is_home"] == 0
            ]

            if home_side.empty or away_side.empty:
                continue

            lh = float(home_side.iloc[0]["pred_goals"])
            la = float(away_side.iloc[0]["pred_goals"])

            probs = market_probs(lh, la)
            actuals = actual_market(
                float(match["home_goals"]),
                float(match["away_goals"]),
            )

            for market in MARKETS:
                out.append(
                    {
                        "match_id": int(match["match_id"]),
                        "season": int(match["season"]),
                        "utc_time": str(match["utc_time"]),
                        "match_day": str(test_day.date()),
                        "home_team": match["home_team"],
                        "away_team": match["away_team"],
                        "home_goals": float(match["home_goals"]),
                        "away_goals": float(match["away_goals"]),
                        "lambda_home": lh,
                        "lambda_away": la,
                        "market": market,
                        "raw_probability": float(
                            np.clip(probs[market], 0, 1)
                        ),
                        "baseline_probability": float(
                            np.clip(base_probs[market], 0, 1)
                        ),
                        "actual": int(actuals[market]),
                    }
                )

        if i % 20 == 0 or i == len(test_days):
            print(
                f"    [{i}/{len(test_days)}] {test_day.date()} "
                f"| train matches={len(train_matches)}"
            )

    return pd.DataFrame(out)


def choose_shrink_weight(history, market):
    """
    Busca cuánto confiar en el modelo frente al baseline.
    0 = solo baseline
    1 = solo modelo
    """
    h = history[
        history["market"] == market
    ].copy()

    if len(h) < 150:
        return 0.50

    y = h["actual"].to_numpy(dtype=int)
    raw = h["raw_probability"].to_numpy(dtype=float)
    base = h["baseline_probability"].to_numpy(dtype=float)

    best_w = 0.0
    best_brier = float("inf")

    for w in np.arange(0.0, 1.0001, 0.05):
        p = w * raw + (1 - w) * base
        b = brier_score_loss(y, p)

        if b < best_brier:
            best_brier = b
            best_w = float(w)

    return best_w


def apply_walkforward_shrink(pred):
    pred = pred.sort_values(
        ["season", "match_day", "match_id", "market"]
    ).copy()

    pred["shrink_weight"] = np.nan
    pred["final_probability"] = np.nan

    for season in sorted(pred["season"].unique()):
        season = int(season)

        for market in MARKETS:
            mask = (
                (pred["season"] == season)
                & (pred["market"] == market)
            )

            prior = pred[
                (pred["season"] < season)
            ].copy()

            w = choose_shrink_weight(
                prior,
                market,
            )

            raw = pred.loc[
                mask,
                "raw_probability"
            ].to_numpy(dtype=float)

            base = pred.loc[
                mask,
                "baseline_probability"
            ].to_numpy(dtype=float)

            pred.loc[mask, "shrink_weight"] = w
            pred.loc[mask, "final_probability"] = (
                w * raw + (1 - w) * base
            )

    return pred


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def summarize(pred):
    rows = []

    for market, label in MARKETS.items():
        m = pred[pred["market"] == market].copy()

        groups = [("GLOBAL", m)]
        for season, g in m.groupby("season"):
            groups.append((str(int(season)), g))

        for period, g in groups:
            if g.empty:
                continue

            y = g["actual"].to_numpy(dtype=int)
            p = g["final_probability"].to_numpy(dtype=float)
            raw = g["raw_probability"].to_numpy(dtype=float)
            base = g["baseline_probability"].to_numpy(dtype=float)

            b_final = float(brier_score_loss(y, p))
            b_raw = float(brier_score_loss(y, raw))
            b_base = float(brier_score_loss(y, base))

            rows.append(
                {
                    "market": market,
                    "label": label,
                    "period": period,
                    "matches": len(g),
                    "auc_final": safe_auc(y, p),
                    "brier_final": b_final,
                    "brier_raw": b_raw,
                    "brier_baseline": b_base,
                    "improvement_vs_baseline": b_base - b_final,
                    "avg_shrink_weight": float(
                        g["shrink_weight"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def train_final_model(long_df, numeric, categorical):
    feature_cols = numeric + categorical
    cutoff = long_df["match_day"].max() + pd.Timedelta(days=1)

    model = make_model(numeric, categorical)
    weights = recency_weights(long_df, cutoff)

    model.fit(
        long_df[feature_cols],
        long_df["target_goals"],
        model__sample_weight=weights,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        model,
        MODELS_DIR / "goals_v4_team_strength_poisson.joblib",
    )


def main():
    print()
    print("=" * 76)
    print("ELITESERIEN EDGE PRO - GOALS V4 CHAMPION")
    print("=" * 76)
    print("Modelo estructurado ataque/defensa + shrink automático.")
    print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        print("1/6 Cargando datos...")
        team_df = load_team_features(conn)

        print("2/6 Emparejando partidos...")
        match_df = pair_matches(team_df)

        print("3/6 Construyendo modelo largo ataque/defensa...")
        long_df, numeric, categorical = build_long_rows(match_df)

        print(
            f"    Partidos: {len(match_df)} | "
            f"Filas goal-model: {len(long_df)} | "
            f"Numéricas: {len(numeric)}"
        )

        print("4/6 Walk-forward diario...")
        pred = walkforward_predictions(
            match_df,
            long_df,
            numeric,
            categorical,
        )

        if pred.empty:
            raise RuntimeError("No se generaron predicciones.")

        print("5/6 Shrink automático contra baseline...")
        pred = apply_walkforward_shrink(pred)
        summary = summarize(pred)

        pred.to_sql(
            "goals_v4_backtest_predictions",
            conn,
            if_exists="replace",
            index=False,
        )

        summary.to_sql(
            "goals_v4_backtest_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.commit()

        pred.to_csv(
            REPORTS_DIR / "goals_v4_backtest_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary.to_csv(
            REPORTS_DIR / "goals_v4_backtest_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("6/6 Entrenando modelo final...")
        train_final_model(
            long_df,
            numeric,
            categorical,
        )

    print()
    print("=" * 76)
    print("BACKTEST GOALS V4")
    print("=" * 76)

    for market, label in MARKETS.items():
        print()
        print(label)
        print("-" * 76)

        g = summary[
            summary["market"] == market
        ]

        for r in g.itertuples(index=False):
            auc = (
                f"{r.auc_final:.3f}"
                if pd.notna(r.auc_final)
                else "N/A"
            )

            verdict = (
                "MEJORA"
                if r.improvement_vs_baseline > 0
                else "NO MEJORA"
            )

            print(
                f"{r.period:>6} | "
                f"N={int(r.matches):3d} | "
                f"AUC={auc} | "
                f"Brier={r.brier_final:.3f} | "
                f"Base={r.brier_baseline:.3f} | "
                f"w={r.avg_shrink_weight:.2f} | "
                f"{verdict}"
            )

    print()
    print("=" * 76)
    print("OBJETIVO")
    print("=" * 76)
    print(
        "Queremos sobre todo que Over 2.5/3.5 mantengan AUC > 0.54 "
        "y Brier mejor que baseline de forma consistente."
    )
    print(
        "Si V4 tampoco lo consigue, dejamos de exprimir goles y pasamos "
        "a corners/tarjetas, donde nuestros datos pueden tener más edge."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
