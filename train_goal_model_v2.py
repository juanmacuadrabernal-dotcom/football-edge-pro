from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "eliteserien.db"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MARKETS = {
    "over_15": "Over 1.5",
    "over_25": "Over 2.5",
    "over_35": "Over 3.5",
    "btts": "BTTS",
}


def load_features(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT *
        FROM team_pre_match_features
        ORDER BY utc_time, match_id, venue
        """,
        conn,
    )

    df["utc_time"] = pd.to_datetime(
        df["utc_time"],
        utc=True,
        errors="coerce",
    )

    return df.dropna(subset=["utc_time"]).copy()


def build_match_dataset(team_df: pd.DataFrame):
    home = team_df[team_df["venue"] == "home"].copy()
    away = team_df[team_df["venue"] == "away"].copy()

    base = [
        "match_id",
        "season",
        "round",
        "utc_time",
        "team",
        "opponent",
        "goals_for",
        "goals_against",
    ]

    compact = []

    for side_prefix in ("pre_last5_", "pre_last10_", "pre_season_"):
        for metric in (
            "goals_for",
            "goals_against",
            "xg_for",
            "xg_against",
            "shots_on_target_for",
            "shots_on_target_against",
            "corners_for",
            "corners_against",
            "points",
        ):
            col = side_prefix + metric
            if col in team_df.columns:
                compact.append(col)

    home_venue = [
        c for c in (
            "pre_home_last5_goals_for",
            "pre_home_last5_goals_against",
            "pre_home_last5_xg_for",
            "pre_home_last5_xg_against",
            "pre_home_last5_shots_on_target_for",
            "pre_home_last5_shots_on_target_against",
            "pre_home_last5_corners_for",
            "pre_home_last5_corners_against",
            "pre_home_last5_points",
        )
        if c in team_df.columns
    ]

    away_venue = [
        c for c in (
            "pre_away_last5_goals_for",
            "pre_away_last5_goals_against",
            "pre_away_last5_xg_for",
            "pre_away_last5_xg_against",
            "pre_away_last5_shots_on_target_for",
            "pre_away_last5_shots_on_target_against",
            "pre_away_last5_corners_for",
            "pre_away_last5_corners_against",
            "pre_away_last5_points",
        )
        if c in team_df.columns
    ]

    common_extra = [
        c for c in (
            "history_matches_before",
            "season_matches_before",
        )
        if c in team_df.columns
    ]

    hcols = list(dict.fromkeys(base + compact + home_venue + common_extra))
    acols = list(dict.fromkeys(base + compact + away_venue + common_extra))

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
        df["home_goals_for"],
        errors="coerce",
    )

    df["away_goals"] = pd.to_numeric(
        df["away_goals_for"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["home_goals", "away_goals"]
    ).copy()

    # Muy importante: no usamos el resultado actual como feature.
    feature_cols = [
        c for c in df.columns
        if (
            c.startswith("home_pre_")
            or c.startswith("away_pre_")
            or c in (
                "home_history_matches_before",
                "away_history_matches_before",
                "home_season_matches_before",
                "away_season_matches_before",
            )
        )
    ]

    # Variables combinadas compactas.
    def add_sum(name, a, b):
        if a in df.columns and b in df.columns:
            df[name] = (
                pd.to_numeric(df[a], errors="coerce")
                + pd.to_numeric(df[b], errors="coerce")
            )
            feature_cols.append(name)

    def add_diff(name, a, b):
        if a in df.columns and b in df.columns:
            df[name] = (
                pd.to_numeric(df[a], errors="coerce")
                - pd.to_numeric(df[b], errors="coerce")
            )
            feature_cols.append(name)

    for w in ("last5", "last10", "season"):
        add_sum(
            f"sum_{w}_xg_attack",
            f"home_pre_{w}_xg_for",
            f"away_pre_{w}_xg_for",
        )
        add_sum(
            f"sum_{w}_xg_defence",
            f"home_pre_{w}_xg_against",
            f"away_pre_{w}_xg_against",
        )
        add_sum(
            f"sum_{w}_goals_attack",
            f"home_pre_{w}_goals_for",
            f"away_pre_{w}_goals_for",
        )
        add_diff(
            f"diff_{w}_form",
            f"home_pre_{w}_points",
            f"away_pre_{w}_points",
        )

    feature_cols = list(dict.fromkeys(feature_cols))

    # Eliminamos columnas casi totalmente vacías.
    keep = []

    for col in feature_cols:
        non_null = df[col].notna().mean()
        if non_null >= 0.35:
            keep.append(col)

    return df, keep


def make_poisson(features):
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),
                        (
                            "scale",
                            StandardScaler(),
                        ),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )

    model = PoissonRegressor(
        alpha=1.5,
        max_iter=3000,
    )

    return Pipeline(
        [
            ("prep", prep),
            ("model", model),
        ]
    )


def recency_weights(train):
    newest = int(train["season"].max())

    age = (
        newest
        - pd.to_numeric(train["season"], errors="coerce")
    ).fillna(0)

    return np.power(
        0.72,
        age.to_numpy(dtype=float),
    )


def poisson_cdf(k: int, lam: float) -> float:
    lam = max(float(lam), 0.0001)

    total = 0.0
    term = math.exp(-lam)
    total += term

    for i in range(1, k + 1):
        term *= lam / i
        total += term

    return min(max(total, 0.0), 1.0)


def market_probs(lambda_home, lambda_away):
    lt = max(lambda_home + lambda_away, 0.0001)

    p_over_15 = 1.0 - poisson_cdf(1, lt)
    p_over_25 = 1.0 - poisson_cdf(2, lt)
    p_over_35 = 1.0 - poisson_cdf(3, lt)

    p_home_zero = math.exp(-max(lambda_home, 0.0001))
    p_away_zero = math.exp(-max(lambda_away, 0.0001))

    p_btts = (
        1.0
        - p_home_zero
        - p_away_zero
        + math.exp(-lt)
    )

    return {
        "over_15": min(max(p_over_15, 0), 1),
        "over_25": min(max(p_over_25, 0), 1),
        "over_35": min(max(p_over_35, 0), 1),
        "btts": min(max(p_btts, 0), 1),
    }


def targets(frame):
    total = frame["home_goals"] + frame["away_goals"]

    return {
        "over_15": (total > 1.5).astype(int).to_numpy(),
        "over_25": (total > 2.5).astype(int).to_numpy(),
        "over_35": (total > 3.5).astype(int).to_numpy(),
        "btts": (
            (frame["home_goals"] > 0)
            & (frame["away_goals"] > 0)
        ).astype(int).to_numpy(),
    }


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None

    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def safe_logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)

    try:
        return float(
            log_loss(
                y,
                p,
                labels=[0, 1],
            )
        )
    except Exception:
        return None


def run_backtest(df, feature_cols):
    predictions = []
    summaries = []

    seasons = sorted(
        int(x)
        for x in df["season"].dropna().unique()
        if int(x) >= 2024
    )

    for test_season in seasons:
        train = df[df["season"] < test_season].copy()
        test = df[df["season"] == test_season].copy()

        if len(train) < 150 or len(test) < 20:
            continue

        home_model = make_poisson(feature_cols)
        away_model = make_poisson(feature_cols)

        weights = recency_weights(train)

        home_model.fit(
            train[feature_cols],
            train["home_goals"],
            model__sample_weight=weights,
        )

        away_model.fit(
            train[feature_cols],
            train["away_goals"],
            model__sample_weight=weights,
        )

        lambda_home = np.clip(
            home_model.predict(test[feature_cols]),
            0.05,
            5.5,
        )

        lambda_away = np.clip(
            away_model.predict(test[feature_cols]),
            0.05,
            5.5,
        )

        y_targets = targets(test)

        # Baseline = media histórica del training.
        train_targets = targets(train)

        market_prob_matrix = {
            key: []
            for key in MARKETS
        }

        for lh, la in zip(lambda_home, lambda_away):
            probs = market_probs(lh, la)

            for key in MARKETS:
                market_prob_matrix[key].append(
                    probs[key]
                )

        for market in MARKETS:
            p = np.asarray(
                market_prob_matrix[market],
                dtype=float,
            )

            y = y_targets[market]

            baseline_prob = float(
                np.mean(train_targets[market])
            )

            baseline = np.repeat(
                baseline_prob,
                len(test),
            )

            brier_model = float(
                brier_score_loss(y, p)
            )

            brier_baseline = float(
                brier_score_loss(y, baseline)
            )

            summaries.append(
                {
                    "market": market,
                    "test_season": test_season,
                    "train_matches": len(train),
                    "test_matches": len(test),
                    "actual_rate": float(np.mean(y)),
                    "mean_model_prob": float(np.mean(p)),
                    "auc": safe_auc(y, p),
                    "brier_model": brier_model,
                    "brier_baseline": brier_baseline,
                    "brier_improvement": (
                        brier_baseline - brier_model
                    ),
                    "logloss_model": safe_logloss(y, p),
                    "logloss_baseline": safe_logloss(y, baseline),
                }
            )

            for i, row in enumerate(
                test.itertuples(index=False)
            ):
                predictions.append(
                    {
                        "match_id": row.match_id,
                        "season": test_season,
                        "utc_time": str(row.utc_time),
                        "home_team": row.home_team,
                        "away_team": row.away_team,
                        "home_goals": row.home_goals,
                        "away_goals": row.away_goals,
                        "lambda_home": float(lambda_home[i]),
                        "lambda_away": float(lambda_away[i]),
                        "market": market,
                        "probability": float(p[i]),
                        "actual": int(y[i]),
                    }
                )

        summaries.append(
            {
                "market": "__goals_mae__",
                "test_season": test_season,
                "train_matches": len(train),
                "test_matches": len(test),
                "actual_rate": None,
                "mean_model_prob": None,
                "auc": None,
                "brier_model": float(
                    mean_absolute_error(
                        test["home_goals"],
                        lambda_home,
                    )
                ),
                "brier_baseline": float(
                    mean_absolute_error(
                        test["away_goals"],
                        lambda_away,
                    )
                ),
                "brier_improvement": None,
                "logloss_model": None,
                "logloss_baseline": None,
            }
        )

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(summaries),
    )


def train_final(df, feature_cols):
    weights = recency_weights(df)

    home_model = make_poisson(feature_cols)
    away_model = make_poisson(feature_cols)

    home_model.fit(
        df[feature_cols],
        df["home_goals"],
        model__sample_weight=weights,
    )

    away_model.fit(
        df[feature_cols],
        df["away_goals"],
        model__sample_weight=weights,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        home_model,
        MODELS_DIR / "goals_v2_home_poisson.joblib",
    )

    joblib.dump(
        away_model,
        MODELS_DIR / "goals_v2_away_poisson.joblib",
    )

    metadata = {
        "model_type": "PoissonRegressor",
        "features": feature_cols,
        "matches": len(df),
        "seasons": sorted(
            int(x)
            for x in df["season"].unique()
        ),
    }

    (
        MODELS_DIR
        / "goals_v2_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    print()
    print("=" * 72)
    print("ELITESERIEN EDGE PRO - MODELO GOLES V2")
    print("=" * 72)
    print(
        "Predice goles local/visitante con Poisson y deriva "
        "Over/BTTS."
    )
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        print("1/5 Cargando datos...")
        team_df = load_features(conn)

        print("2/5 Construyendo dataset compacto...")
        df, features = build_match_dataset(team_df)

        print(
            f"    Partidos: {len(df)} | "
            f"Features utilizadas: {len(features)}"
        )

        print("3/5 Backtesting 2024 -> 2026...")
        predictions, summary = run_backtest(
            df,
            features,
        )

        print("4/5 Guardando resultados...")

        predictions.to_sql(
            "goals_v2_backtest_predictions",
            conn,
            if_exists="replace",
            index=False,
        )

        summary.to_sql(
            "goals_v2_backtest_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.commit()

        predictions.to_csv(
            REPORTS_DIR
            / "goals_v2_backtest_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary.to_csv(
            REPORTS_DIR
            / "goals_v2_backtest_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("5/5 Entrenando modelo final...")
        train_final(
            df,
            features,
        )

    print()
    print("=" * 72)
    print("BACKTEST GOLES V2")
    print("=" * 72)

    market_rows = summary[
        summary["market"] != "__goals_mae__"
    ]

    for market in MARKETS:
        print()
        print(MARKETS[market])
        print("-" * 72)

        g = market_rows[
            market_rows["market"] == market
        ]

        for r in g.itertuples(index=False):
            auc = (
                f"{r.auc:.3f}"
                if pd.notna(r.auc)
                else "N/A"
            )

            verdict = (
                "MEJORA"
                if r.brier_improvement > 0
                else "NO MEJORA"
            )

            print(
                f"Test {int(r.test_season)} | "
                f"AUC={auc} | "
                f"Brier modelo={r.brier_model:.3f} | "
                f"Baseline={r.brier_baseline:.3f} | "
                f"{verdict}"
            )

    print()
    print("=" * 72)
    print("CRITERIO")
    print("=" * 72)
    print(
        "Brier menor que baseline = el modelo aporta información real."
    )
    print(
        "AUC > 0.50 = ordena mejor que el azar; "
        "idealmente queremos bastante más."
    )
    print("=" * 72)

    print()
    print("Archivos creados:")
    print("  models\\goals_v2_home_poisson.joblib")
    print("  models\\goals_v2_away_poisson.joblib")
    print("  reports\\goals_v2_backtest_summary.csv")


if __name__ == "__main__":
    main()
