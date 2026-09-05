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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "eliteserien.db"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TEST_START_SEASON = 2024
MIN_TRAIN_MATCHES = 180

MARKETS = {
    "over_15": "Over 1.5",
    "over_25": "Over 2.5",
    "over_35": "Over 3.5",
    "btts": "BTTS",
}

CORE_METRICS = (
    "goals_for",
    "goals_against",
    "xg_for",
    "xg_against",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "points",
)


def load_team_data(conn):
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


def build_dataset(team_df):
    home = team_df[team_df["venue"] == "home"].copy()
    away = team_df[team_df["venue"] == "away"].copy()

    common = [
        "match_id",
        "season",
        "round",
        "utc_time",
        "team",
        "opponent",
        "goals_for",
        "goals_against",
    ]

    rolling = []
    for prefix in ("pre_last5_", "pre_last10_", "pre_season_"):
        for metric in CORE_METRICS:
            col = prefix + metric
            if col in team_df.columns:
                rolling.append(col)

    home_venue = [
        f"pre_home_last5_{m}"
        for m in CORE_METRICS
        if f"pre_home_last5_{m}" in team_df.columns
    ]

    away_venue = [
        f"pre_away_last5_{m}"
        for m in CORE_METRICS
        if f"pre_away_last5_{m}" in team_df.columns
    ]

    extra = [
        c for c in ("history_matches_before", "season_matches_before")
        if c in team_df.columns
    ]

    hcols = list(dict.fromkeys(common + rolling + home_venue + extra))
    acols = list(dict.fromkeys(common + rolling + away_venue + extra))

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

    # Día de kickoff. Todo lo que ocurra ese mismo día se predice
    # sin usar resultados de otros partidos del día.
    df["match_day"] = df["utc_time"].dt.floor("D")

    numeric_features = [
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

    # Interacciones futbolísticas sencillas:
    # ataque de un equipo frente a defensa reciente del rival.
    def mean_pair(name, a, b):
        if a in df.columns and b in df.columns:
            df[name] = (
                pd.to_numeric(df[a], errors="coerce")
                + pd.to_numeric(df[b], errors="coerce")
            ) / 2.0
            numeric_features.append(name)

    for w in ("last5", "last10", "season"):
        mean_pair(
            f"home_expected_xg_{w}",
            f"home_pre_{w}_xg_for",
            f"away_pre_{w}_xg_against",
        )
        mean_pair(
            f"away_expected_xg_{w}",
            f"away_pre_{w}_xg_for",
            f"home_pre_{w}_xg_against",
        )
        mean_pair(
            f"home_expected_goals_{w}",
            f"home_pre_{w}_goals_for",
            f"away_pre_{w}_goals_against",
        )
        mean_pair(
            f"away_expected_goals_{w}",
            f"away_pre_{w}_goals_for",
            f"home_pre_{w}_goals_against",
        )
        mean_pair(
            f"home_expected_sot_{w}",
            f"home_pre_{w}_shots_on_target_for",
            f"away_pre_{w}_shots_on_target_against",
        )
        mean_pair(
            f"away_expected_sot_{w}",
            f"away_pre_{w}_shots_on_target_for",
            f"home_pre_{w}_shots_on_target_against",
        )

    numeric_features = list(dict.fromkeys(numeric_features))

    # Quitamos columnas demasiado vacías.
    numeric_features = [
        c for c in numeric_features
        if df[c].notna().mean() >= 0.40
    ]

    categorical_features = [
        "home_team",
        "away_team",
    ]

    return df.sort_values(
        ["utc_time", "match_id"]
    ).reset_index(drop=True), numeric_features, categorical_features


def make_model(numeric_features, categorical_features):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            (
                "imputer",
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
    )

    prep = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )

    reg = PoissonRegressor(
        alpha=1.0,
        max_iter=3000,
    )

    return Pipeline(
        steps=[
            ("prep", prep),
            ("model", reg),
        ]
    )


def sample_weights(train, cutoff_day):
    days_old = (
        cutoff_day
        - train["match_day"]
    ).dt.days.clip(lower=0)

    # Media vida aproximada de 420 días.
    weights = np.power(
        0.5,
        days_old.to_numpy(dtype=float) / 420.0,
    )

    # Los partidos muy antiguos siguen contando un poco.
    return np.clip(weights, 0.20, 1.0)


def poisson_cdf(k, lam):
    lam = max(float(lam), 1e-6)
    term = math.exp(-lam)
    total = term

    for i in range(1, k + 1):
        term *= lam / i
        total += term

    return min(max(total, 0.0), 1.0)


def probs_from_lambdas(home_lam, away_lam):
    total = max(home_lam + away_lam, 1e-6)

    over15 = 1.0 - poisson_cdf(1, total)
    over25 = 1.0 - poisson_cdf(2, total)
    over35 = 1.0 - poisson_cdf(3, total)

    home_zero = math.exp(-max(home_lam, 1e-6))
    away_zero = math.exp(-max(away_lam, 1e-6))

    btts = (
        1.0
        - home_zero
        - away_zero
        + math.exp(-total)
    )

    return {
        "over_15": float(np.clip(over15, 0, 1)),
        "over_25": float(np.clip(over25, 0, 1)),
        "over_35": float(np.clip(over35, 0, 1)),
        "btts": float(np.clip(btts, 0, 1)),
    }


def market_actuals(frame):
    total = (
        frame["home_goals"].to_numpy()
        + frame["away_goals"].to_numpy()
    )

    return {
        "over_15": (total > 1.5).astype(int),
        "over_25": (total > 2.5).astype(int),
        "over_35": (total > 3.5).astype(int),
        "btts": (
            (frame["home_goals"].to_numpy() > 0)
            & (frame["away_goals"].to_numpy() > 0)
        ).astype(int),
    }


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def safe_logloss(y, p):
    p = np.clip(
        np.asarray(p, dtype=float),
        1e-6,
        1 - 1e-6,
    )

    try:
        return float(
            log_loss(y, p, labels=[0, 1])
        )
    except Exception:
        return np.nan


def run_walkforward(df, numeric_features, categorical_features):
    prediction_rows = []

    test_days = sorted(
        d for d in df.loc[
            df["season"] >= TEST_START_SEASON,
            "match_day"
        ].dropna().unique()
    )

    feature_cols = numeric_features + categorical_features

    print(f"    Días de predicción: {len(test_days)}")

    for n, test_day_raw in enumerate(test_days, start=1):
        test_day = pd.Timestamp(test_day_raw)

        train = df[
            df["match_day"] < test_day
        ].copy()

        test = df[
            df["match_day"] == test_day
        ].copy()

        if (
            len(train) < MIN_TRAIN_MATCHES
            or test.empty
        ):
            continue

        home_model = make_model(
            numeric_features,
            categorical_features,
        )
        away_model = make_model(
            numeric_features,
            categorical_features,
        )

        weights = sample_weights(
            train,
            test_day,
        )

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

        home_lambda = np.clip(
            home_model.predict(test[feature_cols]),
            0.05,
            5.5,
        )

        away_lambda = np.clip(
            away_model.predict(test[feature_cols]),
            0.05,
            5.5,
        )

        actual = market_actuals(test)

        # Baseline fuerte: Poisson usando la media de goles de
        # TODOS los partidos disponibles antes de ese día.
        baseline_home_lambda = float(
            np.average(
                train["home_goals"],
                weights=weights,
            )
        )
        baseline_away_lambda = float(
            np.average(
                train["away_goals"],
                weights=weights,
            )
        )

        baseline_probs = probs_from_lambdas(
            baseline_home_lambda,
            baseline_away_lambda,
        )

        for i, row in enumerate(test.itertuples(index=False)):
            model_probs = probs_from_lambdas(
                home_lambda[i],
                away_lambda[i],
            )

            for market in MARKETS:
                prediction_rows.append(
                    {
                        "match_id": row.match_id,
                        "season": int(row.season),
                        "utc_time": str(row.utc_time),
                        "match_day": str(test_day.date()),
                        "home_team": row.home_team,
                        "away_team": row.away_team,
                        "home_goals": float(row.home_goals),
                        "away_goals": float(row.away_goals),
                        "lambda_home": float(home_lambda[i]),
                        "lambda_away": float(away_lambda[i]),
                        "market": market,
                        "probability": model_probs[market],
                        "baseline_probability": baseline_probs[market],
                        "actual": int(actual[market][i]),
                        "train_matches": len(train),
                    }
                )

        if n % 20 == 0 or n == len(test_days):
            print(
                f"    [{n}/{len(test_days)}] "
                f"{test_day.date()} | train={len(train)}"
            )

    return pd.DataFrame(prediction_rows)


def summarize(pred):
    rows = []

    for market in MARKETS:
        m = pred[pred["market"] == market].copy()

        if m.empty:
            continue

        # Global.
        groups = [("GLOBAL", m)]

        for season, g in m.groupby("season"):
            groups.append((str(int(season)), g))

        for label, g in groups:
            y = g["actual"].to_numpy(dtype=int)
            p = g["probability"].to_numpy(dtype=float)
            bp = g["baseline_probability"].to_numpy(dtype=float)

            brier_model = float(
                brier_score_loss(y, p)
            )
            brier_base = float(
                brier_score_loss(y, bp)
            )

            rows.append(
                {
                    "market": market,
                    "period": label,
                    "matches": len(g),
                    "actual_rate": float(np.mean(y)),
                    "mean_probability": float(np.mean(p)),
                    "auc": safe_auc(y, p),
                    "brier_model": brier_model,
                    "brier_baseline": brier_base,
                    "brier_improvement": (
                        brier_base - brier_model
                    ),
                    "logloss_model": safe_logloss(y, p),
                    "logloss_baseline": safe_logloss(y, bp),
                }
            )

    return pd.DataFrame(rows)


def final_models(df, numeric_features, categorical_features):
    feature_cols = numeric_features + categorical_features

    cutoff = df["match_day"].max() + pd.Timedelta(days=1)

    weights = sample_weights(
        df,
        cutoff,
    )

    home_model = make_model(
        numeric_features,
        categorical_features,
    )
    away_model = make_model(
        numeric_features,
        categorical_features,
    )

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

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        home_model,
        MODELS_DIR / "goals_v3_home_poisson.joblib",
    )
    joblib.dump(
        away_model,
        MODELS_DIR / "goals_v3_away_poisson.joblib",
    )


def main():
    print()
    print("=" * 72)
    print("ELITESERIEN EDGE PRO - GOALS MODEL V3")
    print("=" * 72)
    print("Walk-forward diario + fuerza de equipos + Poisson.")
    print()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DB_PATH) as conn:
        print("1/5 Cargando features...")
        team_df = load_team_data(conn)

        print("2/5 Construyendo dataset...")
        df, numeric_features, categorical_features = build_dataset(
            team_df
        )

        print(
            f"    Partidos: {len(df)} | "
            f"Numéricas: {len(numeric_features)} | "
            f"Categóricas: {len(categorical_features)}"
        )

        print("3/5 Walk-forward realista...")
        pred = run_walkforward(
            df,
            numeric_features,
            categorical_features,
        )

        if pred.empty:
            raise RuntimeError(
                "No se generaron predicciones."
            )

        print("4/5 Evaluando...")
        summary = summarize(pred)

        pred.to_sql(
            "goals_v3_backtest_predictions",
            conn,
            if_exists="replace",
            index=False,
        )
        summary.to_sql(
            "goals_v3_backtest_summary",
            conn,
            if_exists="replace",
            index=False,
        )
        conn.commit()

        pred.to_csv(
            REPORTS_DIR / "goals_v3_backtest_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary.to_csv(
            REPORTS_DIR / "goals_v3_backtest_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("5/5 Entrenando modelos finales...")
        final_models(
            df,
            numeric_features,
            categorical_features,
        )

    print()
    print("=" * 72)
    print("BACKTEST GOALS V3")
    print("=" * 72)

    for market, label in MARKETS.items():
        print()
        print(label)
        print("-" * 72)

        g = summary[
            summary["market"] == market
        ]

        for r in g.itertuples(index=False):
            auc_text = (
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
                f"{r.period:>6} | "
                f"N={int(r.matches):3d} | "
                f"AUC={auc_text} | "
                f"Brier={r.brier_model:.3f} | "
                f"Base={r.brier_baseline:.3f} | "
                f"{verdict}"
            )

    print()
    print("=" * 72)
    print("LECTURA")
    print("=" * 72)
    print(
        "El baseline también es Poisson y se actualiza con "
        "todos los partidos anteriores."
    )
    print(
        "Por tanto, MEJORA significa superar una referencia "
        "más exigente que simplemente usar la frecuencia histórica."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
