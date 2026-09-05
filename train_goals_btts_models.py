from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print()
    print("=" * 72)
    print("FALTA SCIKIT-LEARN")
    print("=" * 72)
    print("Ejecuta en CMD:")
    print()
    print("pip install scikit-learn joblib")
    print()
    print("Después vuelve a ejecutar:")
    print()
    print("python train_goals_btts_models.py")
    print("=" * 72)
    raise SystemExit(1)


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "eliteserien.db"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MARKETS = {
    "over_15": "Over 1.5 goles",
    "over_25": "Over 2.5 goles",
    "over_35": "Over 3.5 goles",
    "btts": "Ambos marcan (BTTS)",
}

# Primera versión deliberadamente estable e interpretable.
# Después compararemos contra modelos más complejos.
METRICS = [
    "goals_for",
    "goals_against",
    "xg_for",
    "xg_against",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "points",
    "win",
    "btts",
    "over_25",
]

ROLLING_PREFIXES = [
    "pre_last5_",
    "pre_last10_",
    "pre_season_",
]

VENUE_METRICS = [
    "goals_for",
    "goals_against",
    "xg_for",
    "xg_against",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "points",
    "win",
    "btts",
    "over_25",
]


def load_team_features(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT *
        FROM team_pre_match_features
        ORDER BY utc_time, match_id, venue
        """,
        conn,
    )

    if df.empty:
        raise RuntimeError(
            "La tabla team_pre_match_features está vacía. "
            "Ejecuta build_team_stats.py primero."
        )

    df["utc_time"] = pd.to_datetime(
        df["utc_time"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["utc_time"]).copy()

    return df


def build_match_dataset(team_df: pd.DataFrame) -> pd.DataFrame:
    home = team_df[team_df["venue"] == "home"].copy()
    away = team_df[team_df["venue"] == "away"].copy()

    if home.empty or away.empty:
        raise RuntimeError("No encuentro filas home/away.")

    base_cols = [
        "match_id",
        "season",
        "round",
        "utc_time",
        "team",
        "opponent",
        "goals_for",
        "goals_against",
    ]

    # Features base para cualquier venue.
    feature_cols = [
        "history_matches_before",
        "season_matches_before",
    ]

    for prefix in ROLLING_PREFIXES:
        for metric in METRICS:
            col = f"{prefix}{metric}"
            if col in team_df.columns:
                feature_cols.append(col)

    # Split específico local/visitante.
    home_venue_cols = [
        f"pre_home_last5_{m}"
        for m in VENUE_METRICS
        if f"pre_home_last5_{m}" in team_df.columns
    ]

    away_venue_cols = [
        f"pre_away_last5_{m}"
        for m in VENUE_METRICS
        if f"pre_away_last5_{m}" in team_df.columns
    ]

    home_keep = list(dict.fromkeys(base_cols + feature_cols + home_venue_cols))
    away_keep = list(dict.fromkeys(base_cols + feature_cols + away_venue_cols))

    home = home[home_keep].copy()
    away = away[away_keep].copy()

    # Renombrar para que el modelo vea claramente cada lado.
    home_rename = {}
    away_rename = {}

    for col in home.columns:
        if col not in ("match_id", "season", "round", "utc_time"):
            home_rename[col] = f"home_{col}"

    for col in away.columns:
        if col not in ("match_id", "season", "round", "utc_time"):
            away_rename[col] = f"away_{col}"

    home = home.rename(columns=home_rename)
    away = away.rename(columns=away_rename)

    # Evitar duplicar season/time desde away.
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

    # Targets reales, solo usados como etiqueta.
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

    total = df["home_goals"] + df["away_goals"]

    df["target_over_15"] = (total > 1.5).astype(int)
    df["target_over_25"] = (total > 2.5).astype(int)
    df["target_over_35"] = (total > 3.5).astype(int)
    df["target_btts"] = (
        (df["home_goals"] > 0)
        & (df["away_goals"] > 0)
    ).astype(int)

    # Features del modelo = exclusivamente columnas PRE.
    model_features = [
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

    # Añadimos algunas diferencias útiles para un modelo lineal.
    def add_diff(name, hcol, acol):
        if hcol in df.columns and acol in df.columns:
            df[name] = (
                pd.to_numeric(df[hcol], errors="coerce")
                - pd.to_numeric(df[acol], errors="coerce")
            )
            model_features.append(name)

    def add_sum(name, hcol, acol):
        if hcol in df.columns and acol in df.columns:
            df[name] = (
                pd.to_numeric(df[hcol], errors="coerce")
                + pd.to_numeric(df[acol], errors="coerce")
            )
            model_features.append(name)

    for window in ("last5", "last10", "season"):
        add_diff(
            f"diff_{window}_xg_attack",
            f"home_pre_{window}_xg_for",
            f"away_pre_{window}_xg_for",
        )

        add_sum(
            f"sum_{window}_xg_attack",
            f"home_pre_{window}_xg_for",
            f"away_pre_{window}_xg_for",
        )

        add_sum(
            f"sum_{window}_xg_conceded",
            f"home_pre_{window}_xg_against",
            f"away_pre_{window}_xg_against",
        )

        add_sum(
            f"sum_{window}_goals_attack",
            f"home_pre_{window}_goals_for",
            f"away_pre_{window}_goals_for",
        )

        add_sum(
            f"sum_{window}_goals_conceded",
            f"home_pre_{window}_goals_against",
            f"away_pre_{window}_goals_against",
        )

        add_sum(
            f"sum_{window}_sot",
            f"home_pre_{window}_shots_on_target_for",
            f"away_pre_{window}_shots_on_target_for",
        )

    model_features = list(dict.fromkeys(model_features))

    if len(model_features) < 20:
        raise RuntimeError(
            f"Muy pocas features detectadas: {len(model_features)}"
        )

    return df, model_features


def make_model(feature_columns: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                feature_columns,
            )
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        C=0.70,
        max_iter=3000,
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("prep", prep),
            ("model", clf),
        ]
    )


def season_weights(
    train: pd.DataFrame,
) -> np.ndarray:
    """
    Peso temporal:
    temporada más reciente del training = 1.00
    una temporada atrás = 0.72
    dos atrás = 0.52 aprox.
    """
    max_season = int(train["season"].max())

    age = (
        max_season
        - pd.to_numeric(train["season"], errors="coerce")
    ).fillna(0)

    weights = np.power(0.72, age.to_numpy(dtype=float))

    return weights


def safe_auc(y_true, prob):
    if len(np.unique(y_true)) < 2:
        return None

    try:
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return None


def safe_logloss(y_true, prob):
    try:
        return float(
            log_loss(
                y_true,
                prob,
                labels=[0, 1],
            )
        )
    except Exception:
        return None


def backtest_market(
    df: pd.DataFrame,
    feature_columns: list[str],
    market: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_col = f"target_{market}"

    test_seasons = sorted(
        s for s in df["season"].dropna().unique()
        if int(s) >= 2024
    )

    predictions = []
    summaries = []

    for test_season in test_seasons:
        test_season = int(test_season)

        train = df[
            df["season"] < test_season
        ].copy()

        test = df[
            df["season"] == test_season
        ].copy()

        if len(train) < 150 or len(test) < 20:
            continue

        X_train = train[feature_columns]
        y_train = train[target_col].astype(int)

        X_test = test[feature_columns]
        y_test = test[target_col].astype(int)

        if y_train.nunique() < 2:
            continue

        model = make_model(feature_columns)
        weights = season_weights(train)

        model.fit(
            X_train,
            y_train,
            model__sample_weight=weights,
        )

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.50).astype(int)

        fold = pd.DataFrame(
            {
                "match_id": test["match_id"].values,
                "season": test["season"].values,
                "utc_time": test["utc_time"].astype(str).values,
                "home_team": test["home_team"].values,
                "away_team": test["away_team"].values,
                "market": market,
                "probability": prob,
                "prediction": pred,
                "actual": y_test.to_numpy(),
                "home_goals": test["home_goals"].values,
                "away_goals": test["away_goals"].values,
            }
        )

        predictions.append(fold)

        summaries.append(
            {
                "market": market,
                "test_season": test_season,
                "train_rows": len(train),
                "test_rows": len(test),
                "base_rate": float(y_test.mean()),
                "mean_probability": float(np.mean(prob)),
                "accuracy": float(
                    accuracy_score(y_test, pred)
                ),
                "brier": float(
                    brier_score_loss(y_test, prob)
                ),
                "log_loss": safe_logloss(y_test, prob),
                "auc": safe_auc(y_test, prob),
            }
        )

    pred_df = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame()
    )

    summary_df = pd.DataFrame(summaries)

    return pred_df, summary_df


def probability_buckets(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    df = predictions.copy()

    bins = [
        0.00,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        1.01,
    ]

    labels = [
        "<50%",
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-75%",
        "75-80%",
        "80-85%",
        "85-90%",
        "90%+",
    ]

    df["prob_bucket"] = pd.cut(
        df["probability"],
        bins=bins,
        labels=labels,
        right=False,
    )

    rows = []

    for (market, bucket), g in df.groupby(
        ["market", "prob_bucket"],
        observed=True,
    ):
        if len(g) == 0:
            continue

        rows.append(
            {
                "market": market,
                "prob_bucket": str(bucket),
                "bets": len(g),
                "avg_model_prob": float(
                    g["probability"].mean()
                ),
                "actual_hit_rate": float(
                    g["actual"].mean()
                ),
                "calibration_gap": float(
                    g["actual"].mean()
                    - g["probability"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def train_final_models(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "features": feature_columns,
        "markets": {},
    }

    for market in MARKETS:
        target_col = f"target_{market}"

        y = df[target_col].astype(int)

        if y.nunique() < 2:
            continue

        model = make_model(feature_columns)

        weights = season_weights(df)

        model.fit(
            df[feature_columns],
            y,
            model__sample_weight=weights,
        )

        model_path = (
            MODELS_DIR
            / f"goals_{market}_logistic.joblib"
        )

        joblib.dump(model, model_path)

        metadata["markets"][market] = {
            "label": MARKETS[market],
            "model_path": str(
                model_path.relative_to(BASE_DIR)
            ),
            "training_matches": len(df),
            "positive_rate": float(y.mean()),
        }

    metadata_path = (
        MODELS_DIR
        / "goals_btts_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata


def save_results(
    conn: sqlite3.Connection,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    buckets: pd.DataFrame,
) -> None:
    if not predictions.empty:
        predictions.to_sql(
            "model_backtest_predictions",
            conn,
            if_exists="replace",
            index=False,
        )

    if not summary.empty:
        summary.to_sql(
            "model_backtest_summary",
            conn,
            if_exists="replace",
            index=False,
        )

    if not buckets.empty:
        buckets.to_sql(
            "model_probability_buckets",
            conn,
            if_exists="replace",
            index=False,
        )

    conn.commit()


def print_summary(
    summary: pd.DataFrame,
    buckets: pd.DataFrame,
) -> None:
    print()
    print("=" * 72)
    print("BACKTEST WALK-FORWARD")
    print("=" * 72)

    if summary.empty:
        print("No hay resultados.")
        return

    for market in MARKETS:
        g = summary[
            summary["market"] == market
        ]

        if g.empty:
            continue

        print()
        print(MARKETS[market])
        print("-" * 72)

        for r in g.itertuples(index=False):
            auc_text = (
                f"{r.auc:.3f}"
                if pd.notna(r.auc)
                else "N/A"
            )

            print(
                f"Test {int(r.test_season)} | "
                f"N={int(r.test_rows):3d} | "
                f"Hit base={r.base_rate*100:5.1f}% | "
                f"Acc={r.accuracy*100:5.1f}% | "
                f"AUC={auc_text} | "
                f"Brier={r.brier:.3f}"
            )

        # Resultado global OOS del mercado.
        # Lo sacaremos desde buckets/predictions en el main.

    print()
    print("=" * 72)
    print("NOTA")
    print("=" * 72)
    print(
        "Esto mide predicción fuera de muestra. "
        "Todavía NO es un backtest de rentabilidad."
    )
    print(
        "La rentabilidad llegará cuando crucemos "
        "probabilidad del modelo con cuotas y EV."
    )
    print("=" * 72)


def main():
    print()
    print("=" * 72)
    print("ELITESERIEN EDGE PRO - MODELOS GOLES / BTTS")
    print("=" * 72)
    print("Modelo inicial: Regresión logística ponderada por recencia")
    print("Backtest: walk-forward por temporada")
    print()

    if not DB_PATH.exists():
        print(f"ERROR: no existe {DB_PATH}")
        raise SystemExit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        print("1/5 Cargando features pre-partido...")
        team_df = load_team_features(conn)

        print("2/5 Construyendo dataset por partido...")
        df, feature_columns = build_match_dataset(team_df)

        print(
            f"    Partidos: {len(df)} | "
            f"Features: {len(feature_columns)}"
        )

        print("3/5 Ejecutando backtesting cronológico...")

        all_predictions = []
        all_summaries = []

        for market in MARKETS:
            pred, summary = backtest_market(
                df,
                feature_columns,
                market,
            )

            if not pred.empty:
                all_predictions.append(pred)

            if not summary.empty:
                all_summaries.append(summary)

        predictions = (
            pd.concat(
                all_predictions,
                ignore_index=True,
            )
            if all_predictions
            else pd.DataFrame()
        )

        summary = (
            pd.concat(
                all_summaries,
                ignore_index=True,
            )
            if all_summaries
            else pd.DataFrame()
        )

        buckets = probability_buckets(
            predictions
        )

        print("4/5 Guardando backtesting...")
        save_results(
            conn,
            predictions,
            summary,
            buckets,
        )

        print("5/5 Entrenando modelos finales...")
        metadata = train_final_models(
            df,
            feature_columns,
        )

    # CSV fáciles de revisar.
    if not predictions.empty:
        predictions.to_csv(
            REPORTS_DIR
            / "goals_btts_backtest_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not summary.empty:
        summary.to_csv(
            REPORTS_DIR
            / "goals_btts_backtest_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not buckets.empty:
        buckets.to_csv(
            REPORTS_DIR
            / "goals_btts_probability_buckets.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print_summary(
        summary,
        buckets,
    )

    print()
    print("Modelos guardados:")
    for market, info in metadata["markets"].items():
        print(
            f"  {market:<10} -> "
            f"{info['model_path']}"
        )

    print()
    print("Tablas SQLite:")
    print("  model_backtest_predictions")
    print("  model_backtest_summary")
    print("  model_probability_buckets")

    print()
    print("Informes CSV:")
    print("  reports\\goals_btts_backtest_predictions.csv")
    print("  reports\\goals_btts_backtest_summary.csv")
    print("  reports\\goals_btts_probability_buckets.csv")

    print()
    print("=" * 72)
    print("PRIMER BLOQUE DE MODELOS COMPLETADO")
    print("=" * 72)


if __name__ == "__main__":
    main()
