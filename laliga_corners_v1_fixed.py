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
DB_PATH = BASE_DIR / "data" / "laliga.db"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TEST_START_SEASON = 2024
MIN_TRAIN_MATCHES = 180
HALF_LIFE_DAYS = 420.0
POISSON_ALPHA = 1.5

TOTAL_LINES = (7.5, 8.5, 9.5, 10.5, 11.5)

ATTACK_METRICS = (
    "corners_for",
    "shots_for",
    "shots_on_target_for",
    "xg_for",
    "goals_for",
)

DEFENCE_METRICS = (
    "corners_against",
    "shots_against",
    "shots_on_target_against",
    "xg_against",
    "goals_against",
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

    base = [
        "match_id",
        "season",
        "round",
        "utc_time",
        "team",
        "opponent",
        "corners_for",
        "corners_against",
    ]

    rolling = []

    for prefix in ("pre_last5_", "pre_last10_", "pre_season_"):
        for metric in set(ATTACK_METRICS + DEFENCE_METRICS):
            col = prefix + metric
            if col in team_df.columns:
                rolling.append(col)

    home_specific = [
        c for c in (
            "pre_home_last5_corners_for",
            "pre_home_last5_corners_against",
            "pre_home_last5_shots_for",
            "pre_home_last5_shots_against",
            "pre_home_last5_shots_on_target_for",
            "pre_home_last5_shots_on_target_against",
            "pre_home_last5_xg_for",
            "pre_home_last5_xg_against",
        )
        if c in team_df.columns
    ]

    away_specific = [
        c for c in (
            "pre_away_last5_corners_for",
            "pre_away_last5_corners_against",
            "pre_away_last5_shots_for",
            "pre_away_last5_shots_against",
            "pre_away_last5_shots_on_target_for",
            "pre_away_last5_shots_on_target_against",
            "pre_away_last5_xg_for",
            "pre_away_last5_xg_against",
        )
        if c in team_df.columns
    ]

    hcols = list(dict.fromkeys(base + rolling + home_specific))
    acols = list(dict.fromkeys(base + rolling + away_specific))

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

    df["home_corners"] = pd.to_numeric(
        df["home_corners_for"],
        errors="coerce",
    )

    df["away_corners"] = pd.to_numeric(
        df["away_corners_for"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["home_corners", "away_corners"]
    ).copy()

    df["match_day"] = df["utc_time"].dt.floor("D")

    return df.sort_values(
        ["utc_time", "match_id"]
    ).reset_index(drop=True)


def build_long_rows(match_df):
    rows = []

    def getv(row, col):
        return row[col] if col in row.index else np.nan

    def safe_mean(values):
        clean = pd.to_numeric(
            pd.Series(list(values)),
            errors="coerce",
        ).dropna()

        if clean.empty:
            return np.nan

        return float(clean.mean())

    for _, r in match_df.iterrows():
        home_row = {
            "match_id": int(r["match_id"]),
            "season": int(r["season"]),
            "utc_time": r["utc_time"],
            "match_day": r["match_day"],
            "attack_team": r["home_team"],
            "defence_team": r["away_team"],
            "is_home": 1,
            "target_corners": float(r["home_corners"]),
        }

        away_row = {
            "match_id": int(r["match_id"]),
            "season": int(r["season"]),
            "utc_time": r["utc_time"],
            "match_day": r["match_day"],
            "attack_team": r["away_team"],
            "defence_team": r["home_team"],
            "is_home": 0,
            "target_corners": float(r["away_corners"]),
        }

        for w in ("last5", "last10", "season"):
            for metric in ATTACK_METRICS:
                home_row[f"att_{w}_{metric}"] = getv(
                    r, f"home_pre_{w}_{metric}"
                )
                away_row[f"att_{w}_{metric}"] = getv(
                    r, f"away_pre_{w}_{metric}"
                )

            for metric in DEFENCE_METRICS:
                home_row[f"oppdef_{w}_{metric}"] = getv(
                    r, f"away_pre_{w}_{metric}"
                )
                away_row[f"oppdef_{w}_{metric}"] = getv(
                    r, f"home_pre_{w}_{metric}"
                )

            home_row[f"expected_corners_{w}"] = safe_mean([
                home_row.get(f"att_{w}_corners_for", np.nan),
                home_row.get(f"oppdef_{w}_corners_against", np.nan),
            ])

            away_row[f"expected_corners_{w}"] = safe_mean([
                away_row.get(f"att_{w}_corners_for", np.nan),
                away_row.get(f"oppdef_{w}_corners_against", np.nan),
            ])

        for metric in (
            "corners_for",
            "corners_against",
            "shots_for",
            "shots_against",
            "shots_on_target_for",
            "shots_on_target_against",
            "xg_for",
            "xg_against",
        ):
            home_row[f"venue_{metric}"] = getv(
                r, f"home_pre_home_last5_{metric}"
            )
            away_row[f"venue_{metric}"] = getv(
                r, f"away_pre_away_last5_{metric}"
            )

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
        "target_corners",
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


def recency_weights(frame, cutoff):
    days = (
        cutoff - frame["match_day"]
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


def over_prob(line, total_lambda):
    threshold = int(math.floor(line))
    return 1.0 - poisson_cdf(threshold, total_lambda)


def walkforward(match_df, long_df, numeric, categorical):
    feature_cols = numeric + categorical
    rows = []

    test_days = sorted(
        match_df.loc[
            match_df["season"] >= TEST_START_SEASON,
            "match_day"
        ].unique()
    )

    print(f"    Días OOS: {len(test_days)}")

    for i, day_raw in enumerate(test_days, start=1):
        day = pd.Timestamp(day_raw)

        train_long = long_df[
            long_df["match_day"] < day
        ].copy()

        test_matches = match_df[
            match_df["match_day"] == day
        ].copy()

        if (
            len(train_long) < MIN_TRAIN_MATCHES * 2
            or test_matches.empty
        ):
            continue

        model = make_model(
            numeric,
            categorical,
        )

        weights = recency_weights(
            train_long,
            day,
        )

        model.fit(
            train_long[feature_cols],
            train_long["target_corners"],
            model__sample_weight=weights,
        )

        train_matches = match_df[
            match_df["match_day"] < day
        ].copy()

        match_weight_frame = pd.DataFrame({
            "match_day": train_matches["match_day"]
        })

        match_weights = recency_weights(
            match_weight_frame,
            day,
        )

        baseline_total_lambda = float(
            np.average(
                train_matches["home_corners"]
                + train_matches["away_corners"],
                weights=match_weights,
            )
        )

        test_long = long_df[
            long_df["match_day"] == day
        ].copy()

        test_long["pred_corners"] = np.clip(
            model.predict(test_long[feature_cols]),
            0.20,
            12.0,
        )

        for _, match in test_matches.iterrows():
            sides = test_long[
                test_long["match_id"] == match["match_id"]
            ]

            home = sides[
                sides["is_home"] == 1
            ]
            away = sides[
                sides["is_home"] == 0
            ]

            if home.empty or away.empty:
                continue

            lh = float(home.iloc[0]["pred_corners"])
            la = float(away.iloc[0]["pred_corners"])

            total_lambda = lh + la
            actual_total = (
                float(match["home_corners"])
                + float(match["away_corners"])
            )

            for line in TOTAL_LINES:
                market = f"over_{str(line).replace('.', '_')}"

                rows.append(
                    {
                        "match_id": int(match["match_id"]),
                        "season": int(match["season"]),
                        "match_day": str(day.date()),
                        "home_team": match["home_team"],
                        "away_team": match["away_team"],
                        "home_corners": float(match["home_corners"]),
                        "away_corners": float(match["away_corners"]),
                        "total_corners": actual_total,
                        "lambda_home": lh,
                        "lambda_away": la,
                        "lambda_total": total_lambda,
                        "market": market,
                        "line": line,
                        "raw_probability": over_prob(
                            line,
                            total_lambda,
                        ),
                        "baseline_probability": over_prob(
                            line,
                            baseline_total_lambda,
                        ),
                        "actual": int(
                            actual_total > line
                        ),
                    }
                )

        if i % 20 == 0 or i == len(test_days):
            print(
                f"    [{i}/{len(test_days)}] "
                f"{day.date()} | "
                f"train={len(train_matches)}"
            )

    return pd.DataFrame(rows)


def choose_weight(history, market):
    h = history[
        history["market"] == market
    ]

    if len(h) < 150:
        return 0.50

    y = h["actual"].to_numpy(dtype=int)
    raw = h["raw_probability"].to_numpy(dtype=float)
    base = h["baseline_probability"].to_numpy(dtype=float)

    best_w = 0.0
    best_brier = float("inf")

    for w in np.arange(0, 1.0001, 0.05):
        p = w * raw + (1 - w) * base
        b = brier_score_loss(y, p)

        if b < best_brier:
            best_brier = b
            best_w = float(w)

    return best_w


def shrink_probs(pred):
    pred = pred.copy()
    pred["shrink_weight"] = np.nan
    pred["final_probability"] = np.nan

    seasons = sorted(
        int(x) for x in pred["season"].unique()
    )

    markets = sorted(
        pred["market"].unique()
    )

    for season in seasons:
        for market in markets:
            mask = (
                (pred["season"] == season)
                & (pred["market"] == market)
            )

            prior = pred[
                pred["season"] < season
            ]

            w = choose_weight(
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

            pred.loc[
                mask,
                "shrink_weight"
            ] = w

            pred.loc[
                mask,
                "final_probability"
            ] = w * raw + (1 - w) * base

    return pred


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan

    try:
        return float(
            roc_auc_score(y, p)
        )
    except Exception:
        return np.nan


def summarize(pred):
    rows = []

    for market, m in pred.groupby("market"):
        line = float(m["line"].iloc[0])

        groups = [("GLOBAL", m)]

        for season, g in m.groupby("season"):
            groups.append(
                (str(int(season)), g)
            )

        for period, g in groups:
            y = g["actual"].to_numpy(dtype=int)
            p = g["final_probability"].to_numpy(dtype=float)
            base = g["baseline_probability"].to_numpy(dtype=float)

            brier = float(
                brier_score_loss(y, p)
            )
            brier_base = float(
                brier_score_loss(y, base)
            )

            rows.append(
                {
                    "market": market,
                    "line": line,
                    "period": period,
                    "matches": len(g),
                    "hit_rate": float(np.mean(y)),
                    "auc": safe_auc(y, p),
                    "brier": brier,
                    "brier_baseline": brier_base,
                    "improvement": (
                        brier_base - brier
                    ),
                    "avg_weight": float(
                        g["shrink_weight"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def train_final(long_df, numeric, categorical):
    feature_cols = numeric + categorical

    cutoff = (
        long_df["match_day"].max()
        + pd.Timedelta(days=1)
    )

    weights = recency_weights(
        long_df,
        cutoff,
    )

    model = make_model(
        numeric,
        categorical,
    )

    model.fit(
        long_df[feature_cols],
        long_df["target_corners"],
        model__sample_weight=weights,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODELS_DIR / "laliga_corners_v1_poisson.joblib",
    )


def main():
    print()
    print("=" * 76)
    print("LALIGA EDGE PRO - CORNERS MODEL V1")
    print("=" * 76)
    print("Ataque/defensa + forma + local/visitante + walk-forward.")
    print()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DB_PATH) as conn:
        print("1/6 Cargando features...")
        team_df = load_team_features(conn)

        print("2/6 Construyendo partidos...")
        match_df = pair_matches(team_df)

        print("3/6 Construyendo modelo largo...")
        long_df, numeric, categorical = build_long_rows(
            match_df
        )

        print(
            f"    Partidos: {len(match_df)} | "
            f"Filas: {len(long_df)} | "
            f"Numéricas: {len(numeric)}"
        )

        print("4/6 Walk-forward...")
        pred = walkforward(
            match_df,
            long_df,
            numeric,
            categorical,
        )

        if pred.empty:
            raise RuntimeError(
                "No se generaron predicciones."
            )

        print("5/6 Shrink + evaluación...")
        pred = shrink_probs(pred)
        summary = summarize(pred)

        pred.to_sql(
            "corners_v1_backtest_predictions",
            conn,
            if_exists="replace",
            index=False,
        )

        summary.to_sql(
            "corners_v1_backtest_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.commit()

        pred.to_csv(
            REPORTS_DIR / "laliga_corners_v1_backtest_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary.to_csv(
            REPORTS_DIR / "laliga_corners_v1_backtest_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("6/6 Entrenando modelo final...")
        train_final(
            long_df,
            numeric,
            categorical,
        )

    print()
    print("=" * 76)
    print("BACKTEST CORNERS V1")
    print("=" * 76)

    for line in TOTAL_LINES:
        market = f"over_{str(line).replace('.', '_')}"

        print()
        print(f"Total corners Over {line}")
        print("-" * 76)

        g = summary[
            summary["market"] == market
        ]

        for r in g.itertuples(index=False):
            auc = (
                f"{r.auc:.3f}"
                if pd.notna(r.auc)
                else "N/A"
            )

            verdict = (
                "MEJORA"
                if r.improvement > 0
                else "NO MEJORA"
            )

            print(
                f"{r.period:>6} | "
                f"N={int(r.matches):3d} | "
                f"Hit={r.hit_rate*100:5.1f}% | "
                f"AUC={auc} | "
                f"Brier={r.brier:.3f} | "
                f"Base={r.brier_baseline:.3f} | "
                f"w={r.avg_weight:.2f} | "
                f"{verdict}"
            )

    print()
    print("=" * 76)
    print("SIGUIENTE DECISION")
    print("=" * 76)
    print(
        "Nos quedaremos solo con las líneas que mejoren el baseline "
        "de forma consistente."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
