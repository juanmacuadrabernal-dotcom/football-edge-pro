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
POISSON_ALPHA = 1.4

CARD_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)


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


def load_referee_history(conn):
    ref = pd.read_sql_query(
        """
        SELECT *
        FROM referee_match_history
        ORDER BY utc_time, match_id
        """,
        conn,
    )

    if ref.empty:
        raise RuntimeError("referee_match_history está vacía.")

    ref["utc_time"] = pd.to_datetime(
        ref["utc_time"],
        utc=True,
        errors="coerce",
    )

    return ref.dropna(subset=["utc_time"]).copy()


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
        "yellow_for",
        "yellow_against",
        "fouls_for",
        "fouls_against",
        "referee",
    ]

    metrics = (
        "yellow_for",
        "yellow_against",
        "fouls_for",
        "fouls_against",
        "red_for",
        "red_against",
        "points",
    )

    rolling = []

    for prefix in ("pre_last5_", "pre_last10_", "pre_season_"):
        for metric in metrics:
            col = prefix + metric
            if col in team_df.columns:
                rolling.append(col)

    home_specific = [
        c for c in (
            "pre_home_last5_yellow_for",
            "pre_home_last5_yellow_against",
            "pre_home_last5_points",
        )
        if c in team_df.columns
    ]

    away_specific = [
        c for c in (
            "pre_away_last5_yellow_for",
            "pre_away_last5_yellow_against",
            "pre_away_last5_points",
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

    df["home_yellow"] = pd.to_numeric(
        df["home_yellow_for"],
        errors="coerce",
    )

    df["away_yellow"] = pd.to_numeric(
        df["away_yellow_for"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["home_yellow", "away_yellow"]
    ).copy()

    df["match_day"] = df["utc_time"].dt.floor("D")

    # árbitro: usamos el del lado local
    if "home_referee" in df.columns:
        df["referee"] = df["home_referee"]
    else:
        df["referee"] = None

    return df.sort_values(
        ["utc_time", "match_id"]
    ).reset_index(drop=True)


def referee_features(ref_hist, referee, cutoff):
    h = ref_hist[
        (ref_hist["referee"] == referee)
        & (ref_hist["utc_time"] < cutoff)
    ].copy()

    if h.empty:
        return {
            "ref_matches": 0,
            "ref_avg_yellow": np.nan,
            "ref_last10_yellow": np.nan,
            "ref_last20_yellow": np.nan,
            "ref_avg_fouls": np.nan,
            "ref_last10_fouls": np.nan,
            "ref_last20_fouls": np.nan,
            "ref_home_yellow": np.nan,
            "ref_away_yellow": np.nan,
        }

    h = h.sort_values("utc_time")

    def avg(col, n=None):
        s = pd.to_numeric(h[col], errors="coerce")
        if n:
            s = s.tail(n)
        val = s.mean()
        return np.nan if pd.isna(val) else float(val)

    return {
        "ref_matches": len(h),
        "ref_avg_yellow": avg("total_yellow"),
        "ref_last10_yellow": avg("total_yellow", 10),
        "ref_last20_yellow": avg("total_yellow", 20),
        "ref_avg_fouls": avg("total_fouls"),
        "ref_last10_fouls": avg("total_fouls", 10),
        "ref_last20_fouls": avg("total_fouls", 20),
        "ref_home_yellow": avg("home_yellow"),
        "ref_away_yellow": avg("away_yellow"),
    }


def build_model_rows(match_df, ref_hist):
    rows = []

    for _, r in match_df.iterrows():
        row = {
            "match_id": int(r["match_id"]),
            "season": int(r["season"]),
            "utc_time": r["utc_time"],
            "match_day": r["match_day"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "referee": r["referee"],
            "target_yellow": float(
                r["home_yellow"] + r["away_yellow"]
            ),
        }

        for prefix in ("last5", "last10", "season"):
            for metric in (
                "yellow_for",
                "yellow_against",
                "fouls_for",
                "fouls_against",
                "red_for",
                "red_against",
                "points",
            ):
                hcol = f"home_pre_{prefix}_{metric}"
                acol = f"away_pre_{prefix}_{metric}"

                if hcol in r.index:
                    row[f"home_{prefix}_{metric}"] = r[hcol]
                if acol in r.index:
                    row[f"away_{prefix}_{metric}"] = r[acol]

            # interacción de tarjetas esperadas
            h_yf = r.get(f"home_pre_{prefix}_yellow_for", np.nan)
            h_ya = r.get(f"home_pre_{prefix}_yellow_against", np.nan)
            a_yf = r.get(f"away_pre_{prefix}_yellow_for", np.nan)
            a_ya = r.get(f"away_pre_{prefix}_yellow_against", np.nan)

            row[f"expected_cards_{prefix}"] = np.nanmean([
                h_yf, h_ya, a_yf, a_ya
            ])

        # venue
        for col in (
            "pre_home_last5_yellow_for",
            "pre_home_last5_yellow_against",
        ):
            source = f"home_{col}"
            if source in r.index:
                row[f"home_venue_{col}"] = r[source]

        for col in (
            "pre_away_last5_yellow_for",
            "pre_away_last5_yellow_against",
        ):
            source = f"away_{col}"
            if source in r.index:
                row[f"away_venue_{col}"] = r[source]

        row.update(
            referee_features(
                ref_hist,
                r["referee"],
                r["utc_time"],
            )
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    categorical = [
        "home_team",
        "away_team",
        "referee",
    ]

    exclude = {
        "match_id",
        "season",
        "utc_time",
        "match_day",
        "target_yellow",
        *categorical,
    }

    numeric = [
        c for c in df.columns
        if c not in exclude
    ]

    numeric = [
        c for c in numeric
        if pd.to_numeric(
            df[c], errors="coerce"
        ).notna().mean() >= 0.35
    ]

    return df, numeric, categorical


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


def over_prob(line, lam):
    return 1.0 - poisson_cdf(
        int(math.floor(line)),
        lam,
    )


def walkforward(model_df, numeric, categorical):
    rows = []
    feature_cols = numeric + categorical

    days = sorted(
        model_df.loc[
            model_df["season"] >= TEST_START_SEASON,
            "match_day"
        ].unique()
    )

    print(f"    Días OOS: {len(days)}")

    for i, day_raw in enumerate(days, start=1):
        day = pd.Timestamp(day_raw)

        train = model_df[
            model_df["match_day"] < day
        ].copy()

        test = model_df[
            model_df["match_day"] == day
        ].copy()

        if len(train) < MIN_TRAIN_MATCHES or test.empty:
            continue

        model = make_model(
            numeric,
            categorical,
        )

        weights = recency_weights(
            train,
            day,
        )

        model.fit(
            train[feature_cols],
            train["target_yellow"],
            model__sample_weight=weights,
        )

        lam = np.clip(
            model.predict(test[feature_cols]),
            0.20,
            10.0,
        )

        baseline_lambda = float(
            np.average(
                train["target_yellow"],
                weights=weights,
            )
        )

        for j, (_, r) in enumerate(test.iterrows()):
            for line in CARD_LINES:
                market = f"over_{str(line).replace('.', '_')}"

                rows.append(
                    {
                        "match_id": int(r["match_id"]),
                        "season": int(r["season"]),
                        "match_day": str(day.date()),
                        "home_team": r["home_team"],
                        "away_team": r["away_team"],
                        "referee": r["referee"],
                        "total_yellow": float(r["target_yellow"]),
                        "lambda_cards": float(lam[j]),
                        "market": market,
                        "line": line,
                        "raw_probability": over_prob(
                            line,
                            lam[j],
                        ),
                        "baseline_probability": over_prob(
                            line,
                            baseline_lambda,
                        ),
                        "actual": int(
                            r["target_yellow"] > line
                        ),
                    }
                )

        if i % 20 == 0 or i == len(days):
            print(
                f"    [{i}/{len(days)}] "
                f"{day.date()} | train={len(train)}"
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


def apply_shrink(pred):
    pred = pred.copy()
    pred["shrink_weight"] = np.nan
    pred["final_probability"] = np.nan

    for season in sorted(pred["season"].unique()):
        for market in sorted(pred["market"].unique()):
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
                mask, "raw_probability"
            ].to_numpy(dtype=float)

            base = pred.loc[
                mask, "baseline_probability"
            ].to_numpy(dtype=float)

            pred.loc[
                mask, "shrink_weight"
            ] = w

            pred.loc[
                mask, "final_probability"
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
                    "improvement": brier_base - brier,
                    "avg_weight": float(
                        g["shrink_weight"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def train_final(model_df, numeric, categorical):
    feature_cols = numeric + categorical

    cutoff = (
        model_df["match_day"].max()
        + pd.Timedelta(days=1)
    )

    weights = recency_weights(
        model_df,
        cutoff,
    )

    model = make_model(
        numeric,
        categorical,
    )

    model.fit(
        model_df[feature_cols],
        model_df["target_yellow"],
        model__sample_weight=weights,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODELS_DIR / "cards_v1_yellow_poisson.joblib",
    )


def main():
    print()
    print("=" * 76)
    print("ELITESERIEN EDGE PRO - CARDS MODEL V1")
    print("=" * 76)
    print("Equipos + faltas + forma + árbitro histórico 2020-2026.")
    print()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DB_PATH) as conn:
        print("1/6 Cargando equipo + árbitros...")
        team_df = load_team_features(conn)
        ref_hist = load_referee_history(conn)

        print("2/6 Construyendo partidos...")
        match_df = pair_matches(team_df)

        print("3/6 Creando features de tarjetas...")
        model_df, numeric, categorical = build_model_rows(
            match_df,
            ref_hist,
        )

        print(
            f"    Partidos: {len(model_df)} | "
            f"Numéricas: {len(numeric)} | "
            f"Árbitros históricos: {ref_hist['referee'].nunique()}"
        )

        print("4/6 Walk-forward...")
        pred = walkforward(
            model_df,
            numeric,
            categorical,
        )

        if pred.empty:
            raise RuntimeError(
                "No se generaron predicciones."
            )

        print("5/6 Shrink + evaluación...")
        pred = apply_shrink(pred)
        summary = summarize(pred)

        pred.to_sql(
            "cards_v1_backtest_predictions",
            conn,
            if_exists="replace",
            index=False,
        )

        summary.to_sql(
            "cards_v1_backtest_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.commit()

        pred.to_csv(
            REPORTS_DIR / "cards_v1_backtest_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary.to_csv(
            REPORTS_DIR / "cards_v1_backtest_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("6/6 Entrenando modelo final...")
        train_final(
            model_df,
            numeric,
            categorical,
        )

    print()
    print("=" * 76)
    print("BACKTEST CARDS V1")
    print("=" * 76)

    for line in CARD_LINES:
        market = f"over_{str(line).replace('.', '_')}"

        print()
        print(f"Total amarillas Over {line}")
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
    print("SIGUIENTE")
    print("=" * 76)
    print(
        "Nos quedaremos solo con líneas que mejoren baseline "
        "en varias temporadas."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
