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
HALF_LIFE_DAYS = 365.0
POISSON_ALPHA = 1.8

CARD_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)


def safe_mean(values):
    vals = [
        float(v)
        for v in values
        if v is not None and not pd.isna(v)
    ]
    return float(np.mean(vals)) if vals else np.nan


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
        df["utc_time"], utc=True, errors="coerce"
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
        ref["utc_time"], utc=True, errors="coerce"
    )

    return ref.dropna(subset=["utc_time"]).copy()


def pair_matches(team_df):
    home = team_df[team_df["venue"] == "home"].copy()
    away = team_df[team_df["venue"] == "away"].copy()

    base = [
        "match_id", "season", "round", "utc_time",
        "team", "opponent",
        "yellow_for", "yellow_against",
        "fouls_for", "fouls_against",
        "red_for", "red_against",
        "referee",
    ]

    metrics = (
        "yellow_for", "yellow_against",
        "fouls_for", "fouls_against",
        "red_for", "red_against",
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
        df["home_yellow_for"], errors="coerce"
    )
    df["away_yellow"] = pd.to_numeric(
        df["away_yellow_for"], errors="coerce"
    )

    df = df.dropna(
        subset=["home_yellow", "away_yellow"]
    ).copy()

    df["total_yellow"] = df["home_yellow"] + df["away_yellow"]
    df["match_day"] = df["utc_time"].dt.floor("D")

    if "home_referee" in df.columns:
        df["referee"] = df["home_referee"]
    else:
        df["referee"] = None

    # Tendencias de liga PRE-PARTIDO, sin leakage.
    df = df.sort_values(["utc_time", "match_id"]).reset_index(drop=True)

    df["league_last50_cards"] = (
        df["total_yellow"]
        .shift(1)
        .rolling(50, min_periods=20)
        .mean()
    )

    df["league_last50_home_cards"] = (
        df["home_yellow"]
        .shift(1)
        .rolling(50, min_periods=20)
        .mean()
    )

    df["league_last50_away_cards"] = (
        df["away_yellow"]
        .shift(1)
        .rolling(50, min_periods=20)
        .mean()
    )

    df["league_season_cards"] = np.nan

    for season, idx in df.groupby("season").groups.items():
        s = df.loc[idx].sort_values(["utc_time", "match_id"])
        vals = (
            s["total_yellow"]
            .shift(1)
            .expanding(min_periods=10)
            .mean()
        )
        df.loc[s.index, "league_season_cards"] = vals.values

    return df


def referee_features(ref_hist, referee, cutoff):
    prior_all = ref_hist[
        ref_hist["utc_time"] < cutoff
    ].copy()

    league_avg_total = pd.to_numeric(
        prior_all["total_yellow"], errors="coerce"
    ).mean()

    league_avg_home = pd.to_numeric(
        prior_all["home_yellow"], errors="coerce"
    ).mean()

    league_avg_away = pd.to_numeric(
        prior_all["away_yellow"], errors="coerce"
    ).mean()

    h = prior_all[
        prior_all["referee"] == referee
    ].copy()

    if h.empty:
        return {
            "ref_matches": 0,
            "ref_total_yellow": league_avg_total,
            "ref_home_yellow": league_avg_home,
            "ref_away_yellow": league_avg_away,
            "ref_last10_total": league_avg_total,
            "ref_last20_total": league_avg_total,
            "ref_fouls": pd.to_numeric(
                prior_all["total_fouls"], errors="coerce"
            ).mean(),
        }

    h = h.sort_values("utc_time")

    n = len(h)
    prior_strength = 20.0

    def shrunk(col, league_avg):
        vals = pd.to_numeric(h[col], errors="coerce").dropna()
        if vals.empty or pd.isna(league_avg):
            return league_avg

        return float(
            (
                vals.sum()
                + prior_strength * league_avg
            )
            / (len(vals) + prior_strength)
        )

    total = pd.to_numeric(
        h["total_yellow"], errors="coerce"
    )

    fouls = pd.to_numeric(
        h["total_fouls"], errors="coerce"
    )

    return {
        "ref_matches": n,
        "ref_total_yellow": shrunk(
            "total_yellow", league_avg_total
        ),
        "ref_home_yellow": shrunk(
            "home_yellow", league_avg_home
        ),
        "ref_away_yellow": shrunk(
            "away_yellow", league_avg_away
        ),
        "ref_last10_total": float(
            total.tail(10).mean()
        ) if total.notna().any() else league_avg_total,
        "ref_last20_total": float(
            total.tail(20).mean()
        ) if total.notna().any() else league_avg_total,
        "ref_fouls": float(
            fouls.mean()
        ) if fouls.notna().any() else np.nan,
    }


def build_side_rows(match_df, ref_hist):
    rows = []

    for _, r in match_df.iterrows():
        ref = referee_features(
            ref_hist,
            r["referee"],
            r["utc_time"],
        )

        common = {
            "match_id": int(r["match_id"]),
            "season": int(r["season"]),
            "utc_time": r["utc_time"],
            "match_day": r["match_day"],
            "referee": r["referee"],
            "league_last50_cards": r["league_last50_cards"],
            "league_season_cards": r["league_season_cards"],
            **ref,
        }

        home = {
            **common,
            "side": "home",
            "team": r["home_team"],
            "opponent": r["away_team"],
            "target_yellow": float(r["home_yellow"]),
            "league_side_cards": r["league_last50_home_cards"],
            "ref_side_yellow": ref["ref_home_yellow"],
        }

        away = {
            **common,
            "side": "away",
            "team": r["away_team"],
            "opponent": r["home_team"],
            "target_yellow": float(r["away_yellow"]),
            "league_side_cards": r["league_last50_away_cards"],
            "ref_side_yellow": ref["ref_away_yellow"],
        }

        for w in ("last5", "last10", "season"):
            home_yf = r.get(
                f"home_pre_{w}_yellow_for", np.nan
            )
            home_ya = r.get(
                f"home_pre_{w}_yellow_against", np.nan
            )
            away_yf = r.get(
                f"away_pre_{w}_yellow_for", np.nan
            )
            away_ya = r.get(
                f"away_pre_{w}_yellow_against", np.nan
            )

            home_ff = r.get(
                f"home_pre_{w}_fouls_for", np.nan
            )
            home_fa = r.get(
                f"home_pre_{w}_fouls_against", np.nan
            )
            away_ff = r.get(
                f"away_pre_{w}_fouls_for", np.nan
            )
            away_fa = r.get(
                f"away_pre_{w}_fouls_against", np.nan
            )

            home[f"team_yellow_for_{w}"] = home_yf
            home[f"opp_yellow_against_{w}"] = away_ya
            home[f"team_fouls_for_{w}"] = home_ff
            home[f"opp_fouls_against_{w}"] = away_fa
            home[f"expected_cards_{w}"] = safe_mean(
                [home_yf, away_ya, ref["ref_home_yellow"]]
            )

            away[f"team_yellow_for_{w}"] = away_yf
            away[f"opp_yellow_against_{w}"] = home_ya
            away[f"team_fouls_for_{w}"] = away_ff
            away[f"opp_fouls_against_{w}"] = home_fa
            away[f"expected_cards_{w}"] = safe_mean(
                [away_yf, home_ya, ref["ref_away_yellow"]]
            )

        rows.extend([home, away])

    df = pd.DataFrame(rows)

    categorical = [
        "side",
        "team",
        "opponent",
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
                        (
                            "impute",
                            SimpleImputer(strategy="median"),
                        ),
                        (
                            "scale",
                            StandardScaler(),
                        ),
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
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
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

    w = np.power(
        0.5,
        days.to_numpy(dtype=float)
        / HALF_LIFE_DAYS,
    )

    return np.clip(w, 0.15, 1.0)


def estimate_nb_alpha(total_cards):
    y = np.asarray(total_cards, dtype=float)

    if len(y) < 30:
        return 0.10

    mu = float(np.mean(y))
    var = float(np.var(y, ddof=1))

    if mu <= 0:
        return 0.10

    alpha = (var - mu) / (mu * mu)

    return float(
        np.clip(alpha, 0.01, 1.50)
    )


def nb_cdf(k, mu, alpha):
    mu = max(float(mu), 1e-6)
    alpha = max(float(alpha), 1e-6)

    r = 1.0 / alpha
    p = r / (r + mu)

    total = 0.0

    for x in range(0, k + 1):
        log_pmf = (
            math.lgamma(x + r)
            - math.lgamma(r)
            - math.lgamma(x + 1)
            + r * math.log(p)
            + x * math.log(1 - p)
        )

        total += math.exp(log_pmf)

    return float(np.clip(total, 0, 1))


def over_prob(line, mu, alpha):
    return 1.0 - nb_cdf(
        int(math.floor(line)),
        mu,
        alpha,
    )


def walkforward(match_df, side_df, numeric, categorical):
    feature_cols = numeric + categorical
    rows = []

    days = sorted(
        match_df.loc[
            match_df["season"] >= TEST_START_SEASON,
            "match_day"
        ].unique()
    )

    print(f"    Días OOS: {len(days)}")

    for i, day_raw in enumerate(days, start=1):
        day = pd.Timestamp(day_raw)

        train_side = side_df[
            side_df["match_day"] < day
        ].copy()

        test_side = side_df[
            side_df["match_day"] == day
        ].copy()

        test_matches = match_df[
            match_df["match_day"] == day
        ].copy()

        if (
            len(train_side) < MIN_TRAIN_MATCHES * 2
            or test_side.empty
            or test_matches.empty
        ):
            continue

        model = make_model(
            numeric,
            categorical,
        )

        weights = recency_weights(
            train_side,
            day,
        )

        model.fit(
            train_side[feature_cols],
            train_side["target_yellow"],
            model__sample_weight=weights,
        )

        test_side = test_side.copy()

        test_side["pred_yellow"] = np.clip(
            model.predict(
                test_side[feature_cols]
            ),
            0.05,
            5.0,
        )

        train_matches = match_df[
            match_df["match_day"] < day
        ].copy()

        match_weights = recency_weights(
            pd.DataFrame({
                "match_day": train_matches["match_day"]
            }),
            day,
        )

        baseline_mu = float(
            np.average(
                train_matches["total_yellow"],
                weights=match_weights,
            )
        )

        nb_alpha = estimate_nb_alpha(
            train_matches["total_yellow"]
        )

        for _, m in test_matches.iterrows():
            sides = test_side[
                test_side["match_id"] == m["match_id"]
            ]

            home = sides[
                sides["side"] == "home"
            ]

            away = sides[
                sides["side"] == "away"
            ]

            if home.empty or away.empty:
                continue

            mu_home = float(
                home.iloc[0]["pred_yellow"]
            )

            mu_away = float(
                away.iloc[0]["pred_yellow"]
            )

            mu_total = mu_home + mu_away

            for line in CARD_LINES:
                market = (
                    f"over_{str(line).replace('.', '_')}"
                )

                rows.append(
                    {
                        "match_id": int(m["match_id"]),
                        "season": int(m["season"]),
                        "match_day": str(day.date()),
                        "home_team": m["home_team"],
                        "away_team": m["away_team"],
                        "referee": m["referee"],
                        "total_yellow": float(
                            m["total_yellow"]
                        ),
                        "mu_home": mu_home,
                        "mu_away": mu_away,
                        "mu_total": mu_total,
                        "nb_alpha": nb_alpha,
                        "market": market,
                        "line": line,
                        "raw_probability": over_prob(
                            line,
                            mu_total,
                            nb_alpha,
                        ),
                        "baseline_probability": over_prob(
                            line,
                            baseline_mu,
                            nb_alpha,
                        ),
                        "actual": int(
                            m["total_yellow"] > line
                        ),
                    }
                )

        if i % 20 == 0 or i == len(days):
            print(
                f"    [{i}/{len(days)}] "
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

    for w in np.arange(
        0.0, 1.0001, 0.05
    ):
        p = (
            w * raw
            + (1 - w) * base
        )

        b = brier_score_loss(y, p)

        if b < best_brier:
            best_brier = b
            best_w = float(w)

    return best_w


def apply_shrink(pred):
    pred = pred.copy()

    pred["shrink_weight"] = np.nan
    pred["final_probability"] = np.nan

    for season in sorted(
        pred["season"].unique()
    ):
        for market in sorted(
            pred["market"].unique()
        ):
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
            ] = (
                w * raw
                + (1 - w) * base
            )

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
            base = g[
                "baseline_probability"
            ].to_numpy(dtype=float)

            b = float(
                brier_score_loss(y, p)
            )

            bb = float(
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
                    "brier": b,
                    "brier_baseline": bb,
                    "improvement": bb - b,
                    "avg_weight": float(
                        g["shrink_weight"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def train_final(side_df, numeric, categorical):
    feature_cols = numeric + categorical

    cutoff = (
        side_df["match_day"].max()
        + pd.Timedelta(days=1)
    )

    weights = recency_weights(
        side_df,
        cutoff,
    )

    model = make_model(
        numeric,
        categorical,
    )

    model.fit(
        side_df[feature_cols],
        side_df["target_yellow"],
        model__sample_weight=weights,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODELS_DIR
        / "laliga_cards_v2_side_poisson.joblib",
    )


def main():
    print()
    print("=" * 76)
    print("ELITESERIEN EDGE PRO - CARDS MODEL V2")
    print("=" * 76)
    print(
        "Local/visitante separados + árbitro suavizado "
        "+ tendencia liga + Negative Binomial."
    )
    print()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DB_PATH) as conn:
        print("1/6 Cargando datos...")
        team_df = load_team_features(conn)
        ref_hist = load_referee_history(conn)

        print("2/6 Construyendo partidos...")
        match_df = pair_matches(team_df)

        print("3/6 Creando filas local/visitante...")
        side_df, numeric, categorical = build_side_rows(
            match_df,
            ref_hist,
        )

        print(
            f"    Partidos: {len(match_df)} | "
            f"Filas: {len(side_df)} | "
            f"Numéricas: {len(numeric)}"
        )

        print("4/6 Walk-forward...")
        pred = walkforward(
            match_df,
            side_df,
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
            "cards_v2_backtest_predictions",
            conn,
            if_exists="replace",
            index=False,
        )

        summary.to_sql(
            "cards_v2_backtest_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.commit()

        pred.to_csv(
            REPORTS_DIR
            / "laliga_cards_v2_backtest_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary.to_csv(
            REPORTS_DIR
            / "laliga_cards_v2_backtest_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("6/6 Entrenando modelo final...")
        train_final(
            side_df,
            numeric,
            categorical,
        )

    print()
    print("=" * 76)
    print("BACKTEST CARDS V2")
    print("=" * 76)

    for line in CARD_LINES:
        market = (
            f"over_{str(line).replace('.', '_')}"
        )

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
    print("CRITERIO")
    print("=" * 76)
    print(
        "Si V2 no mejora claramente V1, dejamos tarjetas "
        "como mercado secundario y seguimos con el resto de la app."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
