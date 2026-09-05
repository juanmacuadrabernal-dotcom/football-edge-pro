from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
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

CARD_LINES = (2.5, 3.5, 4.5, 5.5)

V3_MODEL_PATH = MODELS_DIR / "laliga_cards_v3_referee_side_poisson.joblib"
V3_META_PATH = MODELS_DIR / "laliga_cards_v3_referee_meta.joblib"


def norm_text(value):
    if value is None or pd.isna(value):
        return ""

    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def team_key(value):
    s = norm_text(value)

    aliases = {
        "ath madrid": "atletico madrid",
        "atletico de madrid": "atletico madrid",
        "atletico madrid": "atletico madrid",
        "athletic club": "athletic bilbao",
        "ath bilbao": "athletic bilbao",
        "athletic bilbao": "athletic bilbao",
        "real betis": "betis",
        "betis": "betis",
        "real sociedad": "real sociedad",
        "sociedad": "real sociedad",
        "rayo vallecano": "rayo vallecano",
        "vallecano": "rayo vallecano",
        "celta vigo": "celta vigo",
        "celta": "celta vigo",
        "deportivo alaves": "alaves",
        "alaves": "alaves",
        "cd leganes": "leganes",
        "leganes": "leganes",
        "rcd espanyol": "espanyol",
        "espanyol": "espanyol",
        "real valladolid": "valladolid",
        "valladolid": "valladolid",
        "ud las palmas": "las palmas",
        "las palmas": "las palmas",
        "rcd mallorca": "mallorca",
        "mallorca": "mallorca",
        "girona fc": "girona",
        "girona": "girona",
        "fc barcelona": "barcelona",
        "barcelona": "barcelona",
        "real madrid cf": "real madrid",
        "real madrid": "real madrid",
        "sevilla fc": "sevilla",
        "sevilla": "sevilla",
        "valencia cf": "valencia",
        "valencia": "valencia",
        "villarreal cf": "villarreal",
        "villarreal": "villarreal",
        "getafe cf": "getafe",
        "getafe": "getafe",
        "granada cf": "granada",
        "granada": "granada",
        "cadiz cf": "cadiz",
        "cadiz": "cadiz",
        "elche cf": "elche",
        "elche": "elche",
        "osasuna": "osasuna",
        "ca osasuna": "osasuna",
        "levante ud": "levante",
        "levante": "levante",
        "real oviedo": "oviedo",
        "oviedo": "oviedo",
        "real zaragoza": "zaragoza",
        "zaragoza": "zaragoza",
    }

    if s in aliases:
        return aliases[s]

    # Quitar sufijos/prefijos de club que no ayudan.
    tokens = [
        t for t in s.split()
        if t not in {"fc", "cf", "ud", "cd", "rcd"}
    ]
    return " ".join(tokens)


def safe_mean(values):
    vals = [
        float(v)
        for v in values
        if v is not None and not pd.isna(v)
    ]
    return float(np.mean(vals)) if vals else np.nan


def table_exists(conn, name):
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name=?
        LIMIT 1
        """,
        (name,),
    ).fetchone()
    return row is not None


def load_team_features(conn):
    if not table_exists(conn, "team_pre_match_features"):
        raise RuntimeError("Falta la tabla team_pre_match_features.")

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


def load_ref_features(conn):
    if not table_exists(conn, "referee_pre_match_features"):
        raise RuntimeError(
            "Falta referee_pre_match_features. "
            "Ejecuta primero build_laliga_referee_features_v1.py"
        )

    ref = pd.read_sql_query(
        """
        SELECT *
        FROM referee_pre_match_features
        ORDER BY utc_time, match_id
        """,
        conn,
    )

    if ref.empty:
        raise RuntimeError("referee_pre_match_features está vacía.")

    ref["utc_time"] = pd.to_datetime(
        ref["utc_time"], utc=True, errors="coerce"
    )

    return ref.dropna(subset=["utc_time"]).copy()


def existing_cols(df, cols):
    return [c for c in cols if c in df.columns]


def pair_matches(team_df):
    if "venue" not in team_df.columns:
        raise RuntimeError("team_pre_match_features no contiene venue.")

    home = team_df[team_df["venue"].astype(str).str.lower() == "home"].copy()
    away = team_df[team_df["venue"].astype(str).str.lower() == "away"].copy()

    required = [
        "match_id", "season", "utc_time",
        "team", "opponent",
        "yellow_for", "yellow_against",
    ]
    missing = [c for c in required if c not in team_df.columns]
    if missing:
        raise RuntimeError(
            "Faltan columnas en team_pre_match_features: "
            + ", ".join(missing)
        )

    shared_base = [
        "match_id", "season", "round", "utc_time",
        "team", "opponent",
        "yellow_for", "yellow_against",
        "fouls_for", "fouls_against",
        "red_for", "red_against",
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

    home_specific = existing_cols(
        team_df,
        [
            "pre_home_last5_yellow_for",
            "pre_home_last5_yellow_against",
            "pre_home_last5_fouls_for",
            "pre_home_last5_fouls_against",
            "pre_home_last5_points",
        ],
    )

    away_specific = existing_cols(
        team_df,
        [
            "pre_away_last5_yellow_for",
            "pre_away_last5_yellow_against",
            "pre_away_last5_fouls_for",
            "pre_away_last5_fouls_against",
            "pre_away_last5_points",
        ],
    )

    hcols = list(dict.fromkeys(existing_cols(home, shared_base) + rolling + home_specific))
    acols = list(dict.fromkeys(existing_cols(away, shared_base) + rolling + away_specific))

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

    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)

    # Tendencias de liga PRE-PARTIDO.
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

    for _, idx in df.groupby("season").groups.items():
        s = df.loc[idx].sort_values(["utc_time", "match_id"])
        vals = (
            s["total_yellow"]
            .shift(1)
            .expanding(min_periods=10)
            .mean()
        )
        df.loc[s.index, "league_season_cards"] = vals.values

    return df


def attach_referee_features(match_df, ref_df):
    m = match_df.copy()
    r = ref_df.copy()

    # Conservamos únicamente variables válidas PRE-PARTIDO.
    ref_pre_cols = [
        c for c in r.columns
        if (
            c == "match_id"
            or c in {
                "utc_time", "home_team", "away_team", "referee"
            }
            or c.startswith("ref_")
            or c.startswith("league_")
        )
    ]

    # Nunca permitir targets actuales como features.
    forbidden = {
        "total_yellow", "total_red", "total_fouls",
        "home_yellow", "away_yellow",
    }
    ref_pre_cols = [c for c in ref_pre_cols if c not in forbidden]

    r = r[ref_pre_cols].copy()

    # 1) Intento por match_id.
    common_ids = set(m["match_id"]).intersection(set(r["match_id"]))
    id_overlap = len(common_ids)

    merged = m.merge(
        r,
        on="match_id",
        how="left",
        suffixes=("", "_ref"),
    )

    if "referee" not in merged.columns and "referee_ref" in merged.columns:
        merged["referee"] = merged["referee_ref"]

    # Marcar filas sin árbitro/features tras match_id.
    matched = merged["referee"].notna() if "referee" in merged.columns else pd.Series(False, index=merged.index)

    # 2) Fallback robusto por fecha + equipos normalizados.
    if (~matched).any() or id_overlap < len(m) * 0.8:
        fallback = r.copy()

        fallback["date_key"] = fallback["utc_time"].dt.date.astype(str)
        fallback["home_key"] = fallback["home_team"].map(team_key)
        fallback["away_key"] = fallback["away_team"].map(team_key)

        ref_feature_names = [
            c for c in fallback.columns
            if (
                c == "referee"
                or c.startswith("ref_")
                or c.startswith("league_")
            )
        ]

        fallback = fallback[
            ["date_key", "home_key", "away_key"] + ref_feature_names
        ].drop_duplicates(
            subset=["date_key", "home_key", "away_key"],
            keep="last",
        )

        mkeys = m[["match_id", "utc_time", "home_team", "away_team"]].copy()
        mkeys["date_key"] = mkeys["utc_time"].dt.date.astype(str)
        mkeys["home_key"] = mkeys["home_team"].map(team_key)
        mkeys["away_key"] = mkeys["away_team"].map(team_key)

        by_key = mkeys.merge(
            fallback,
            on=["date_key", "home_key", "away_key"],
            how="left",
        )

        by_key = by_key.set_index("match_id")

        merged = merged.set_index("match_id")

        for col in ref_feature_names:
            if col not in merged.columns:
                merged[col] = np.nan

            fill_vals = by_key[col].reindex(merged.index)

            if col == "referee":
                merged[col] = merged[col].where(
                    merged[col].notna(),
                    fill_vals,
                )
            else:
                merged[col] = pd.to_numeric(
                    merged[col], errors="coerce"
                ).where(
                    pd.to_numeric(merged[col], errors="coerce").notna(),
                    pd.to_numeric(fill_vals, errors="coerce"),
                )

        merged = merged.reset_index()

    if "referee" not in merged.columns:
        merged["referee"] = np.nan

    coverage = float(merged["referee"].notna().mean())

    print(
        f"    Match IDs compartidos: {id_overlap}/{len(m)} | "
        f"cobertura árbitro tras fallback: {coverage*100:.1f}%"
    )

    if coverage < 0.90:
        sample = merged.loc[
            merged["referee"].isna(),
            ["match_id", "utc_time", "home_team", "away_team"],
        ].head(15)

        print()
        print("    AVISO: partidos sin emparejar con árbitro:")
        print(sample.to_string(index=False))
        print()

    return merged


def referee_feature_columns(df):
    candidates = [
        "ref_matches_before",
        "ref_yellow_avg_before",
        "ref_yellow_last5_before",
        "ref_yellow_last10_before",
        "ref_yellow_last20_before",
        "ref_yellow_shrunk_before",
        "ref_yellow_delta_before",
        "ref_fouls_avg_before",
        "ref_fouls_last5_before",
        "ref_fouls_last10_before",
        "ref_fouls_last20_before",
        "ref_fouls_shrunk_before",
        "ref_fouls_delta_before",
        "ref_red_avg_before",
        "ref_red_last5_before",
        "ref_red_last10_before",
        "ref_red_last20_before",
        "ref_experience_log",
        "league_yellow_avg_before",
        "league_fouls_avg_before",
        "league_red_avg_before",
    ]

    return [
        c for c in candidates
        if c in df.columns
        and pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.30
    ]


def build_side_rows(match_df):
    ref_cols = referee_feature_columns(match_df)

    rows = []

    for _, r in match_df.iterrows():
        common = {
            "match_id": int(r["match_id"]),
            "season": int(r["season"]),
            "utc_time": r["utc_time"],
            "match_day": r["match_day"],
            "referee": r.get("referee", np.nan),
            "league_last50_cards": r.get("league_last50_cards", np.nan),
            "league_season_cards": r.get("league_season_cards", np.nan),
        }

        for col in ref_cols:
            common[col] = r.get(col, np.nan)

        home = {
            **common,
            "side": "home",
            "team": r["home_team"],
            "opponent": r["away_team"],
            "target_yellow": float(r["home_yellow"]),
            "league_side_cards": r.get("league_last50_home_cards", np.nan),
        }

        away = {
            **common,
            "side": "away",
            "team": r["away_team"],
            "opponent": r["home_team"],
            "target_yellow": float(r["away_yellow"]),
            "league_side_cards": r.get("league_last50_away_cards", np.nan),
        }

        # El árbitro está expresado en TOTAL partido, por eso para una fila
        # local/visitante usamos la mitad como componente orientativo.
        ref_side = r.get("ref_yellow_shrunk_before", np.nan)
        if pd.notna(ref_side):
            ref_side = float(ref_side) / 2.0

        for w in ("last5", "last10", "season"):
            home_yf = r.get(f"home_pre_{w}_yellow_for", np.nan)
            home_ya = r.get(f"home_pre_{w}_yellow_against", np.nan)
            away_yf = r.get(f"away_pre_{w}_yellow_for", np.nan)
            away_ya = r.get(f"away_pre_{w}_yellow_against", np.nan)

            home_ff = r.get(f"home_pre_{w}_fouls_for", np.nan)
            home_fa = r.get(f"home_pre_{w}_fouls_against", np.nan)
            away_ff = r.get(f"away_pre_{w}_fouls_for", np.nan)
            away_fa = r.get(f"away_pre_{w}_fouls_against", np.nan)

            home[f"team_yellow_for_{w}"] = home_yf
            home[f"opp_yellow_against_{w}"] = away_ya
            home[f"team_fouls_for_{w}"] = home_ff
            home[f"opp_fouls_against_{w}"] = away_fa

            away[f"team_yellow_for_{w}"] = away_yf
            away[f"opp_yellow_against_{w}"] = home_ya
            away[f"team_fouls_for_{w}"] = away_ff
            away[f"opp_fouls_against_{w}"] = home_fa

            # V2 equivalente: equipos + liga, SIN árbitro.
            home[f"expected_cards_v2_{w}"] = safe_mean(
                [home_yf, away_ya]
            )
            away[f"expected_cards_v2_{w}"] = safe_mean(
                [away_yf, home_ya]
            )

            # V3: misma información + perfil arbitral.
            home[f"expected_cards_v3_{w}"] = safe_mean(
                [home_yf, away_ya, ref_side]
            )
            away[f"expected_cards_v3_{w}"] = safe_mean(
                [away_yf, home_ya, ref_side]
            )

        rows.extend([home, away])

    df = pd.DataFrame(rows)

    categorical = ["side", "team", "opponent"]

    # IMPORTANTE:
    # No usamos el nombre del árbitro como categoría. Queremos que V3 aprenda
    # por sus estadísticas PRE-PARTIDO, no memorizar identidades ni aliases.
    id_exclude = {
        "match_id",
        "season",
        "utc_time",
        "match_day",
        "target_yellow",
        "referee",
        *categorical,
    }

    numeric_all = [
        c for c in df.columns
        if c not in id_exclude
        and pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.35
    ]

    ref_numeric = set(ref_cols)
    ref_numeric.update(
        c for c in numeric_all if c.startswith("expected_cards_v3_")
    )

    v2_numeric = [
        c for c in numeric_all
        if (
            c not in ref_numeric
            and not c.startswith("expected_cards_v3_")
        )
    ]

    # Evitar que V2 vea indirectamente features específicas de V3.
    v2_numeric = [
        c for c in v2_numeric
        if not c.startswith("ref_")
        and c not in {
            "league_yellow_avg_before",
            "league_fouls_avg_before",
            "league_red_avg_before",
        }
    ]

    # V3 usa exactamente la base de V2 + árbitro.
    v3_numeric = list(v2_numeric)

    for c in ref_cols:
        if c in numeric_all and c not in v3_numeric:
            v3_numeric.append(c)

    for w in ("last5", "last10", "season"):
        col = f"expected_cards_v3_{w}"
        if col in numeric_all and col not in v3_numeric:
            v3_numeric.append(col)

    # Las expected V2 siguen siendo válidas en ambos modelos.
    return df, v2_numeric, v3_numeric, categorical, ref_cols


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

    w = np.power(
        0.5,
        days.to_numpy(dtype=float) / HALF_LIFE_DAYS,
    )

    return np.clip(w, 0.15, 1.0)


def estimate_nb_alpha(total_cards):
    y = pd.to_numeric(
        pd.Series(total_cards),
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if len(y) < 30:
        return 0.10

    mu = float(np.mean(y))
    var = float(np.var(y, ddof=1))

    if mu <= 0:
        return 0.10

    return float(
        np.clip(
            (var - mu) / (mu * mu),
            0.01,
            1.50,
        )
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


def walkforward(match_df, side_df, v2_numeric, v3_numeric, categorical):
    rows = []

    v2_cols = v2_numeric + categorical
    v3_cols = v3_numeric + categorical

    days = sorted(
        match_df.loc[
            match_df["season"] >= TEST_START_SEASON,
            "match_day",
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

        weights = recency_weights(
            train_side,
            day,
        )

        model_v2 = make_model(
            v2_numeric,
            categorical,
        )

        model_v3 = make_model(
            v3_numeric,
            categorical,
        )

        model_v2.fit(
            train_side[v2_cols],
            train_side["target_yellow"],
            model__sample_weight=weights,
        )

        model_v3.fit(
            train_side[v3_cols],
            train_side["target_yellow"],
            model__sample_weight=weights,
        )

        test_side = test_side.copy()

        test_side["pred_v2"] = np.clip(
            model_v2.predict(test_side[v2_cols]),
            0.05,
            5.0,
        )

        test_side["pred_v3"] = np.clip(
            model_v3.predict(test_side[v3_cols]),
            0.05,
            5.0,
        )

        train_matches = match_df[
            match_df["match_day"] < day
        ].copy()

        match_weights = recency_weights(
            pd.DataFrame(
                {"match_day": train_matches["match_day"]}
            ),
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

            home = sides[sides["side"] == "home"]
            away = sides[sides["side"] == "away"]

            if home.empty or away.empty:
                continue

            v2_home = float(home.iloc[0]["pred_v2"])
            v2_away = float(away.iloc[0]["pred_v2"])
            v3_home = float(home.iloc[0]["pred_v3"])
            v3_away = float(away.iloc[0]["pred_v3"])

            mu_v2 = v2_home + v2_away
            mu_v3 = v3_home + v3_away

            for line in CARD_LINES:
                market = f"over_{str(line).replace('.', '_')}"

                rows.append(
                    {
                        "match_id": int(m["match_id"]),
                        "season": int(m["season"]),
                        "match_day": str(day.date()),
                        "home_team": m["home_team"],
                        "away_team": m["away_team"],
                        "referee": m.get("referee", np.nan),
                        "ref_matches_before": m.get(
                            "ref_matches_before", np.nan
                        ),
                        "ref_yellow_shrunk_before": m.get(
                            "ref_yellow_shrunk_before", np.nan
                        ),
                        "ref_yellow_delta_before": m.get(
                            "ref_yellow_delta_before", np.nan
                        ),
                        "total_yellow": float(m["total_yellow"]),
                        "mu_v2": mu_v2,
                        "mu_v3": mu_v3,
                        "ref_mu_effect": mu_v3 - mu_v2,
                        "nb_alpha": nb_alpha,
                        "market": market,
                        "line": line,
                        "v2_raw_probability": over_prob(
                            line, mu_v2, nb_alpha
                        ),
                        "v3_raw_probability": over_prob(
                            line, mu_v3, nb_alpha
                        ),
                        "baseline_probability": over_prob(
                            line, baseline_mu, nb_alpha
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
                f"train_matches={len(train_matches)}"
            )

    return pd.DataFrame(rows)


def choose_weight(history, market, prob_col):
    h = history[
        history["market"] == market
    ]

    if len(h) < 150:
        return 0.50

    y = h["actual"].to_numpy(dtype=int)
    raw = h[prob_col].to_numpy(dtype=float)
    base = h["baseline_probability"].to_numpy(dtype=float)

    best_w = 0.0
    best_brier = float("inf")

    for w in np.arange(0.0, 1.0001, 0.05):
        p = w * raw + (1.0 - w) * base
        b = brier_score_loss(y, p)

        if b < best_brier:
            best_brier = b
            best_w = float(w)

    return best_w


def apply_shrink(pred):
    pred = pred.copy()

    pred["v2_shrink_weight"] = np.nan
    pred["v3_shrink_weight"] = np.nan
    pred["v2_probability"] = np.nan
    pred["v3_probability"] = np.nan

    for season in sorted(pred["season"].unique()):
        for market in sorted(pred["market"].unique()):
            mask = (
                (pred["season"] == season)
                & (pred["market"] == market)
            )

            prior = pred[
                pred["season"] < season
            ]

            w2 = choose_weight(
                prior,
                market,
                "v2_raw_probability",
            )

            w3 = choose_weight(
                prior,
                market,
                "v3_raw_probability",
            )

            base = pred.loc[
                mask, "baseline_probability"
            ].to_numpy(dtype=float)

            raw2 = pred.loc[
                mask, "v2_raw_probability"
            ].to_numpy(dtype=float)

            raw3 = pred.loc[
                mask, "v3_raw_probability"
            ].to_numpy(dtype=float)

            pred.loc[
                mask, "v2_shrink_weight"
            ] = w2

            pred.loc[
                mask, "v3_shrink_weight"
            ] = w3

            pred.loc[
                mask, "v2_probability"
            ] = w2 * raw2 + (1.0 - w2) * base

            pred.loc[
                mask, "v3_probability"
            ] = w3 * raw3 + (1.0 - w3) * base

    return pred


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def safe_logloss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    try:
        return float(log_loss(y, p, labels=[0, 1]))
    except Exception:
        return np.nan


def summarize(pred):
    rows = []

    for market, m in pred.groupby("market"):
        line = float(m["line"].iloc[0])

        groups = [("GLOBAL", m)]

        for season, g in m.groupby("season"):
            groups.append((str(int(season)), g))

        for period, g in groups:
            y = g["actual"].to_numpy(dtype=int)
            p2 = g["v2_probability"].to_numpy(dtype=float)
            p3 = g["v3_probability"].to_numpy(dtype=float)
            pb = g["baseline_probability"].to_numpy(dtype=float)

            b2 = float(brier_score_loss(y, p2))
            b3 = float(brier_score_loss(y, p3))
            bb = float(brier_score_loss(y, pb))

            rows.append(
                {
                    "market": market,
                    "line": line,
                    "period": period,
                    "matches": len(g),
                    "hit_rate": float(np.mean(y)),
                    "auc_v2": safe_auc(y, p2),
                    "auc_v3": safe_auc(y, p3),
                    "delta_auc_v3_minus_v2": (
                        safe_auc(y, p3) - safe_auc(y, p2)
                        if pd.notna(safe_auc(y, p3))
                        and pd.notna(safe_auc(y, p2))
                        else np.nan
                    ),
                    "brier_baseline": bb,
                    "brier_v2": b2,
                    "brier_v3": b3,
                    "delta_brier_v2_minus_v3": b2 - b3,
                    "logloss_v2": safe_logloss(y, p2),
                    "logloss_v3": safe_logloss(y, p3),
                    "avg_v2_weight": float(
                        g["v2_shrink_weight"].mean()
                    ),
                    "avg_v3_weight": float(
                        g["v3_shrink_weight"].mean()
                    ),
                    "avg_ref_mu_effect": float(
                        pd.to_numeric(
                            g["ref_mu_effect"],
                            errors="coerce",
                        ).mean()
                    ),
                    "avg_abs_ref_mu_effect": float(
                        pd.to_numeric(
                            g["ref_mu_effect"],
                            errors="coerce",
                        ).abs().mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def decision_for_market(summary, market):
    g = summary[summary["market"] == market].copy()

    global_row = g[g["period"] == "GLOBAL"]
    season_rows = g[g["period"].isin(["2024", "2025", "2026"])]

    if global_row.empty:
        return "SIN DATOS"

    gr = global_row.iloc[0]

    global_brier_win = gr["delta_brier_v2_minus_v3"] > 0
    global_auc_ok = (
        pd.isna(gr["delta_auc_v3_minus_v2"])
        or gr["delta_auc_v3_minus_v2"] >= -0.005
    )

    season_brier_wins = int(
        (season_rows["delta_brier_v2_minus_v3"] > 0).sum()
    )

    if (
        global_brier_win
        and global_auc_ok
        and season_brier_wins >= 3
    ):
        return "V3 FUERTE"

    if (
        global_brier_win
        and global_auc_ok
        and season_brier_wins >= 2
    ):
        return "V3 APROBADO"

    if global_brier_win and season_brier_wins >= 1:
        return "V3 SECUNDARIO"

    return "V2 MEJOR / NO ACTIVAR V3"


def train_final(side_df, v3_numeric, categorical):
    feature_cols = v3_numeric + categorical

    cutoff = (
        side_df["match_day"].max()
        + pd.Timedelta(days=1)
    )

    weights = recency_weights(
        side_df,
        cutoff,
    )

    model = make_model(
        v3_numeric,
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
        V3_MODEL_PATH,
    )

    return model


def main():
    print()
    print("=" * 96)
    print("LALIGA EDGE PRO - CARDS V3 REFEREE")
    print("=" * 96)
    print(
        "Comparación limpia V2 (equipos + liga) vs "
        "V3 (mismo modelo + comportamiento arbitral PRE-PARTIDO)."
    )
    print(
        "El nombre del árbitro NO se usa como categoría: "
        "solo sus estadísticas históricas previas."
    )
    print()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DB_PATH) as conn:
        print("1/7 Cargando datos...")
        team_df = load_team_features(conn)
        ref_df = load_ref_features(conn)

        print("2/7 Construyendo partidos...")
        match_df = pair_matches(team_df)
        print(f"    Partidos de equipo: {len(match_df)}")

        print("3/7 Emparejando árbitros...")
        match_df = attach_referee_features(
            match_df,
            ref_df,
        )

        print("4/7 Construyendo V2 y V3...")
        (
            side_df,
            v2_numeric,
            v3_numeric,
            categorical,
            ref_cols,
        ) = build_side_rows(match_df)

        print(
            f"    Filas equipo: {len(side_df)} | "
            f"V2 numéricas={len(v2_numeric)} | "
            f"V3 numéricas={len(v3_numeric)} | "
            f"features árbitro={len(ref_cols)}"
        )

        print()
        print("    Features arbitrales usadas:")
        for c in ref_cols:
            print(f"      - {c}")

        print()
        print("5/7 Walk-forward V2 vs V3...")
        pred = walkforward(
            match_df,
            side_df,
            v2_numeric,
            v3_numeric,
            categorical,
        )

        if pred.empty:
            raise RuntimeError(
                "No se generaron predicciones OOS."
            )

        print("6/7 Calibrando shrink + métricas...")
        pred = apply_shrink(pred)
        summary = summarize(pred)

        pred.to_sql(
            "cards_v3_referee_backtest_predictions",
            conn,
            if_exists="replace",
            index=False,
        )

        summary.to_sql(
            "cards_v3_referee_backtest_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.commit()

        pred.to_csv(
            REPORTS_DIR
            / "laliga_cards_v3_referee_backtest_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary.to_csv(
            REPORTS_DIR
            / "laliga_cards_v3_referee_backtest_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("7/7 Entrenando modelo V3 final...")
        train_final(
            side_df,
            v3_numeric,
            categorical,
        )

        metadata = {
            "version": "LaLiga Cards V3 Referee",
            "model_path": str(V3_MODEL_PATH.name),
            "v2_numeric": v2_numeric,
            "v3_numeric": v3_numeric,
            "categorical": categorical,
            "referee_features": ref_cols,
            "card_lines": CARD_LINES,
            "test_start_season": TEST_START_SEASON,
            "poisson_alpha": POISSON_ALPHA,
            "half_life_days": HALF_LIFE_DAYS,
            "uses_referee_name_as_category": False,
            "summary": summary.to_dict(orient="records"),
        }

        joblib.dump(
            metadata,
            V3_META_PATH,
        )

    print()
    print("=" * 96)
    print("CARDS V2 vs CARDS V3 REFEREE")
    print("=" * 96)
    print(
        "ΔBrier positivo = V3 mejor. "
        "ΔAUC positivo = V3 discrimina mejor."
    )

    for line in CARD_LINES:
        market = f"over_{str(line).replace('.', '_')}"

        print()
        print(f"TOTAL AMARILLAS OVER {line}")
        print("-" * 96)
        print(
            "Periodo |   N | AUC V2 | AUC V3 | ΔAUC  | "
            "Brier V2 | Brier V3 | ΔBrier"
        )

        g = summary[
            summary["market"] == market
        ]

        for r in g.itertuples(index=False):
            auc2 = (
                f"{r.auc_v2:.3f}"
                if pd.notna(r.auc_v2)
                else " N/A "
            )
            auc3 = (
                f"{r.auc_v3:.3f}"
                if pd.notna(r.auc_v3)
                else " N/A "
            )
            da = (
                f"{r.delta_auc_v3_minus_v2:+.3f}"
                if pd.notna(r.delta_auc_v3_minus_v2)
                else " N/A "
            )

            print(
                f"{r.period:>7} | "
                f"{int(r.matches):3d} | "
                f"{auc2:>6} | "
                f"{auc3:>6} | "
                f"{da:>6} | "
                f"{r.brier_v2:.4f}  | "
                f"{r.brier_v3:.4f}  | "
                f"{r.delta_brier_v2_minus_v3:+.4f}"
            )

        print(
            f"DECISIÓN: {decision_for_market(summary, market)}"
        )

    print()
    print("=" * 96)
    print("EFECTO DEL ÁRBITRO")
    print("=" * 96)

    match_effect = (
        pred[
            [
                "match_id",
                "home_team",
                "away_team",
                "referee",
                "ref_matches_before",
                "ref_yellow_shrunk_before",
                "ref_yellow_delta_before",
                "ref_mu_effect",
            ]
        ]
        .drop_duplicates("match_id")
        .copy()
    )

    print(
        "Cambio medio absoluto en tarjetas esperadas V3 vs V2: "
        f"{match_effect['ref_mu_effect'].abs().mean():.3f}"
    )

    top_up = match_effect.sort_values(
        "ref_mu_effect",
        ascending=False,
    ).head(8)

    top_down = match_effect.sort_values(
        "ref_mu_effect",
        ascending=True,
    ).head(8)

    print()
    print("PARTIDOS DONDE EL ÁRBITRO MÁS SUBIÓ LA EXPECTATIVA:")
    print("-" * 96)

    for r in top_up.itertuples(index=False):
        print(
            f"{str(r.referee):<34} | "
            f"{r.home_team} - {r.away_team} | "
            f"Δmu={r.ref_mu_effect:+.2f} | "
            f"ref_avg={r.ref_yellow_shrunk_before:.2f} "
            f"(N={int(r.ref_matches_before) if pd.notna(r.ref_matches_before) else 0})"
        )

    print()
    print("PARTIDOS DONDE EL ÁRBITRO MÁS BAJÓ LA EXPECTATIVA:")
    print("-" * 96)

    for r in top_down.itertuples(index=False):
        print(
            f"{str(r.referee):<34} | "
            f"{r.home_team} - {r.away_team} | "
            f"Δmu={r.ref_mu_effect:+.2f} | "
            f"ref_avg={r.ref_yellow_shrunk_before:.2f} "
            f"(N={int(r.ref_matches_before) if pd.notna(r.ref_matches_before) else 0})"
        )

    print()
    print("=" * 96)
    print("ARCHIVOS GENERADOS")
    print("=" * 96)
    print(f"Modelo: {V3_MODEL_PATH}")
    print(f"Metadata: {V3_META_PATH}")
    print(
        "NO se ha sustituido Cards V2. "
        "Solo lo cambiaremos en la app si V3 gana el backtest."
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
