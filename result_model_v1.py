from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "eliteserien.db"
REPORTS_DIR = BASE_DIR / "reports"

MAX_GOALS = 10
HALF_LIFE_DAYS = 420.0

CLASSES = ["H", "D", "A"]


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-8)
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def result_probs_from_lambdas(lh: float, la: float):
    home = 0.0
    draw = 0.0
    away = 0.0

    for hg in range(MAX_GOALS + 1):
        ph = poisson_pmf(hg, lh)

        for ag in range(MAX_GOALS + 1):
            p = ph * poisson_pmf(ag, la)

            if hg > ag:
                home += p
            elif hg == ag:
                draw += p
            else:
                away += p

    total = home + draw + away

    if total <= 0:
        return 1/3, 1/3, 1/3

    return home / total, draw / total, away / total


def multiclass_brier(y_true, probs):
    mapping = {
        "H": np.array([1.0, 0.0, 0.0]),
        "D": np.array([0.0, 1.0, 0.0]),
        "A": np.array([0.0, 0.0, 1.0]),
    }

    ys = np.vstack([mapping[y] for y in y_true])
    probs = np.asarray(probs, dtype=float)

    # Brier multicategoría normalizado por clases.
    return float(
        np.mean(
            np.sum((probs - ys) ** 2, axis=1) / 3.0
        )
    )


def recency_weights(days_old):
    w = np.power(
        0.5,
        np.asarray(days_old, dtype=float) / HALF_LIFE_DAYS,
    )
    return np.clip(w, 0.15, 1.0)


def load_v4_predictions(conn):
    pred = pd.read_sql_query(
        """
        SELECT *
        FROM goals_v4_backtest_predictions
        WHERE market = 'over_25'
        ORDER BY season, match_day, match_id
        """,
        conn,
    )

    if pred.empty:
        raise RuntimeError(
            "No existe goals_v4_backtest_predictions. "
            "Ejecuta goals_model_v4_champion.py primero."
        )

    pred["match_day"] = pd.to_datetime(
        pred["match_day"],
        errors="coerce",
    )

    pred["lambda_home"] = pd.to_numeric(
        pred["lambda_home"],
        errors="coerce",
    )

    pred["lambda_away"] = pd.to_numeric(
        pred["lambda_away"],
        errors="coerce",
    )

    pred["home_goals"] = pd.to_numeric(
        pred["home_goals"],
        errors="coerce",
    )

    pred["away_goals"] = pd.to_numeric(
        pred["away_goals"],
        errors="coerce",
    )

    pred = pred.dropna(
        subset=[
            "match_day",
            "lambda_home",
            "lambda_away",
            "home_goals",
            "away_goals",
        ]
    ).copy()

    return pred


def load_all_historical_results(conn):
    hist = pd.read_sql_query(
        """
        SELECT
            match_id,
            season,
            utc_time,
            team AS home_team,
            opponent AS away_team,
            goals_for AS home_goals,
            goals_against AS away_goals
        FROM team_match_history
        WHERE venue = 'home'
        ORDER BY utc_time, match_id
        """,
        conn,
    )

    if hist.empty:
        raise RuntimeError(
            "team_match_history está vacía."
        )

    hist["utc_time"] = pd.to_datetime(
        hist["utc_time"],
        utc=True,
        errors="coerce",
    )

    hist["match_day"] = (
        hist["utc_time"]
        .dt.tz_convert(None)
        .dt.floor("D")
    )

    hist["home_goals"] = pd.to_numeric(
        hist["home_goals"], errors="coerce"
    )

    hist["away_goals"] = pd.to_numeric(
        hist["away_goals"], errors="coerce"
    )

    hist = hist.dropna(
        subset=["match_day", "home_goals", "away_goals"]
    ).copy()

    hist["actual"] = np.where(
        hist["home_goals"] > hist["away_goals"],
        "H",
        np.where(
            hist["home_goals"] == hist["away_goals"],
            "D",
            "A",
        ),
    )

    return hist


def baseline_probs(hist, cutoff):
    prior = hist[
        hist["match_day"] < cutoff
    ].copy()

    if prior.empty:
        return np.array([1/3, 1/3, 1/3], dtype=float)

    days_old = (
        cutoff - prior["match_day"]
    ).dt.days.clip(lower=0)

    weights = recency_weights(days_old)

    probs = []

    for cls in CLASSES:
        mask = (prior["actual"] == cls).to_numpy(dtype=float)
        probs.append(
            float(np.average(mask, weights=weights))
        )

    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()

    return probs


def build_raw_predictions(pred, hist):
    rows = []

    for _, r in pred.iterrows():
        p_home, p_draw, p_away = result_probs_from_lambdas(
            r["lambda_home"],
            r["lambda_away"],
        )

        actual = (
            "H"
            if r["home_goals"] > r["away_goals"]
            else "D"
            if r["home_goals"] == r["away_goals"]
            else "A"
        )

        base = baseline_probs(
            hist,
            r["match_day"],
        )

        rows.append(
            {
                "match_id": int(r["match_id"]),
                "season": int(r["season"]),
                "match_day": r["match_day"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "home_goals": float(r["home_goals"]),
                "away_goals": float(r["away_goals"]),
                "lambda_home": float(r["lambda_home"]),
                "lambda_away": float(r["lambda_away"]),
                "actual": actual,
                "raw_home": p_home,
                "raw_draw": p_draw,
                "raw_away": p_away,
                "base_home": float(base[0]),
                "base_draw": float(base[1]),
                "base_away": float(base[2]),
            }
        )

    return pd.DataFrame(rows)


def logprob_features(frame, prefix):
    arr = frame[
        [
            f"{prefix}_home",
            f"{prefix}_draw",
            f"{prefix}_away",
        ]
    ].to_numpy(dtype=float)

    arr = np.clip(arr, 1e-6, 1 - 1e-6)

    return np.log(arr)


def calibrate_walkforward(raw):
    raw = raw.sort_values(
        ["season", "match_day", "match_id"]
    ).copy()

    raw["final_home"] = np.nan
    raw["final_draw"] = np.nan
    raw["final_away"] = np.nan
    raw["calibration"] = "RAW"

    seasons = sorted(
        int(x) for x in raw["season"].unique()
    )

    for season in seasons:
        mask = raw["season"] == season

        prior = raw[
            raw["season"] < season
        ].copy()

        if len(prior) < 200:
            raw.loc[
                mask,
                ["final_home", "final_draw", "final_away"]
            ] = raw.loc[
                mask,
                ["raw_home", "raw_draw", "raw_away"]
            ].to_numpy()

            continue

        X_train = np.hstack(
            [
                logprob_features(prior, "raw"),
                logprob_features(prior, "base"),
            ]
        )

        y_train = prior["actual"].to_numpy()

        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=0.7,
            max_iter=3000,
        )

        model.fit(X_train, y_train)

        test = raw.loc[mask].copy()

        X_test = np.hstack(
            [
                logprob_features(test, "raw"),
                logprob_features(test, "base"),
            ]
        )

        p = model.predict_proba(X_test)

        class_to_idx = {
            c: i for i, c in enumerate(model.classes_)
        }

        final = np.zeros((len(test), 3), dtype=float)

        for j, cls in enumerate(CLASSES):
            final[:, j] = p[:, class_to_idx[cls]]

        raw.loc[
            mask,
            ["final_home", "final_draw", "final_away"]
        ] = final

        raw.loc[mask, "calibration"] = "WALK_FORWARD"

    return raw


def summarize(df):
    rows = []

    groups = [("GLOBAL", df)]

    for season, g in df.groupby("season"):
        groups.append(
            (str(int(season)), g)
        )

    for period, g in groups:
        y = g["actual"].to_numpy()

        final_probs = g[
            ["final_home", "final_draw", "final_away"]
        ].to_numpy(dtype=float)

        raw_probs = g[
            ["raw_home", "raw_draw", "raw_away"]
        ].to_numpy(dtype=float)

        base_probs = g[
            ["base_home", "base_draw", "base_away"]
        ].to_numpy(dtype=float)

        final_pred = np.asarray(CLASSES)[
            np.argmax(final_probs, axis=1)
        ]

        raw_pred = np.asarray(CLASSES)[
            np.argmax(raw_probs, axis=1)
        ]

        base_pred = np.asarray(CLASSES)[
            np.argmax(base_probs, axis=1)
        ]

        rows.append(
            {
                "period": period,
                "matches": len(g),
                "accuracy_final": float(
                    accuracy_score(y, final_pred)
                ),
                "accuracy_raw": float(
                    accuracy_score(y, raw_pred)
                ),
                "accuracy_baseline": float(
                    accuracy_score(y, base_pred)
                ),
                "brier_final": multiclass_brier(
                    y, final_probs
                ),
                "brier_raw": multiclass_brier(
                    y, raw_probs
                ),
                "brier_baseline": multiclass_brier(
                    y, base_probs
                ),
                "logloss_final": float(
                    log_loss(
                        y,
                        final_probs,
                        labels=CLASSES,
                    )
                ),
                "logloss_raw": float(
                    log_loss(
                        y,
                        raw_probs,
                        labels=CLASSES,
                    )
                ),
                "logloss_baseline": float(
                    log_loss(
                        y,
                        base_probs,
                        labels=CLASSES,
                    )
                ),
                "home_rate": float(
                    np.mean(y == "H")
                ),
                "draw_rate": float(
                    np.mean(y == "D")
                ),
                "away_rate": float(
                    np.mean(y == "A")
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print()
    print("=" * 76)
    print("ELITESERIEN EDGE PRO - RESULT MODEL V1")
    print("=" * 76)
    print(
        "1X2 derivado del modelo de goles V4 + "
        "calibración walk-forward."
    )
    print()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DB_PATH) as conn:
        print("1/5 Cargando predicciones V4...")
        pred = load_v4_predictions(conn)

        print("2/5 Cargando histórico de resultados...")
        hist = load_all_historical_results(conn)

        print("3/5 Calculando probabilidades 1X2...")
        raw = build_raw_predictions(
            pred,
            hist,
        )

        print(
            f"    Partidos OOS: {len(raw)}"
        )

        print("4/5 Calibrando walk-forward...")
        final = calibrate_walkforward(raw)

        print("5/5 Evaluando y guardando...")
        summary = summarize(final)

        final.to_sql(
            "result_v1_backtest_predictions",
            conn,
            if_exists="replace",
            index=False,
        )

        summary.to_sql(
            "result_v1_backtest_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.commit()

    final.to_csv(
        REPORTS_DIR
        / "result_v1_backtest_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        REPORTS_DIR
        / "result_v1_backtest_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 76)
    print("BACKTEST RESULT V1")
    print("=" * 76)

    for r in summary.itertuples(index=False):
        brier_verdict = (
            "MEJORA"
            if r.brier_final < r.brier_baseline
            else "NO MEJORA"
        )

        log_verdict = (
            "MEJORA"
            if r.logloss_final < r.logloss_baseline
            else "NO MEJORA"
        )

        print()
        print(
            f"{r.period:>6} | N={int(r.matches):3d}"
        )
        print(
            f"  Accuracy: modelo={r.accuracy_final*100:5.1f}% | "
            f"raw={r.accuracy_raw*100:5.1f}% | "
            f"base={r.accuracy_baseline*100:5.1f}%"
        )
        print(
            f"  Brier:    modelo={r.brier_final:.3f} | "
            f"raw={r.brier_raw:.3f} | "
            f"base={r.brier_baseline:.3f} | "
            f"{brier_verdict}"
        )
        print(
            f"  LogLoss:  modelo={r.logloss_final:.3f} | "
            f"raw={r.logloss_raw:.3f} | "
            f"base={r.logloss_baseline:.3f} | "
            f"{log_verdict}"
        )

    print()
    print("=" * 76)
    print("CRITERIO")
    print("=" * 76)
    print(
        "Si Brier y LogLoss mejoran el baseline de forma "
        "consistente, 1X2 pasa al siguiente filtro: cuotas + EV."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
