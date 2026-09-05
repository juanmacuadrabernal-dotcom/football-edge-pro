from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


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


def logit_transform(prob):
    p = np.clip(
        np.asarray(prob, dtype=float),
        1e-5,
        1 - 1e-5,
    )
    return np.log(p / (1 - p)).reshape(-1, 1)


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def safe_logloss(y, p):
    p = np.clip(
        np.asarray(p, dtype=float),
        1e-6,
        1 - 1e-6,
    )
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


def make_calibrator():
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
    )


def load_data(conn):
    pred = pd.read_sql_query(
        """
        SELECT *
        FROM goals_v2_backtest_predictions
        ORDER BY season, utc_time, match_id
        """,
        conn,
    )

    summary = pd.read_sql_query(
        """
        SELECT *
        FROM goals_v2_backtest_summary
        WHERE market <> '__goals_mae__'
        ORDER BY market, test_season
        """,
        conn,
    )

    if pred.empty:
        raise RuntimeError(
            "No existe backtest V2. Ejecuta primero train_goal_model_v2.py"
        )

    pred["season"] = pd.to_numeric(
        pred["season"],
        errors="coerce",
    ).astype("Int64")

    pred["probability"] = pd.to_numeric(
        pred["probability"],
        errors="coerce",
    )

    pred["actual"] = pd.to_numeric(
        pred["actual"],
        errors="coerce",
    ).astype("Int64")

    return pred.dropna(
        subset=["season", "probability", "actual"]
    ).copy(), summary


def walk_forward_calibration(pred, summary):
    rows = []
    calibrated_predictions = []

    for market in MARKETS:
        m = pred[
            pred["market"] == market
        ].copy()

        if m.empty:
            continue

        seasons = sorted(
            int(x)
            for x in m["season"].unique()
        )

        for test_season in seasons:
            train_cal = m[
                m["season"] < test_season
            ].copy()

            test = m[
                m["season"] == test_season
            ].copy()

            # La primera temporada OOS no tiene OOS previo suficiente
            # para calibrar de forma limpia.
            if len(train_cal) < 150:
                continue

            x_train = logit_transform(
                train_cal["probability"].to_numpy()
            )
            y_train = train_cal["actual"].astype(int).to_numpy()

            if len(np.unique(y_train)) < 2:
                continue

            calibrator = make_calibrator()
            calibrator.fit(
                x_train,
                y_train,
            )

            raw_p = test["probability"].to_numpy(dtype=float)

            cal_p = calibrator.predict_proba(
                logit_transform(raw_p)
            )[:, 1]

            y = test["actual"].astype(int).to_numpy()

            raw_brier = float(
                brier_score_loss(y, raw_p)
            )

            cal_brier = float(
                brier_score_loss(y, cal_p)
            )

            raw_logloss = safe_logloss(y, raw_p)
            cal_logloss = safe_logloss(y, cal_p)

            auc = safe_auc(y, raw_p)

            base_row = summary[
                (summary["market"] == market)
                & (summary["test_season"] == test_season)
            ]

            if not base_row.empty:
                baseline_brier = float(
                    base_row.iloc[0]["brier_baseline"]
                )
            else:
                baseline_brier = None

            rows.append(
                {
                    "market": market,
                    "test_season": test_season,
                    "calibration_train_rows": len(train_cal),
                    "test_rows": len(test),
                    "auc": auc,
                    "raw_brier": raw_brier,
                    "calibrated_brier": cal_brier,
                    "baseline_brier": baseline_brier,
                    "calibration_improvement": (
                        raw_brier - cal_brier
                    ),
                    "vs_baseline_improvement": (
                        baseline_brier - cal_brier
                        if baseline_brier is not None
                        else None
                    ),
                    "raw_logloss": raw_logloss,
                    "calibrated_logloss": cal_logloss,
                }
            )

            temp = test.copy()
            temp["raw_probability"] = raw_p
            temp["calibrated_probability"] = cal_p

            calibrated_predictions.append(
                temp[
                    [
                        "match_id",
                        "season",
                        "utc_time",
                        "home_team",
                        "away_team",
                        "market",
                        "actual",
                        "raw_probability",
                        "calibrated_probability",
                    ]
                ]
            )

    results = pd.DataFrame(rows)

    all_calibrated = (
        pd.concat(
            calibrated_predictions,
            ignore_index=True,
        )
        if calibrated_predictions
        else pd.DataFrame()
    )

    return results, all_calibrated


def fit_final_calibrators(pred):
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "method": "Platt scaling on V2 out-of-sample predictions",
        "markets": {},
    }

    for market in MARKETS:
        m = pred[
            pred["market"] == market
        ].copy()

        if len(m) < 150:
            continue

        x = logit_transform(
            m["probability"].to_numpy()
        )

        y = m["actual"].astype(int).to_numpy()

        if len(np.unique(y)) < 2:
            continue

        calibrator = make_calibrator()
        calibrator.fit(x, y)

        path = (
            MODELS_DIR
            / f"goals_v2_calibrator_{market}.joblib"
        )

        joblib.dump(
            calibrator,
            path,
        )

        metadata["markets"][market] = {
            "label": MARKETS[market],
            "samples": len(m),
            "model_path": str(
                path.relative_to(BASE_DIR)
            ),
        }

    (
        MODELS_DIR
        / "goals_v2_calibration_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata


def verdict_table(results):
    rows = []

    for market in MARKETS:
        g = results[
            results["market"] == market
        ].copy()

        if g.empty:
            continue

        brier_better = (
            g["vs_baseline_improvement"] > 0
        ).sum()

        folds = len(g)

        auc_values = g["auc"].dropna()

        median_auc = (
            float(auc_values.median())
            if not auc_values.empty
            else None
        )

        avg_cal_brier = float(
            g["calibrated_brier"].mean()
        )

        baseline_vals = g["baseline_brier"].dropna()

        avg_base_brier = (
            float(baseline_vals.mean())
            if not baseline_vals.empty
            else None
        )

        approved = (
            folds >= 2
            and brier_better == folds
            and median_auc is not None
            and median_auc >= 0.53
        )

        rows.append(
            {
                "market": market,
                "label": MARKETS[market],
                "folds": folds,
                "folds_better_than_baseline": int(brier_better),
                "median_auc": median_auc,
                "avg_calibrated_brier": avg_cal_brier,
                "avg_baseline_brier": avg_base_brier,
                "status": (
                    "CANDIDATO"
                    if approved
                    else "NO APROBADO"
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print()
    print("=" * 72)
    print("ELITESERIEN EDGE PRO - CALIBRACION GOLES V2 FIXED")
    print("=" * 72)
    print("Calibración walk-forward con Platt scaling.")
    print()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DB_PATH) as conn:
        print("1/4 Cargando predicciones OOS de V2...")
        pred, summary = load_data(conn)

        print(
            f"    Predicciones: {len(pred)}"
        )

        print("2/4 Calibrando walk-forward...")
        results, calibrated = walk_forward_calibration(
            pred,
            summary,
        )

        print("3/4 Guardando resultados...")

        results.to_sql(
            "goals_v2_calibration_summary",
            conn,
            if_exists="replace",
            index=False,
        )

        if not calibrated.empty:
            calibrated.to_sql(
                "goals_v2_calibrated_predictions",
                conn,
                if_exists="replace",
                index=False,
            )

        conn.commit()

        print("4/4 Entrenando calibradores finales...")
        fit_final_calibrators(pred)

    verdict_df = verdict_table(results)

    results.to_csv(
        REPORTS_DIR
        / "goals_v2_calibration_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not calibrated.empty:
        calibrated.to_csv(
            REPORTS_DIR
            / "goals_v2_calibrated_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    verdict_df.to_csv(
        REPORTS_DIR
        / "goals_v2_market_verdict.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print("RESULTADOS DE CALIBRACION")
    print("=" * 72)

    for market in MARKETS:
        g = results[
            results["market"] == market
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

            baseline_text = (
                f"{r.baseline_brier:.3f}"
                if pd.notna(r.baseline_brier)
                else "N/A"
            )

            verdict_text = (
                "MEJORA BASELINE"
                if (
                    pd.notna(r.vs_baseline_improvement)
                    and r.vs_baseline_improvement > 0
                )
                else "NO MEJORA"
            )

            print(
                f"Test {int(r.test_season)} | "
                f"AUC={auc_text} | "
                f"Raw={r.raw_brier:.3f} | "
                f"Cal={r.calibrated_brier:.3f} | "
                f"Base={baseline_text} | "
                f"{verdict_text}"
            )

    print()
    print("=" * 72)
    print("VEREDICTO POR MERCADO")
    print("=" * 72)

    for r in verdict_df.itertuples(index=False):
        auc_text = (
            f"{r.median_auc:.3f}"
            if pd.notna(r.median_auc)
            else "N/A"
        )

        print(
            f"{r.label:<18} | "
            f"AUC mediana={auc_text} | "
            f"mejora baseline "
            f"{r.folds_better_than_baseline}/{r.folds} | "
            f"{r.status}"
        )

    print()
    print("=" * 72)
    print("IMPORTANTE")
    print("=" * 72)
    print("CANDIDATO no significa rentable.")
    print(
        "El siguiente filtro será cuotas + valor esperado (EV) "
        "y backtest de ROI."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
