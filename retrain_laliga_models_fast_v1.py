from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import joblib

import laliga_goals_v4_fixed as goals
import laliga_corners_v1_fixed as corners
import laliga_shots_on_target_v1 as sot
import laliga_cards_v3_referee as cards


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "laliga.db"
MODELS_DIR = BASE_DIR / "models"

EXPECTED_MODELS = [
    MODELS_DIR / "laliga_goals_v4_team_strength_poisson.joblib",
    MODELS_DIR / "laliga_corners_v1_poisson.joblib",
    MODELS_DIR / "laliga_shots_on_target_v1_poisson.joblib",
    MODELS_DIR / "laliga_cards_v3_referee_side_poisson.joblib",
]



def table_exists(conn, name):
    return conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name=?
        LIMIT 1
        """,
        (name,),
    ).fetchone() is not None


def validate_training_sync(conn):
    season_row = conn.execute(
        """
        SELECT MAX(CAST(season AS INTEGER))
        FROM fotmob_matches
        WHERE COALESCE(finished, 0) = 1
        """
    ).fetchone()

    if not season_row or season_row[0] is None:
        raise RuntimeError(
            "No se puede detectar la temporada actual."
        )

    season = int(season_row[0])

    finished = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM fotmob_matches
            WHERE COALESCE(finished, 0) = 1
              AND CAST(season AS INTEGER) = ?
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
              AND COALESCE(cancelled, 0) = 0
              AND COALESCE(valid_for_model, 1) = 1
            """,
            (season,),
        ).fetchone()[0]
        or 0
    )

    if not table_exists(conn, "team_pre_match_features"):
        raise RuntimeError(
            "Falta team_pre_match_features."
        )

    feature_ids = {
        int(r[0])
        for r in conn.execute(
            """
            SELECT DISTINCT match_id
            FROM team_pre_match_features
            WHERE CAST(season AS INTEGER) = ?
            """,
            (season,),
        ).fetchall()
        if r[0] is not None
    }

    current_ids = {
        int(r[0])
        for r in conn.execute(
            """
            SELECT match_id
            FROM fotmob_matches
            WHERE COALESCE(finished, 0) = 1
              AND CAST(season AS INTEGER) = ?
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
              AND COALESCE(cancelled, 0) = 0
              AND COALESCE(valid_for_model, 1) = 1
            """,
            (season,),
        ).fetchall()
        if r[0] is not None
    }

    if feature_ids != current_ids:
        missing = sorted(current_ids - feature_ids)
        extra = sorted(feature_ids - current_ids)

        raise RuntimeError(
            "Features desincronizadas con la temporada actual. "
            f"DB={len(current_ids)}, features={len(feature_ids)}, "
            f"faltan={missing[:10]}, sobran={extra[:10]}"
        )

    # SOT actual: deben existir 2 filas/equipo por partido y tener dato.
    sot_rows = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM team_pre_match_features
            WHERE CAST(season AS INTEGER) = ?
              AND shots_on_target_for IS NOT NULL
            """,
            (season,),
        ).fetchone()[0]
        or 0
    )

    if sot_rows < finished * 2:
        raise RuntimeError(
            "Tiros a puerta incompletos en features actuales. "
            f"Esperados al menos {finished*2}, disponibles={sot_rows}."
        )

    if not table_exists(conn, "referee_pre_match_features"):
        raise RuntimeError(
            "Falta referee_pre_match_features."
        )

    ref_ids = {
        int(r[0])
        for r in conn.execute(
            """
            SELECT DISTINCT match_id
            FROM referee_pre_match_features
            WHERE CAST(season AS INTEGER) = ?
            """,
            (season,),
        ).fetchall()
        if r[0] is not None
    }

    shared = current_ids.intersection(ref_ids)

    print()
    print("=" * 88)
    print("VALIDACIÓN PREVIA AL REENTRENAMIENTO")
    print("=" * 88)
    print(
        f"Temporada {season}/{season+1}: "
        f"DB={len(current_ids)} · features={len(feature_ids)} · "
        f"refs compartidos={len(shared)} · SOT rows={sot_rows}"
    )

    if len(shared) != finished:
        missing_ref = sorted(current_ids - ref_ids)

        raise RuntimeError(
            "Los árbitros de la temporada actual no están alineados "
            "por match_id con las features. "
            f"Esperados={finished}, compartidos={len(shared)}, "
            f"faltan={missing_ref[:10]}"
        )

    print("✅ DB / features / SOT / árbitros: SINCRONIZADOS")
    print("=" * 88)



def train_goals(conn):
    print("\n[1/4] ⚽ Reentrenando GOALS V4 final...")
    team_df = goals.load_team_features(conn)
    match_df = goals.pair_matches(team_df)
    long_df, numeric, categorical = goals.build_long_rows(match_df)
    goals.train_final_model(long_df, numeric, categorical)
    print(f"      OK · partidos={len(match_df)} · filas={len(long_df)}")


def train_corners(conn):
    print("\n[2/4] 🚩 Reentrenando CORNERS V1 final...")
    team_df = corners.load_team_features(conn)
    match_df = corners.pair_matches(team_df)
    long_df, numeric, categorical = corners.build_long_rows(match_df)
    corners.train_final(long_df, numeric, categorical)
    print(f"      OK · partidos={len(match_df)} · filas={len(long_df)}")


def train_sot(conn):
    print("\n[3/4] 🎯 Reentrenando SHOTS ON TARGET V1 final...")
    team_df = sot.load_team_features(conn)
    match_df = sot.pair_matches(team_df)
    long_df, numeric, categorical = sot.build_long_rows(match_df)
    sot.train_final(long_df, numeric, categorical)
    print(f"      OK · partidos={len(match_df)} · filas={len(long_df)}")


def train_cards(conn):
    print("\n[4/4] 🟨 Reentrenando CARDS V3 REFEREE final...")
    team_df = cards.load_team_features(conn)
    ref_df = cards.load_ref_features(conn)
    match_df = cards.pair_matches(team_df)
    match_df = cards.attach_referee_features(match_df, ref_df)
    (
        side_df,
        _v2_numeric,
        v3_numeric,
        categorical,
        ref_cols,
    ) = cards.build_side_rows(match_df)

    cards.train_final(side_df, v3_numeric, categorical)

    metadata = {
        "version": "LaLiga Cards V3 Referee - fast retrain",
        "v3_numeric": v3_numeric,
        "categorical": categorical,
        "referee_features": ref_cols,
    }
    joblib.dump(
        metadata,
        MODELS_DIR / "laliga_cards_v3_referee_meta_fast.joblib",
    )

    print(
        f"      OK · partidos={len(match_df)} · "
        f"filas={len(side_df)} · ref_features={len(ref_cols)}"
    )



def save_training_status(conn):
    season_row = conn.execute(
        """
        SELECT MAX(season)
        FROM fotmob_matches
        WHERE finished = 1
        """
    ).fetchone()

    season = (
        int(season_row[0])
        if season_row and season_row[0] is not None
        else None
    )

    current_count = 0

    if season is not None:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM fotmob_matches
            WHERE finished = 1
              AND season = ?
            """,
            (season,),
        ).fetchone()

        current_count = int(row[0] or 0)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_status (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
        """
    )

    now = datetime.now().isoformat(timespec="seconds")

    values = {
        "laliga_models_trained_season": season,
        "laliga_models_trained_season_match_count": current_count,
        "laliga_models_last_trained_at": now,
        "laliga_v113_sync_done": 1,
        "laliga_v114_sync_done": 1,
        "laliga_v115_sync_done": 1,
    }

    for key, value in values.items():
        conn.execute(
            """
            INSERT INTO system_status(key, value, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at
            """,
            (key, str(value), now),
        )

    conn.commit()


def main():
    print()
    print("=" * 88)
    print("FOOTBALL EDGE PRO - REENTRENAMIENTO RAPIDO LALIGA")
    print("=" * 88)
    print(
        "Se reentrenan SOLO los modelos finales con todos los datos actuales.\n"
        "No repetimos los walk-forward/backtests cada jornada."
    )

    if not DB_PATH.exists():
        raise SystemExit(f"No existe {DB_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        validate_training_sync(conn)
        train_goals(conn)
        train_corners(conn)
        train_sot(conn)
        train_cards(conn)
        save_training_status(conn)

    missing = [p.name for p in EXPECTED_MODELS if not p.exists()]
    if missing:
        raise RuntimeError(
            "Faltan modelos tras el reentrenamiento: " + ", ".join(missing)
        )

    print()
    print("=" * 88)
    print("MODELOS LALIGA REENTRENADOS CORRECTAMENTE")
    print("=" * 88)
    for path in EXPECTED_MODELS:
        print(f"✅ {path.name} · {path.stat().st_size / 1024:.1f} KB")
    print("=" * 88)


if __name__ == "__main__":
    main()
