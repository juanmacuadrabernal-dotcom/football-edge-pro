from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "laliga.db"
MODELS_DIR = BASE_DIR / "models"
BACKUP_DIR = BASE_DIR / "data" / "backups"

MODEL_FILES = [
    "laliga_goals_v4_team_strength_poisson.joblib",
    "laliga_corners_v1_poisson.joblib",
    "laliga_shots_on_target_v1_poisson.joblib",
    "laliga_cards_v3_referee_side_poisson.joblib",
]


def table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def scalar(conn, sql, params=(), default=0):
    try:
        row = conn.execute(sql, params).fetchone()
        if not row or row[0] is None:
            return default
        return row[0]
    except Exception:
        return default


def status_value(key, default=0):
    if not DB_PATH.exists():
        return default

    with sqlite3.connect(DB_PATH) as conn:
        if not table_exists(conn, "system_status"):
            return default

        value = scalar(
            conn,
            "SELECT value FROM system_status WHERE key=?",
            (key,),
            default,
        )

    try:
        return int(float(value))
    except Exception:
        return value


def create_backup():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = BACKUP_DIR / f"laliga_incremental_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        shutil.copy2(DB_PATH, folder / "laliga.db")

    for name in MODEL_FILES:
        src = MODELS_DIR / name
        if src.exists():
            shutil.copy2(src, folder / name)

    backups = sorted(
        [
            p
            for p in BACKUP_DIR.glob("laliga_incremental_*")
            if p.is_dir()
        ],
        key=lambda p: p.name,
        reverse=True,
    )

    for old in backups[5:]:
        shutil.rmtree(old, ignore_errors=True)

    return folder


def restore_backup(folder):
    print()
    print("⚠️ Restaurando copia anterior...")

    db = folder / "laliga.db"

    if db.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db, DB_PATH)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name in MODEL_FILES:
        src = folder / name
        if src.exists():
            shutil.copy2(src, MODELS_DIR / name)

    print("✅ Restauración terminada.")


def run_step(title, script):
    path = BASE_DIR / script

    if not path.exists():
        raise FileNotFoundError(
            f"Falta {script} en la carpeta del proyecto."
        )

    print()
    print("=" * 96)
    print(title)
    print("=" * 96)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        [sys.executable, "-u", str(path)],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="", flush=True)

    code = process.wait()

    if code != 0:
        raise RuntimeError(
            f"{script} terminó con código {code}."
        )



def current_season_counts():
    with sqlite3.connect(DB_PATH) as conn:
        season = scalar(
            conn,
            """
            SELECT MAX(season)
            FROM fotmob_matches
            WHERE finished = 1
            """,
            default=None,
        )

        if season is None:
            return None, 0, 0, 0, 0

        season = int(season)

        finished = int(
            scalar(
                conn,
                """
                SELECT COUNT(*)
                FROM fotmob_matches
                WHERE finished = 1
                  AND season = ?
                """,
                (season,),
                0,
            )
        )

        team_features = 0
        if table_exists(conn, "team_pre_match_features"):
            team_features = int(
                scalar(
                    conn,
                    """
                    SELECT COUNT(DISTINCT match_id)
                    FROM team_pre_match_features
                    WHERE season = ?
                    """,
                    (season,),
                    0,
                )
            )

        ref_history = 0
        if table_exists(conn, "referee_match_history"):
            ref_history = int(
                scalar(
                    conn,
                    """
                    SELECT COUNT(DISTINCT match_id)
                    FROM referee_match_history
                    WHERE season = ?
                    """,
                    (season,),
                    0,
                )
            )

        ref_features = 0
        if table_exists(conn, "referee_pre_match_features"):
            ref_features = int(
                scalar(
                    conn,
                    """
                    SELECT COUNT(DISTINCT match_id)
                    FROM referee_pre_match_features
                    WHERE season = ?
                    """,
                    (season,),
                    0,
                )
            )

    return (
        season,
        finished,
        team_features,
        ref_history,
        ref_features,
    )


def trained_match_count():
    value = status_value(
        "laliga_models_trained_season_match_count",
        -1,
    )

    try:
        return int(value)
    except Exception:
        return -1


def v115_sync_done():
    value = status_value(
        "laliga_v115_sync_done",
        0,
    )

    try:
        return int(value) == 1
    except Exception:
        return False


def current_sot_coverage():
    with sqlite3.connect(DB_PATH) as conn:
        if not table_exists(conn, "fotmob_match_stats"):
            return 0, 0

        season = scalar(
            conn,
            """
            SELECT MAX(season)
            FROM fotmob_matches
            WHERE finished = 1
            """,
            default=None,
        )

        if season is None:
            return 0, 0

        total = int(
            scalar(
                conn,
                """
                SELECT COUNT(*)
                FROM fotmob_match_stats
                WHERE season = ?
                """,
                (int(season),),
                0,
            )
        )

        covered = int(
            scalar(
                conn,
                """
                SELECT COUNT(*)
                FROM fotmob_match_stats
                WHERE season = ?
                  AND home_shots_on_target IS NOT NULL
                  AND away_shots_on_target IS NOT NULL
                """,
                (int(season),),
                0,
            )
        )

    return covered, total


def final_summary(new_matches, new_refs):
    print()
    print("#" * 96)
    print("🚀 ACTUALIZACIÓN INCREMENTAL COMPLETADA")
    print("#" * 96)

    with sqlite3.connect(DB_PATH) as conn:
        total = scalar(
            conn,
            "SELECT COUNT(*) FROM fotmob_matches WHERE finished = 1",
        )
        season = scalar(
            conn,
            "SELECT MAX(season) FROM fotmob_matches WHERE finished = 1",
            default=None,
        )
        season_count = 0

        if season is not None:
            season_count = scalar(
                conn,
                """
                SELECT COUNT(*)
                FROM fotmob_matches
                WHERE finished = 1 AND season = ?
                """,
                (int(season),),
            )

        upcoming = (
            scalar(
                conn,
                "SELECT COUNT(*) FROM laliga_upcoming_fixtures",
            )
            if table_exists(
                conn,
                "laliga_upcoming_fixtures",
            )
            else 0
        )

        assigned = (
            scalar(
                conn,
                """
                SELECT COUNT(*)
                FROM laliga_upcoming_fixtures
                WHERE referee IS NOT NULL
                  AND TRIM(referee) <> ''
                """,
            )
            if table_exists(
                conn,
                "laliga_upcoming_fixtures",
            )
            else 0
        )

    print(f"✅ Partidos nuevos: {new_matches}")
    print(f"✅ Árbitros nuevos: {new_refs}")
    print(f"✅ Histórico total: {total}")

    if season is not None:
        print(
            f"✅ Temporada {int(season)}/{int(season)+1}: "
            f"{season_count} partidos terminados"
        )

    trained_now = trained_match_count()

    if trained_now == season_count and season_count > 0:
        print(
            f"✅ Features/modelos sincronizados con "
            f"{season_count} partidos de la temporada actual"
        )
    else:
        print(
            f"⚠️ Modelos registrados con {trained_now} partidos "
            f"frente a {season_count} terminados"
        )

    print(
        f"✅ Próxima jornada: {upcoming} partidos · "
        f"árbitro asignado {assigned}/{upcoming}"
    )
    print()
    print(
        "La próxima vez volverá a consultar SOLO la temporada actual en FotMob "
        "y SOLO descargará árbitros que falten."
    )
    print("#" * 96)


def main():
    print()
    print("#" * 96)
    print("FOOTBALL EDGE PRO · LALIGA · ACTUALIZACIÓN INCREMENTAL V11.5")
    print("#" * 96)
    print(
        "NO recorre 2020+. Resultados recientes desde FotMob + solo temporada actual."
    )

    backup = create_backup()
    print(f"\n🛡️ Backup: {backup}")

    try:
        run_step(
            "1/6 · 📥 RESULTADOS NUEVOS FOTMOB · SOLO TEMPORADA ACTUAL",
            "update_laliga_history_v3_fotmob_incremental.py",
        )

        new_matches = status_value(
            "laliga_last_new_matches",
            0,
        )

        run_step(
            "2/6 · 🧑‍⚖️ ÁRBITROS NUEVOS · SOLO TEMPORADA ACTUAL",
            "update_laliga_referees_incremental_v1.py",
        )

        new_refs = status_value(
            "laliga_last_new_referee_matches",
            0,
        )

        (
            season,
            finished_count,
            team_feature_count,
            ref_history_count,
            ref_feature_count,
        ) = current_season_counts()

        sot_covered, sot_total = current_sot_coverage()

        print()
        print(
            f"🔎 Estado temporada {season}/{season+1}: "
            f"finalizados={finished_count} · "
            f"features equipo={team_feature_count} · "
            f"refs={ref_history_count}/{ref_feature_count} · "
            f"SOT={sot_covered}/{sot_total}"
        )

        ref_sync_needed = (
            new_refs > 0
            or ref_feature_count < ref_history_count
        )

        if ref_sync_needed:
            run_step(
                "3/6 · 🧹 SINCRONIZAR ALIASES + FEATURES DE ÁRBITRO",
                "normalize_laliga_referee_aliases_v1.py",
            )
        else:
            print()
            print("=" * 96)
            print("3/6 · 🧹 ÁRBITROS")
            print("=" * 96)
            print(
                "⏩ Histórico y features arbitrales ya están sincronizados."
            )

        # V11.5 fuerza UNA sincronización tras instalarse para cubrir:
        # - una ejecución anterior cerrada entre guardado y reentrenamiento;
        # - el nuevo parser de tiros a puerta.
        first_v113_sync = not v115_sync_done()

        team_sync_needed = (
            new_matches > 0
            or team_feature_count < finished_count
            or first_v113_sync
        )

        if team_sync_needed:
            run_step(
                "4/6 · 🧠 SINCRONIZAR FEATURES DE EQUIPOS",
                "build_laliga_team_stats.py",
            )

            (
                season_after,
                finished_after,
                team_features_after,
                ref_history_after,
                ref_features_after,
            ) = current_season_counts()

            print()
            print(
                f"✅ Verificación features {season_after}/{season_after+1}: "
                f"{team_features_after}/{finished_after} partidos"
            )

            if team_features_after != finished_after:
                raise RuntimeError(
                    "Las features de la temporada actual no están completas. "
                    f"Esperados={finished_after}, "
                    f"features={team_features_after}. "
                    "Se aborta antes de reentrenar."
                )
        else:
            print()
            print("=" * 96)
            print("4/6 · 🧠 FEATURES DE EQUIPOS")
            print("=" * 96)
            print(
                "⏩ Features de equipos ya sincronizadas con todos los partidos."
            )

        trained_count = trained_match_count()

        model_sync_needed = (
            new_matches > 0
            or trained_count != finished_count
            or first_v113_sync
        )

        # Si FotMob sigue sin darnos SOT, no fingimos que SOT está actualizado.
        if sot_total > 0 and sot_covered == 0:
            raise RuntimeError(
                "FotMob no ha proporcionado tiros a puerta de la temporada actual. "
                "V11.3 no reentrenará modelos con SOT vacío."
            )

        if model_sync_needed:
            run_step(
                "5/6 · 🤖 SINCRONIZAR MODELOS FINALES",
                "retrain_laliga_models_fast_v1.py",
            )
        else:
            print()
            print("=" * 96)
            print("5/6 · 🤖 MODELOS")
            print("=" * 96)
            print(
                f"⏩ Modelos ya entrenados con los {finished_count} "
                "partidos de la temporada actual."
            )

        run_step(
            "6/6 · 📅 PRÓXIMA JORNADA + ÁRBITROS ASIGNADOS",
            "update_laliga_fixtures_v4_referee.py",
        )

        final_summary(
            int(new_matches),
            int(new_refs),
        )

    except KeyboardInterrupt:
        print("\nActualización cancelada.")
        restore_backup(backup)
        raise SystemExit(130)

    except Exception as exc:
        print()
        print("❌ ERROR:")
        print(str(exc))
        restore_backup(backup)
        raise


if __name__ == "__main__":
    main()
