from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


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

STEPS = [
    (
        "📥 Resultados + estadísticas",
        "update_laliga_history_v1.py",
    ),
    (
        "🧠 Features de equipos",
        "build_laliga_team_stats.py",
    ),
    (
        "🧑‍⚖️ Histórico de árbitros",
        "download_laliga_referees_fast_v2.py",
    ),
    (
        "🧹 Aliases + features de árbitros",
        "normalize_laliga_referee_aliases_v1.py",
    ),
    (
        "🤖 Reentrenar modelos finales",
        "retrain_laliga_models_fast_v1.py",
    ),
    (
        "📅 Próxima jornada + árbitros asignados",
        "update_laliga_fixtures_v4_referee.py",
    ),
]


def table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def scalar(conn, sql, default=0):
    try:
        row = conn.execute(sql).fetchone()
        if not row or row[0] is None:
            return default
        return row[0]
    except Exception:
        return default


def create_backup():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = BACKUP_DIR / f"laliga_update_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        shutil.copy2(DB_PATH, folder / "laliga.db")

    for name in MODEL_FILES:
        src = MODELS_DIR / name
        if src.exists():
            shutil.copy2(src, folder / name)

    # Conservar solo las 5 copias más recientes.
    backups = sorted(
        [p for p in BACKUP_DIR.glob("laliga_update_*") if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for old in backups[5:]:
        shutil.rmtree(old, ignore_errors=True)

    return folder


def restore_backup(folder: Path):
    print()
    print("⚠️ Restaurando copia de seguridad por fallo...")

    db_backup = folder / "laliga.db"
    if db_backup.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_backup, DB_PATH)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        src = folder / name
        if src.exists():
            shutil.copy2(src, MODELS_DIR / name)

    print("✅ Copia anterior restaurada.")


def run_step(index, title, script):
    path = BASE_DIR / script
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {script} en la carpeta del proyecto."
        )

    print()
    print("=" * 96)
    print(f"PASO {index}/{len(STEPS)} · {title}")
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


def final_summary():
    print()
    print("=" * 96)
    print("🚀 ACTUALIZACIÓN GENERAL COMPLETADA")
    print("=" * 96)

    if not DB_PATH.exists():
        print("No existe laliga.db")
        return

    with sqlite3.connect(DB_PATH) as conn:
        total = scalar(
            conn,
            "SELECT COUNT(*) FROM fotmob_matches WHERE finished = 1",
        ) if table_exists(conn, "fotmob_matches") else 0

        current_season = scalar(
            conn,
            "SELECT MAX(season) FROM fotmob_matches WHERE finished = 1",
            default=None,
        ) if table_exists(conn, "fotmob_matches") else None

        current_count = 0
        if current_season is not None:
            current_count = scalar(
                conn,
                f"SELECT COUNT(*) FROM fotmob_matches "
                f"WHERE finished = 1 AND season = {int(current_season)}",
            )

        refs = scalar(
            conn,
            "SELECT COUNT(*) FROM referee_match_history WHERE referee IS NOT NULL",
        ) if table_exists(conn, "referee_match_history") else 0

        ref_names = scalar(
            conn,
            "SELECT COUNT(DISTINCT referee) FROM referee_match_history WHERE referee IS NOT NULL",
        ) if table_exists(conn, "referee_match_history") else 0

        upcoming = scalar(
            conn,
            "SELECT COUNT(*) FROM laliga_upcoming_fixtures",
        ) if table_exists(conn, "laliga_upcoming_fixtures") else 0

        assigned = scalar(
            conn,
            "SELECT COUNT(*) FROM laliga_upcoming_fixtures "
            "WHERE referee IS NOT NULL AND TRIM(referee) <> ''",
        ) if table_exists(conn, "laliga_upcoming_fixtures") else 0

        team_rows = scalar(
            conn,
            "SELECT COUNT(*) FROM team_match_history",
        ) if table_exists(conn, "team_match_history") else 0

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
        conn.execute(
            """
            INSERT INTO system_status(key, value, updated_at)
            VALUES('laliga_last_full_update', ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at
            """,
            (now, now),
        )
        conn.commit()

    print(f"✅ Histórico terminado: {total} partidos")
    if current_season is not None:
        print(
            f"✅ Temporada {int(current_season)}/{int(current_season)+1}: "
            f"{current_count} partidos terminados"
        )
    print(f"✅ Filas equipo-partido: {team_rows}")
    print(f"✅ Árbitros: {refs} partidos · {ref_names} nombres normalizados")
    print(f"✅ Próximos partidos: {upcoming} · árbitro asignado: {assigned}/{upcoming}")

    print()
    print("MODELOS ACTIVOS:")
    for name in MODEL_FILES:
        path = MODELS_DIR / name
        print(f"  {'✅' if path.exists() else '❌'} {name}")

    print()
    print("Ya puedes volver a la app. No necesitas ejecutar ningún otro .py.")
    print("=" * 96)


def main():
    print()
    print("#" * 96)
    print("FOOTBALL EDGE PRO · LALIGA · ACTUALIZAR TODO")
    print("#" * 96)
    print(
        "Resultados → stats → features equipos → árbitros → "
        "features árbitro → modelos → próxima jornada"
    )

    backup = create_backup()
    print(f"\n🛡️ Backup creado: {backup}")

    try:
        for i, (title, script) in enumerate(STEPS, start=1):
            run_step(i, title, script)

        final_summary()

    except KeyboardInterrupt:
        print("\nActualización interrumpida por el usuario.")
        restore_backup(backup)
        raise SystemExit(130)

    except Exception as exc:
        print()
        print("❌ ERROR EN LA ACTUALIZACIÓN:")
        print(str(exc))
        restore_backup(backup)
        raise


if __name__ == "__main__":
    main()
