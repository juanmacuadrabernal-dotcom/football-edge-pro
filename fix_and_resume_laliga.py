from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "laliga.db"

STEPS = [
    ("FEATURES DE EQUIPOS", "build_laliga_team_stats.py"),
    ("MODELO GOLES V4", "laliga_goals_v4.py"),
    ("MODELO CORNERS V1", "laliga_corners_v1.py"),
    ("MODELO TARJETAS V2", "laliga_cards_v2_fixed.py"),
]


def table_count(conn, table):
    exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table,),
    ).fetchone()

    if not exists:
        return None

    return conn.execute(
        f'SELECT COUNT(*) FROM "{table}"'
    ).fetchone()[0]


def ensure_referee_summary_schema(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS referee_summary (
            referee TEXT,
            matches INTEGER,
            avg_yellow REAL,
            avg_home_yellow REAL,
            avg_away_yellow REAL,
            avg_red REAL,
            avg_fouls REAL,
            first_season INTEGER,
            last_season INTEGER
        )
        """
    )
    conn.commit()


def main():
    print()
    print("=" * 80)
    print("LALIGA EDGE PRO - FIX + RESUME PHASE 1")
    print("=" * 80)

    if not DB_PATH.exists():
        print()
        print("No encuentro data/laliga.db.")
        print(
            "En ese caso sustituye setup_laliga_data.py por "
            "setup_laliga_data_v2.py y vuelve a ejecutar el setup."
        )
        raise SystemExit(1)

    with sqlite3.connect(DB_PATH) as conn:
        matches = table_count(conn, "fotmob_matches")
        stats = table_count(conn, "fotmob_match_stats")
        refs = table_count(conn, "referee_match_history")

        print()
        print("Base encontrada:")
        print(f"  fotmob_matches: {matches}")
        print(f"  fotmob_match_stats: {stats}")
        print(f"  referee_match_history: {refs}")

        if not matches or not stats:
            print()
            print(
                "La base quedó incompleta antes del error. "
                "Ejecuta: python setup_laliga_data_v2.py"
            )
            raise SystemExit(2)

        ensure_referee_summary_schema(conn)

    print()
    print(
        "El error de SQLite queda reparado. "
        "No vamos a descargar otra vez los 2311 partidos."
    )

    if refs == 0:
        print(
            "Árbitros: Football-Data no aportó nombres en estos CSV. "
            "Phase 1 entrenará tarjetas sin árbitro; "
            "después lo enriqueceremos con FotMob."
        )

    for i, (title, filename) in enumerate(STEPS, 1):
        path = BASE_DIR / filename

        if not path.exists():
            print()
            print(f"Falta {filename}")
            raise SystemExit(3)

        print()
        print("=" * 80)
        print(f"{i}/{len(STEPS)} {title}")
        print("=" * 80)

        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(BASE_DIR),
        )

        if result.returncode != 0:
            print()
            print(
                f"ERROR en {filename}. "
                "He parado aquí para no contaminar los siguientes modelos."
            )
            raise SystemExit(result.returncode)

    print()
    print("=" * 80)
    print("LALIGA PHASE 1 COMPLETADA")
    print("=" * 80)
    print(
        "Datos + features + goles + corners + tarjetas generados."
    )
    print(
        "El siguiente paso será decidir qué mercados españoles "
        "pasan el backtest y enriquecer árbitros/fixtures con FotMob."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
