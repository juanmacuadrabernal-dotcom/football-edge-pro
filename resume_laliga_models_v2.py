from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "laliga.db"

STEPS = [
    ("MODELO GOLES V4", "laliga_goals_v4_fixed.py"),
    ("MODELO CORNERS V1", "laliga_corners_v1_fixed.py"),
    ("MODELO TARJETAS V2", "laliga_cards_v2_fixed.py"),
]


def count_rows(conn, table):
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()

    if not exists:
        return 0

    return conn.execute(
        f'SELECT COUNT(*) FROM "{table}"'
    ).fetchone()[0]


def main():
    print()
    print("=" * 80)
    print("LALIGA EDGE PRO - RESUME MODELOS V2")
    print("=" * 80)

    if not DB_PATH.exists():
        raise SystemExit("ERROR: no existe data/laliga.db")

    with sqlite3.connect(DB_PATH) as conn:
        h = count_rows(conn, "team_match_history")
        f = count_rows(conn, "team_pre_match_features")

    print(f"team_match_history: {h}")
    print(f"team_pre_match_features: {f}")

    if h == 0 or f == 0:
        raise SystemExit(
            "ERROR: faltan features. Ejecuta build_laliga_team_stats.py"
        )

    print(
        "Features OK. No se repiten descargas ni estadísticas."
    )

    for i, (title, filename) in enumerate(STEPS, 1):
        print()
        print("=" * 80)
        print(f"{i}/{len(STEPS)} {title}")
        print("=" * 80)

        path = BASE_DIR / filename
        if not path.exists():
            raise SystemExit(f"ERROR: falta {filename}")

        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(BASE_DIR),
        )

        if result.returncode != 0:
            raise SystemExit(
                f"ERROR en {filename}. Proceso detenido."
            )

    print()
    print("=" * 80)
    print("LALIGA - MODELOS PHASE 1 COMPLETADOS")
    print("=" * 80)
    print(
        "Pásame los resúmenes finales de GOLES, CORNERS y TARJETAS."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
