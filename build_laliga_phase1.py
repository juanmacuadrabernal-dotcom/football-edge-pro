from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    ("DATOS LALIGA", "setup_laliga_data.py"),
    ("FEATURES EQUIPOS", "build_laliga_team_stats.py"),
    ("MODELO GOLES V4", "laliga_goals_v4.py"),
    ("MODELO CORNERS V1", "laliga_corners_v1.py"),
    ("MODELO TARJETAS V2", "laliga_cards_v2.py"),
]


def main():
    print()
    print("=" * 80)
    print("LALIGA EDGE PRO - BUILD COMPLETO PHASE 1")
    print("=" * 80)

    for i, (name, filename) in enumerate(STEPS, 1):
        path = BASE_DIR / filename

        print()
        print("=" * 80)
        print(f"{i}/{len(STEPS)} {name}")
        print("=" * 80)

        if not path.exists():
            raise FileNotFoundError(
                f"Falta {filename} en {BASE_DIR}"
            )

        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(BASE_DIR),
        )

        if result.returncode != 0:
            print()
            print(
                f"ERROR en {filename}. "
                "No continúo para no entrenar sobre datos incompletos."
            )
            raise SystemExit(result.returncode)

    print()
    print("=" * 80)
    print("LALIGA PHASE 1 COMPLETADA")
    print("=" * 80)
    print(
        "Ya tienes datos, features y modelos separados "
        "de los de Noruega."
    )
    print(
        "Ahora abre la app multiliga con: "
        "streamlit run app_v4_multiliga.py"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
