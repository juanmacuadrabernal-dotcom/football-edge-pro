from __future__ import annotations

import json
import sys
from pathlib import Path
from pprint import pprint

BASE_DIR = Path(__file__).resolve().parent
SEASONS_DIR = BASE_DIR / "data" / "fotmob_raw" / "seasons"

DEFAULT_MATCH_ID = 4583736


def find_objects_with_match_id(node, match_id, found):
    if isinstance(node, dict):
        if node.get("id") == match_id:
            found.append(node)

        for value in node.values():
            find_objects_with_match_id(value, match_id, found)

    elif isinstance(node, list):
        for value in node:
            find_objects_with_match_id(value, match_id, found)


def main():
    match_id = DEFAULT_MATCH_ID

    if len(sys.argv) > 1:
        try:
            match_id = int(sys.argv[1])
        except ValueError:
            print("El ID debe ser numérico.")
            raise SystemExit(1)

    files = sorted(SEASONS_DIR.glob("eliteserien_*.json"))

    if not files:
        print(f"No encuentro JSON de temporadas en: {SEASONS_DIR}")
        raise SystemExit(1)

    print()
    print("=" * 70)
    print("ELITESERIEN EDGE PRO - INSPECCIONAR PARTIDO")
    print("=" * 70)
    print(f"Buscando match_id: {match_id}")
    print()

    found = []

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        local_found = []
        find_objects_with_match_id(payload, match_id, local_found)

        if local_found:
            print(f"Encontrado en: {path.name}")
            found.extend(local_found)

    if not found:
        print("No se encontró el partido.")
        raise SystemExit(1)

    print(f"Objetos encontrados: {len(found)}")
    print()

    for i, obj in enumerate(found, start=1):
        print("-" * 70)
        print(f"OBJETO {i}")
        print("-" * 70)

        # Primero una vista rápida de las claves.
        print("CLAVES:")
        print(sorted(obj.keys()))
        print()

        # Campos más interesantes.
        for key in (
            "id",
            "round",
            "roundName",
            "home",
            "away",
            "status",
            "score",
            "scores",
            "result",
            "winner",
            "time",
            "utcTime",
        ):
            if key in obj:
                print(f"{key}:")
                pprint(obj.get(key), width=120)
                print()

        print("OBJETO COMPLETO:")
        pprint(obj, width=140)
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
