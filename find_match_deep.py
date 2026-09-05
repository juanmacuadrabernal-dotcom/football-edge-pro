from __future__ import annotations

import json
import sys
from pathlib import Path
from pprint import pprint

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "fotmob_raw"
DEFAULT_MATCH_ID = 4583736


def walk(node, match_id, path="$", found=None):
    if found is None:
        found = []

    if isinstance(node, dict):
        # Buscar cualquier campo que contenga el ID.
        for key, value in node.items():
            if value == match_id or str(value) == str(match_id):
                found.append((path, node))
                break

        for key, value in node.items():
            walk(value, match_id, f"{path}.{key}", found)

    elif isinstance(node, list):
        for i, value in enumerate(node):
            walk(value, match_id, f"{path}[{i}]", found)

    return found


def main():
    match_id = DEFAULT_MATCH_ID

    if len(sys.argv) > 1:
        try:
            match_id = int(sys.argv[1])
        except ValueError:
            print("El ID debe ser numérico.")
            raise SystemExit(1)

    if not RAW_DIR.exists():
        print(f"No existe la carpeta: {RAW_DIR}")
        raise SystemExit(1)

    files = sorted(RAW_DIR.rglob("*.json"))

    print()
    print("=" * 72)
    print("ELITESERIEN EDGE PRO - BUSCADOR PROFUNDO DE PARTIDO")
    print("=" * 72)
    print(f"Buscando ID: {match_id}")
    print(f"JSON disponibles: {len(files)}")
    print()

    text_hits = []
    structured_hits = []

    needle = str(match_id)

    for idx, path in enumerate(files, start=1):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue

        if needle not in text:
            continue

        text_hits.append(path)

        try:
            payload = json.loads(text)
        except Exception:
            continue

        found = walk(payload, match_id)

        for json_path, obj in found:
            structured_hits.append((path, json_path, obj))

    print(f"Archivos que contienen el ID como texto: {len(text_hits)}")

    for path in text_hits:
        print(" -", path.relative_to(BASE_DIR))

    print()
    print(f"Coincidencias estructuradas: {len(structured_hits)}")
    print()

    if not text_hits:
        print("El ID no aparece en ningún JSON guardado.")
        print("Eso significaría que el calendario fue parseado desde otra estructura")
        print("pero ese objeto concreto no quedó guardado en caché.")
        raise SystemExit(0)

    if not structured_hits:
        print("El ID aparece como texto, pero no como valor JSON simple.")
        print("Revisaremos el archivo manualmente.")
        raise SystemExit(0)

    # Mostrar solo hasta 5 coincidencias para no inundar el CMD.
    for n, (file_path, json_path, obj) in enumerate(structured_hits[:5], start=1):
        print("-" * 72)
        print(f"COINCIDENCIA {n}")
        print("Archivo:", file_path.relative_to(BASE_DIR))
        print("Ruta JSON:", json_path)
        print()
        print("CLAVES DEL OBJETO:")
        print(sorted(obj.keys()))
        print()

        interesting = (
            "id", "matchId", "match_id",
            "home", "away",
            "homeScore", "awayScore",
            "score", "scores",
            "status",
            "round", "roundName",
            "utcTime", "time",
            "result",
        )

        for key in interesting:
            if key in obj:
                print(f"{key}:")
                pprint(obj.get(key), width=120)
                print()

        print("OBJETO COMPLETO:")
        pprint(obj, width=140)
        print()

    print("=" * 72)


if __name__ == "__main__":
    main()
