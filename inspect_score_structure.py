from __future__ import annotations

import json
from pathlib import Path
from pprint import pprint

BASE_DIR = Path(__file__).resolve().parent
MATCH_PATH = (
    BASE_DIR
    / "data"
    / "fotmob_raw"
    / "matches"
    / "2024"
    / "4583736.json"
)


def walk(node, path="$", results=None):
    if results is None:
        results = []

    keywords = (
        "score",
        "goal",
        "home",
        "away",
        "status",
        "result",
        "header",
        "matchfacts",
    )

    if isinstance(node, dict):
        for key, value in node.items():
            key_lower = str(key).lower()

            if any(word in key_lower for word in keywords):
                results.append((f"{path}.{key}", value))

            walk(value, f"{path}.{key}", results)

    elif isinstance(node, list):
        for i, value in enumerate(node):
            walk(value, f"{path}[{i}]", results)

    return results


def main():
    if not MATCH_PATH.exists():
        print("No existe:")
        print(MATCH_PATH)
        raise SystemExit(1)

    payload = json.loads(MATCH_PATH.read_text(encoding="utf-8"))

    print()
    print("=" * 72)
    print("ELITESERIEN EDGE PRO - INSPECCION MARCADOR")
    print("=" * 72)
    print("Archivo:")
    print(MATCH_PATH)
    print()

    print("CLAVES PRINCIPALES:")
    print(sorted(payload.keys()))
    print()

    for key in ("header", "general", "content"):
        if key in payload:
            print("-" * 72)
            print(f"{key.upper()}:")
            pprint(payload[key], width=140, depth=4)
            print()

    print("=" * 72)
    print("CAMPOS RELACIONADOS CON MARCADOR / EQUIPOS")
    print("=" * 72)

    results = walk(payload)

    shown = 0

    for path, value in results:
        text = repr(value)

        # Evita inundar el CMD con estructuras enormes.
        if len(text) > 1000:
            text = text[:1000] + " ..."

        print(path)
        print(text)
        print()

        shown += 1

        if shown >= 80:
            print("... salida limitada a 80 coincidencias ...")
            break


if __name__ == "__main__":
    main()
