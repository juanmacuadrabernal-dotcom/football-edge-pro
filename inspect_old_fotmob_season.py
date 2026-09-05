from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Any

import requests


SEASON = 2022
LEAGUE_ID = 59
COUNTRY_CODE = "NOR"

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "data" / f"fotmob_debug_{SEASON}.json"

URL = "https://www.fotmob.com/api/data/leagues"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.fotmob.com/",
}


def walk(node: Any, path: str = "$", results=None):
    if results is None:
        results = {
            "match_like": [],
            "lists": [],
            "dict_keys": Counter(),
        }

    if isinstance(node, dict):
        keys = tuple(sorted(node.keys()))
        results["dict_keys"][keys] += 1

        # Candidatos con señales de que representan un partido.
        score = 0

        for key in (
            "id",
            "matchId",
            "matchUrl",
            "home",
            "away",
            "homeTeam",
            "awayTeam",
            "status",
            "time",
            "utcTime",
            "round",
        ):
            if key in node:
                score += 1

        if score >= 3:
            results["match_like"].append((path, node))

        for key, value in node.items():
            walk(value, f"{path}.{key}", results)

    elif isinstance(node, list):
        if node:
            results["lists"].append(
                (
                    path,
                    len(node),
                    type(node[0]).__name__,
                    list(node[0].keys())[:20]
                    if isinstance(node[0], dict)
                    else None,
                )
            )

        for i, value in enumerate(node):
            walk(value, f"{path}[{i}]", results)

    return results


def compact(obj: dict) -> dict:
    interesting = {}

    for key in (
        "id",
        "matchId",
        "matchUrl",
        "home",
        "away",
        "homeTeam",
        "awayTeam",
        "status",
        "time",
        "utcTime",
        "round",
        "roundName",
        "league",
        "tournament",
        "finished",
    ):
        if key in obj:
            interesting[key] = obj.get(key)

    return interesting


def main():
    print()
    print("=" * 72)
    print("ELITESERIEN EDGE PRO - INSPECCION TEMPORADA ANTIGUA")
    print("=" * 72)
    print(f"Temporada: {SEASON}")
    print()

    response = requests.get(
        URL,
        params={
            "id": LEAGUE_ID,
            "ccode3": COUNTRY_CODE,
            "season": str(SEASON),
        },
        headers=HEADERS,
        timeout=25,
    )

    print("STATUS:", response.status_code)
    print("Content-Type:", response.headers.get("content-type"))

    if response.status_code != 200:
        print(response.text[:1000])
        raise SystemExit(1)

    payload = response.json()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Guardado en:")
    print(OUT_PATH)
    print()

    print("CLAVES PRINCIPALES:")
    print(sorted(payload.keys()))
    print()

    for key in payload.keys():
        value = payload[key]
        print(
            f"{key:<25} "
            f"{type(value).__name__:<10} "
            f"{len(value) if hasattr(value, '__len__') else ''}"
        )

    results = walk(payload)

    print()
    print("=" * 72)
    print("LISTAS GRANDES / POSIBLES CONTENEDORES DE PARTIDOS")
    print("=" * 72)

    lists_sorted = sorted(
        results["lists"],
        key=lambda x: x[1],
        reverse=True,
    )

    for path, length, first_type, first_keys in lists_sorted[:30]:
        print()
        print("Ruta:", path)
        print("Elementos:", length)
        print("Primer tipo:", first_type)

        if first_keys:
            print("Claves primer elemento:", first_keys)

    print()
    print("=" * 72)
    print("OBJETOS CON ASPECTO DE PARTIDO")
    print("=" * 72)
    print("Candidatos encontrados:", len(results["match_like"]))
    print()

    shown = 0
    seen = set()

    for path, obj in results["match_like"]:
        preview = compact(obj)
        signature = json.dumps(preview, sort_keys=True, default=str)

        if signature in seen:
            continue

        seen.add(signature)

        print("-" * 72)
        print("Ruta:", path)
        pprint(preview, width=130, depth=4)

        shown += 1

        if shown >= 15:
            break

    print()
    print("=" * 72)
    print("FIN DE INSPECCION")
    print("=" * 72)


if __name__ == "__main__":
    main()
