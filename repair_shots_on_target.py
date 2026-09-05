from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_MATCH_DIR = DATA_DIR / "fotmob_raw" / "matches"
DB_PATH = DATA_DIR / "eliteserien.db"


def scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def normalize_name(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = (
        s.replace("(", " ")
        .replace(")", " ")
        .replace("%", " ")
        .replace("-", " ")
        .replace("/", " ")
    )
    s = re.sub(r"[^a-z0-9áéíóúüøæå ]+", " ", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s


def numeric(v: Any) -> float | None:
    if v is None:
        return None

    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)

    text = str(v).strip().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)

    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def collect_stat_pairs(node: Any, out: list[tuple[str, Any, Any]]) -> None:
    if isinstance(node, dict):
        values = node.get("stats")
        name = node.get("key") or node.get("title") or node.get("name")

        if (
            name
            and isinstance(values, list)
            and len(values) >= 2
            and scalar(values[0])
            and scalar(values[1])
        ):
            out.append((normalize_name(name), values[0], values[1]))

        for value in node.values():
            collect_stat_pairs(value, out)

    elif isinstance(node, list):
        for value in node:
            collect_stat_pairs(value, out)


def extract_shots_on_target(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    periods = (
        (payload.get("content") or {})
        .get("stats", {})
        .get("Periods", {})
        .get("All", {})
    )

    pairs: list[tuple[str, Any, Any]] = []
    collect_stat_pairs(periods, pairs)

    aliases = {
        "shotsontarget",
        "shots_on_target",
        "shots_on_goal",
        "ontarget_scoring_att",
    }

    for name, home, away in pairs:
        if name in aliases:
            return numeric(home), numeric(away)

    return None, None


def main() -> None:
    if not DB_PATH.exists():
        print(f"ERROR: no existe la base de datos: {DB_PATH}")
        raise SystemExit(1)

    if not RAW_MATCH_DIR.exists():
        print(f"ERROR: no existe la carpeta de JSON: {RAW_MATCH_DIR}")
        raise SystemExit(1)

    json_files = sorted(RAW_MATCH_DIR.rglob("*.json"))

    print()
    print("=" * 68)
    print("ELITESERIEN EDGE PRO - REPARAR TIROS A PUERTA")
    print("=" * 68)
    print(f"JSON encontrados: {len(json_files)}")
    print()

    updated = 0
    without_stat = 0
    invalid = 0

    with sqlite3.connect(DB_PATH) as conn:
        for index, path in enumerate(json_files, start=1):
            try:
                match_id = int(path.stem)
            except ValueError:
                invalid += 1
                continue

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                invalid += 1
                continue

            home, away = extract_shots_on_target(payload)

            if home is None and away is None:
                without_stat += 1
                continue

            cur = conn.execute(
                """
                UPDATE fotmob_match_stats
                SET
                    home_shots_on_target = ?,
                    away_shots_on_target = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE match_id = ?
                """,
                (home, away, match_id),
            )

            if cur.rowcount:
                updated += 1

            if index % 100 == 0:
                print(f"[{index}/{len(json_files)}] revisados...")
                conn.commit()

        conn.commit()

        coverage = conn.execute(
            """
            SELECT COUNT(*)
            FROM fotmob_match_stats
            WHERE home_shots_on_target IS NOT NULL
              AND away_shots_on_target IS NOT NULL
            """
        ).fetchone()[0]

        total = conn.execute(
            "SELECT COUNT(*) FROM fotmob_match_stats"
        ).fetchone()[0]

    print()
    print("=" * 68)
    print("REPARACION COMPLETADA")
    print("=" * 68)
    print(f"Partidos actualizados: {updated}")
    print(f"Partidos sin shotsontarget en JSON: {without_stat}")
    print(f"JSON inválidos/ignorados: {invalid}")
    print(f"Cobertura tiros a puerta: {coverage}/{total}")
    if total:
        print(f"Cobertura: {coverage / total * 100:.2f}%")
    print("=" * 68)


if __name__ == "__main__":
    main()
