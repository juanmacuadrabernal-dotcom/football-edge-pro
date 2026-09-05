from __future__ import annotations

import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "eliteserien.db"
RAW_MATCH_DIR = BASE_DIR / "data" / "fotmob_raw" / "matches"


def as_dict(value):
    return value if isinstance(value, dict) else {}


def extract_score(payload):
    header = as_dict(payload.get("header"))
    teams = header.get("teams")

    if not isinstance(teams, list) or len(teams) < 2:
        return None, None

    home = as_dict(teams[0])
    away = as_dict(teams[1])

    home_score = home.get("score")
    away_score = away.get("score")

    try:
        home_score = int(home_score) if home_score is not None else None
    except Exception:
        home_score = None

    try:
        away_score = int(away_score) if away_score is not None else None
    except Exception:
        away_score = None

    return home_score, away_score


def extract_status(payload):
    header = as_dict(payload.get("header"))
    status = as_dict(header.get("status"))
    reason = as_dict(status.get("reason"))

    cancelled = 1 if status.get("cancelled") else 0
    finished = 1 if status.get("finished") else 0

    reason_short = reason.get("short")
    reason_long = reason.get("long")
    score_str = status.get("scoreStr")

    return cancelled, finished, reason_short, reason_long, score_str


def ensure_columns(conn):
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(fotmob_matches)").fetchall()
    }

    if "valid_for_model" not in columns:
        conn.execute(
            "ALTER TABLE fotmob_matches ADD COLUMN valid_for_model INTEGER DEFAULT 1"
        )

    if "exclusion_reason" not in columns:
        conn.execute(
            "ALTER TABLE fotmob_matches ADD COLUMN exclusion_reason TEXT"
        )


def main():
    if not DB_PATH.exists():
        print(f"ERROR: no existe la base de datos: {DB_PATH}")
        raise SystemExit(1)

    if not RAW_MATCH_DIR.exists():
        print(f"ERROR: no existe la carpeta de JSON: {RAW_MATCH_DIR}")
        raise SystemExit(1)

    json_files = sorted(RAW_MATCH_DIR.rglob("*.json"))

    print()
    print("=" * 70)
    print("ELITESERIEN EDGE PRO - REPARAR MARCADORES")
    print("=" * 70)
    print(f"JSON encontrados: {len(json_files)}")
    print()

    updated_scores = 0
    excluded = 0
    no_score = 0
    invalid_json = 0

    with sqlite3.connect(DB_PATH) as conn:
        ensure_columns(conn)

        for index, path in enumerate(json_files, start=1):
            try:
                match_id = int(path.stem)
            except ValueError:
                continue

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                invalid_json += 1
                continue

            home_score, away_score = extract_score(payload)
            cancelled, finished, reason_short, reason_long, score_str = extract_status(payload)

            if cancelled:
                valid_for_model = 0
                exclusion_reason = reason_long or reason_short or "Cancelled/Abandoned"
                excluded += 1
            else:
                valid_for_model = 1
                exclusion_reason = None

            if home_score is not None and away_score is not None:
                cur = conn.execute(
                    """
                    UPDATE fotmob_matches
                    SET
                        home_goals = ?,
                        away_goals = ?,
                        finished = ?,
                        cancelled = ?,
                        valid_for_model = ?,
                        exclusion_reason = ?,
                        status_reason = COALESCE(status_reason, ?),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE match_id = ?
                    """,
                    (
                        home_score,
                        away_score,
                        finished,
                        cancelled,
                        valid_for_model,
                        exclusion_reason,
                        reason_long or reason_short,
                        match_id,
                    ),
                )

                if cur.rowcount:
                    updated_scores += 1

            else:
                # Aunque no haya marcador, marcamos igualmente si está cancelado.
                conn.execute(
                    """
                    UPDATE fotmob_matches
                    SET
                        finished = ?,
                        cancelled = ?,
                        valid_for_model = ?,
                        exclusion_reason = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE match_id = ?
                    """,
                    (
                        finished,
                        cancelled,
                        valid_for_model,
                        exclusion_reason,
                        match_id,
                    ),
                )

                if not cancelled:
                    no_score += 1

            if index % 100 == 0:
                print(f"[{index}/{len(json_files)}] revisados...")
                conn.commit()

        conn.commit()

        total_valid_finished = conn.execute(
            """
            SELECT COUNT(*)
            FROM fotmob_matches
            WHERE finished = 1
              AND cancelled = 0
              AND valid_for_model = 1
            """
        ).fetchone()[0]

        valid_with_score = conn.execute(
            """
            SELECT COUNT(*)
            FROM fotmob_matches
            WHERE finished = 1
              AND cancelled = 0
              AND valid_for_model = 1
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
            """
        ).fetchone()[0]

        excluded_rows = conn.execute(
            """
            SELECT match_id, season, home_team, away_team, exclusion_reason
            FROM fotmob_matches
            WHERE valid_for_model = 0
            ORDER BY season, utc_time
            """
        ).fetchall()

    print()
    print("=" * 70)
    print("REPARACION COMPLETADA")
    print("=" * 70)
    print(f"Partidos con marcador actualizado: {updated_scores}")
    print(f"Partidos válidos finalizados: {total_valid_finished}")
    print(f"Partidos válidos con marcador: {valid_with_score}")
    print(f"Partidos sin marcador (no cancelados): {no_score}")
    print(f"JSON inválidos: {invalid_json}")
    print(f"Partidos excluidos del modelo: {len(excluded_rows)}")

    if total_valid_finished:
        pct = valid_with_score / total_valid_finished * 100
        print(f"Cobertura de marcadores: {pct:.2f}%")

    if excluded_rows:
        print()
        print("EXCLUIDOS DEL MODELO:")
        for row in excluded_rows:
            print(
                f"  {row[0]} | {row[1]} | {row[2]} - {row[3]} | {row[4]}"
            )

    print("=" * 70)


if __name__ == "__main__":
    main()
