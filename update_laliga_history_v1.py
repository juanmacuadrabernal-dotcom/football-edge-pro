from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from setup_laliga_data_v2 import (
    DATA_DIR,
    CACHE_DIR,
    DB_PATH,
    download_history,
    prepare_history,
    build_tables,
)


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def existing_match_ids(conn: sqlite3.Connection) -> set[int]:
    if not table_exists(conn, "fotmob_matches"):
        return set()

    try:
        df = pd.read_sql_query(
            "SELECT match_id FROM fotmob_matches WHERE finished = 1",
            conn,
        )
    except Exception:
        return set()

    return set(pd.to_numeric(df["match_id"], errors="coerce").dropna().astype(int))


def main():
    print()
    print("=" * 88)
    print("FOOTBALL EDGE PRO - ACTUALIZAR HISTORICO LALIGA")
    print("=" * 88)
    print("Fuente: Football-Data SP1. Se preservan árbitros, fixtures y backtests.")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        before_ids = existing_match_ids(conn)

    print(f"Partidos terminados antes: {len(before_ids)}")
    print("Descargando temporadas 2020/21 -> 2026/27...")

    raw = download_history()
    hist = prepare_history(raw)
    matches, stats, _refs_unused = build_tables(hist)

    after_ids = set(
        pd.to_numeric(matches["match_id"], errors="coerce")
        .dropna()
        .astype(int)
    )

    new_ids = after_ids - before_ids
    removed_ids = before_ids - after_ids

    # IMPORTANTE: solo reemplazamos las tablas de resultados/estadísticas.
    # referee_match_history, referee_pre_match_features, laliga_upcoming_fixtures
    # y los backtests NO se tocan aquí.
    with sqlite3.connect(DB_PATH) as conn:
        raw.to_sql(
            "football_data_raw",
            conn,
            if_exists="replace",
            index=False,
        )

        matches.to_sql(
            "matches",
            conn,
            if_exists="replace",
            index=False,
        )

        matches.to_sql(
            "fotmob_matches",
            conn,
            if_exists="replace",
            index=False,
        )

        stats.to_sql(
            "fotmob_match_stats",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_laliga_matches_time
            ON fotmob_matches(utc_time);

            CREATE INDEX IF NOT EXISTS idx_laliga_matches_season
            ON fotmob_matches(season);

            CREATE INDEX IF NOT EXISTS idx_laliga_stats_match
            ON fotmob_match_stats(match_id);
            """
        )
        conn.commit()

    current_season = int(matches["season"].max())
    current = matches[matches["season"] == current_season]
    current_stats = stats[stats["match_id"].isin(current["match_id"])]

    print()
    print("=" * 88)
    print("HISTORICO ACTUALIZADO")
    print("=" * 88)
    print(f"Partidos terminados ahora: {len(matches)}")
    print(f"Partidos nuevos detectados: {len(new_ids)}")
    print(f"Temporada actual: {current_season}/{current_season + 1}")
    print(f"Partidos temporada actual: {len(current)}")
    print(
        "Con tiros a puerta: "
        f"{int(current_stats['home_shots_on_target'].notna().sum())}/{len(current_stats)}"
    )
    print(
        "Con córners: "
        f"{int(current_stats['home_corners'].notna().sum())}/{len(current_stats)}"
    )
    print(
        "Con amarillas: "
        f"{int(current_stats['home_yellow_cards'].notna().sum())}/{len(current_stats)}"
    )
    print(
        "Con faltas: "
        f"{int(current_stats['home_fouls'].notna().sum())}/{len(current_stats)}"
    )

    if removed_ids:
        print(
            "AVISO: hay IDs históricos que ya no aparecen en la descarga: "
            f"{len(removed_ids)}. Puede deberse a correcciones de fecha/formato."
        )

    if len(new_ids) == 0:
        print(
            "No hay partidos nuevos todavía. Si la jornada acaba de terminar, "
            "Football-Data puede tardar un poco en publicar el CSV actualizado."
        )

    print("=" * 88)


if __name__ == "__main__":
    main()
