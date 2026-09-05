from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from setup_laliga_data_v2 import (
    CACHE_DIR,
    DB_PATH,
    SP1_URL,
    download_csv,
    prepare_history,
    build_tables,
)


def table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def current_season_start():
    now = pd.Timestamp.now(tz="Europe/Madrid")
    return int(now.year if now.month >= 7 else now.year - 1)


def football_data_folder(season):
    # Football-Data usa 2122, 2223, ... para temporadas modernas.
    if season == 2020:
        return "2021"
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def read_table(conn, name):
    if not table_exists(conn, name):
        return pd.DataFrame()
    return pd.read_sql_query(f"SELECT * FROM {name}", conn)


def save_status(conn, key, value):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_status (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
        """
    )
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO system_status(key, value, updated_at)
        VALUES(?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
        """,
        (key, str(value), now),
    )


def merge_current_season(existing, current, season, season_col="season"):
    if existing.empty:
        return current.copy()

    if season_col not in existing.columns:
        return current.copy()

    old = existing[
        pd.to_numeric(existing[season_col], errors="coerce") != int(season)
    ].copy()

    # Alinear columnas sin perder columnas históricas.
    cols = list(dict.fromkeys(list(old.columns) + list(current.columns)))
    old = old.reindex(columns=cols)
    current = current.reindex(columns=cols)

    return pd.concat(
        [old, current],
        ignore_index=True,
        sort=False,
    )


def main():
    print()
    print("=" * 92)
    print("FOOTBALL EDGE PRO - ACTUALIZACIÓN INCREMENTAL LALIGA")
    print("=" * 92)

    season = current_season_start()
    folder = football_data_folder(season)

    print(f"Temporada actual: {season}/{season+1}")
    print("Solo se consulta ESTA temporada. No se recorre 2020+.")
    print()

    cache = CACHE_DIR / f"SP1_{season}_{season+1}.csv"
    url = SP1_URL.format(folder=folder)

    raw_current = download_csv(url, cache)
    raw_current["season"] = season

    hist_current = prepare_history(raw_current)
    matches_current, stats_current, _ = build_tables(hist_current)

    current_ids = set(
        pd.to_numeric(
            matches_current["match_id"],
            errors="coerce",
        ).dropna().astype(int)
    )

    with sqlite3.connect(DB_PATH) as conn:
        old_matches = read_table(conn, "fotmob_matches")
        old_matches_alias = read_table(conn, "matches")
        old_stats = read_table(conn, "fotmob_match_stats")
        old_raw = read_table(conn, "football_data_raw")

        if not old_matches.empty and "season" in old_matches.columns:
            old_current = old_matches[
                pd.to_numeric(
                    old_matches["season"],
                    errors="coerce",
                ) == season
            ].copy()
        else:
            old_current = pd.DataFrame()

        old_current_ids = set(
            pd.to_numeric(
                old_current.get(
                    "match_id",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).dropna().astype(int)
        )

        new_ids = current_ids - old_current_ids
        corrected_ids = old_current_ids.intersection(current_ids)

        # Guardar histórico raw: sustituir SOLO temporada actual.
        raw_merged = merge_current_season(
            old_raw,
            raw_current,
            season,
            "season",
        )

        matches_merged = merge_current_season(
            old_matches,
            matches_current,
            season,
            "season",
        )

        matches_alias_merged = merge_current_season(
            old_matches_alias,
            matches_current,
            season,
            "season",
        )

        # Stats no tiene season: borrar solo IDs que pertenecían a la
        # temporada actual y añadir la versión actual.
        if old_stats.empty:
            stats_merged = stats_current.copy()
        else:
            if "match_id" not in old_stats.columns:
                stats_merged = stats_current.copy()
            else:
                mask = ~pd.to_numeric(
                    old_stats["match_id"],
                    errors="coerce",
                ).isin(old_current_ids)

                old_stats_kept = old_stats[mask].copy()
                cols = list(
                    dict.fromkeys(
                        list(old_stats_kept.columns)
                        + list(stats_current.columns)
                    )
                )
                stats_merged = pd.concat(
                    [
                        old_stats_kept.reindex(columns=cols),
                        stats_current.reindex(columns=cols),
                    ],
                    ignore_index=True,
                    sort=False,
                )

        raw_merged.to_sql(
            "football_data_raw",
            conn,
            if_exists="replace",
            index=False,
        )
        matches_alias_merged.to_sql(
            "matches",
            conn,
            if_exists="replace",
            index=False,
        )
        matches_merged.to_sql(
            "fotmob_matches",
            conn,
            if_exists="replace",
            index=False,
        )
        stats_merged.to_sql(
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

        save_status(
            conn,
            "laliga_last_new_matches",
            len(new_ids),
        )
        save_status(
            conn,
            "laliga_current_season",
            season,
        )
        save_status(
            conn,
            "laliga_current_season_matches",
            len(matches_current),
        )
        conn.commit()

    print("=" * 92)
    print("ACTUALIZACIÓN INCREMENTAL COMPLETADA")
    print("=" * 92)
    print(f"Partidos temporada actual en Football-Data: {len(matches_current)}")
    print(f"Partidos nuevos detectados: {len(new_ids)}")
    print(f"Partidos ya existentes refrescados: {len(corrected_ids)}")

    if new_ids:
        print("IDs nuevos incorporados:")
        for mid in sorted(new_ids):
            row = matches_current[
                pd.to_numeric(
                    matches_current["match_id"],
                    errors="coerce",
                ) == mid
            ]
            if not row.empty:
                r = row.iloc[0]
                print(
                    f"  + {r['date']} · "
                    f"{r['home_team']} {int(r['home_goals'])}-"
                    f"{int(r['away_goals'])} {r['away_team']}"
                )
    else:
        print(
            "No hay resultados nuevos. El resto del histórico NO se ha tocado."
        )

    print("=" * 92)


if __name__ == "__main__":
    main()
