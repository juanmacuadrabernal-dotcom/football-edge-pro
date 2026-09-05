from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

import download_laliga_referees_fast_v2 as src


DB_PATH = src.DB_PATH
MAX_WORKERS = 4


def table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def current_season_start():
    now = pd.Timestamp.now(tz="Europe/Madrid")
    return int(now.year if now.month >= 7 else now.year - 1)


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


def main():
    print()
    print("=" * 92)
    print("FOOTBALL EDGE PRO - ÁRBITROS INCREMENTALES LALIGA")
    print("=" * 92)

    season = current_season_start()

    print(f"Temporada consultada: {season}/{season+1}")
    print("Solo se buscan partidos finalizados NUEVOS de esta temporada.")
    print()

    label, fixtures = src.finished_matches_for_season(season)

    with sqlite3.connect(DB_PATH) as conn:
        if table_exists(conn, "referee_match_history"):
            existing = pd.read_sql_query(
                "SELECT * FROM referee_match_history",
                conn,
            )
        else:
            existing = pd.DataFrame()

    existing_ids = set()

    if (
        not existing.empty
        and "match_id" in existing.columns
    ):
        existing_ids = set(
            pd.to_numeric(
                existing["match_id"],
                errors="coerce",
            ).dropna().astype(int)
        )

    missing = [
        f
        for f in fixtures
        if int(f["id"]) not in existing_ids
    ]

    print(
        f"FotMob finalizados temporada: {len(fixtures)} | "
        f"ya guardados: {len(fixtures)-len(missing)} | "
        f"nuevos: {len(missing)}"
    )

    new_rows = []
    new_downloads = 0
    cache_hits = 0
    errors = 0

    if missing:
        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            futures = {
                executor.submit(
                    src.process_one,
                    season,
                    fixture,
                ): fixture
                for fixture in missing
            }

            done = 0

            for future in as_completed(futures):
                done += 1

                try:
                    row, downloaded = future.result()
                    row["referee_raw"] = row.get("referee")
                    new_rows.append(row)

                    if downloaded:
                        new_downloads += 1
                    else:
                        cache_hits += 1

                except Exception as exc:
                    errors += 1
                    fixture = futures[future]
                    print(
                        f"ERROR match {fixture.get('id')}: {exc}"
                    )

                if done % 5 == 0 or done == len(futures):
                    print(
                        f"  [{done}/{len(futures)}] "
                        f"ok={len(new_rows)} | "
                        f"descargas={new_downloads} | "
                        f"cache={cache_hits} | "
                        f"errores={errors}"
                    )

    if new_rows:
        new_df = pd.DataFrame(new_rows)

        if existing.empty:
            merged = new_df.copy()
        else:
            cols = list(
                dict.fromkeys(
                    list(existing.columns)
                    + list(new_df.columns)
                )
            )
            merged = pd.concat(
                [
                    existing.reindex(columns=cols),
                    new_df.reindex(columns=cols),
                ],
                ignore_index=True,
                sort=False,
            )

        merged = (
            merged.sort_values(
                ["season", "utc_time", "match_id"]
            )
            .drop_duplicates(
                subset=["match_id"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        summary = src.build_summary(merged)

        with sqlite3.connect(DB_PATH) as conn:
            merged.to_sql(
                "referee_match_history",
                conn,
                if_exists="replace",
                index=False,
            )

            if not summary.empty:
                summary.to_sql(
                    "referee_summary",
                    conn,
                    if_exists="replace",
                    index=False,
                )

            save_status(
                conn,
                "laliga_last_new_referee_matches",
                len(new_rows),
            )
            conn.commit()

    else:
        with sqlite3.connect(DB_PATH) as conn:
            save_status(
                conn,
                "laliga_last_new_referee_matches",
                0,
            )
            conn.commit()

    print()
    print("=" * 92)
    print("ÁRBITROS INCREMENTALES COMPLETADOS")
    print("=" * 92)
    print(f"Partidos nuevos de árbitro: {len(new_rows)}")
    print(f"Descargas nuevas: {new_downloads}")
    print(f"Leídos desde caché: {cache_hits}")
    print(f"Errores: {errors}")
    print("=" * 92)

    if errors:
        raise RuntimeError(
            f"Hubo {errors} errores descargando árbitros nuevos."
        )


if __name__ == "__main__":
    main()
