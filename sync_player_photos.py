# ============================================================
# TENNIS EDGE PRO · SYNC ALL OFFICIAL ATP PHOTOS
# ------------------------------------------------------------
# Recorre todos los jugadores ATP + Challenger existentes
# en tennis_edge.db.
#
# Es incremental:
# - "found" y "missing" ya revisados se saltan.
# - puedes cerrar y continuar más tarde.
# ============================================================

import argparse
import sqlite3
import time

from player_photos import (
    DB_PATH,
    init_player_photos_table,
    download_atp_id_index,
    get_player_photo_record,
    resolve_player_photo,
)


def get_all_players():
    with sqlite3.connect(
        DB_PATH
    ) as con:
        rows = con.execute(
            """
            SELECT player_name
            FROM (
                SELECT winner_name AS player_name
                FROM matches
                WHERE winner_name IS NOT NULL

                UNION

                SELECT loser_name AS player_name
                FROM matches
                WHERE loser_name IS NOT NULL
            )
            WHERE TRIM(player_name) <> ''
            ORDER BY player_name
            """
        ).fetchall()

    return [
        str(
            row[0]
        ).strip()
        for row in rows
        if row
        and row[0]
    ]


def get_counts():
    with sqlite3.connect(
        DB_PATH
    ) as con:
        rows = con.execute(
            """
            SELECT status, COUNT(*)
            FROM player_photos
            GROUP BY status
            """
        ).fetchall()

    return {
        status: count
        for status, count
        in rows
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help=(
            "Reintenta también jugadores "
            "marcados como missing."
        ),
    )

    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help=(
            "Vuelve a descargar el índice "
            "Wikidata ATP IDs."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.12,
        help=(
            "Pausa entre jugadores nuevos. "
            "Default: 0.12s."
        ),
    )

    args = parser.parse_args()

    init_player_photos_table()

    players = get_all_players()

    print("=" * 76)
    print(
        "TENNIS EDGE PRO · FOTOS OFICIALES ATP + CHALLENGER"
    )
    print("=" * 76)
    print(
        "Base:",
        DB_PATH,
    )
    print(
        "Jugadores únicos:",
        len(players),
    )
    print()
    print(
        "Descargando/abriendo índice de ATP Player IDs..."
    )

    id_index = download_atp_id_index(
        refresh=bool(
            args.refresh_index
        )
    )

    print(
        "IDs indexados:",
        len(id_index),
    )
    print()

    processed = 0
    found_now = 0
    missing_now = 0
    errors = 0
    skipped = 0

    for i, player in enumerate(
        players,
        start=1,
    ):
        current = get_player_photo_record(
            player
        )

        if current:
            status = current.get(
                "status"
            )

            skip = (
                status == "found"
                or (
                    status == "missing"
                    and not args.retry_missing
                )
            )

            if skip:
                skipped += 1

                if (
                    i % 100 == 0
                    or i == len(players)
                ):
                    print(
                        f"[{i}/{len(players)}] "
                        f"continuando... "
                        f"found total="
                        f"{get_counts().get('found',0)}"
                    )

                continue

        try:
            row = resolve_player_photo(
                player,
                id_index=id_index,
                refresh=bool(
                    args.retry_missing
                ),
            )

            processed += 1

            status = (
                row.get(
                    "status"
                )
                if row
                else "error"
            )

            if status == "found":
                found_now += 1
                icon = "✅"

            elif status == "missing":
                missing_now += 1
                icon = "➖"

            else:
                errors += 1
                icon = "⚠️"

            atp_id = (
                row.get(
                    "atp_id"
                )
                if row
                else None
            )

            suffix = (
                f" · ATP ID {atp_id}"
                if atp_id
                else ""
            )

            print(
                f"[{i}/{len(players)}] "
                f"{icon} {player}{suffix}"
            )

        except KeyboardInterrupt:
            print()
            print(
                "Proceso detenido por el usuario."
            )
            print(
                "Puedes ejecutarlo otra vez: "
                "continuará donde quedó."
            )
            break

        except Exception as exc:
            errors += 1

            print(
                f"[{i}/{len(players)}] "
                f"⚠️ {player} · {exc}"
            )

        time.sleep(
            max(
                0.0,
                float(
                    args.delay
                ),
            )
        )

    counts = get_counts()

    print()
    print("=" * 76)
    print("RESUMEN")
    print("=" * 76)
    print(
        "Procesados en esta ejecución:",
        processed,
    )
    print(
        "Fotos encontradas ahora:",
        found_now,
    )
    print(
        "Sin foto oficial / sin ID ahora:",
        missing_now,
    )
    print(
        "Errores:",
        errors,
    )
    print(
        "Ya revisados y saltados:",
        skipped,
    )
    print()
    print(
        "TOTAL FOUND:",
        counts.get(
            "found",
            0,
        ),
    )
    print(
        "TOTAL MISSING:",
        counts.get(
            "missing",
            0,
        ),
    )
    print(
        "TOTAL ERROR:",
        counts.get(
            "error",
            0,
        ),
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
