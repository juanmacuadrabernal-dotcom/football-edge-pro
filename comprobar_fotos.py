# ============================================================
# TENNIS EDGE PRO · CHECK PHOTO COVERAGE
# ============================================================

import sqlite3

from player_photos import (
    DB_PATH,
    init_player_photos_table,
)


def main():
    init_player_photos_table()

    with sqlite3.connect(
        DB_PATH
    ) as con:

        total_players = con.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT winner_name AS n
                FROM matches
                WHERE winner_name IS NOT NULL
                UNION
                SELECT loser_name AS n
                FROM matches
                WHERE loser_name IS NOT NULL
            )
            """
        ).fetchone()[0]

        rows = con.execute(
            """
            SELECT status, COUNT(*)
            FROM player_photos
            GROUP BY status
            ORDER BY status
            """
        ).fetchall()

        missing_examples = con.execute(
            """
            SELECT player_name
            FROM player_photos
            WHERE status='missing'
            ORDER BY player_name
            LIMIT 30
            """
        ).fetchall()

    counts = dict(
        rows
    )

    found = int(
        counts.get(
            "found",
            0,
        )
    )

    coverage = (
        found / total_players
        if total_players
        else 0.0
    )

    print("=" * 72)
    print(
        "TENNIS EDGE PRO · COBERTURA DE FOTOS"
    )
    print("=" * 72)
    print(
        "Jugadores únicos:",
        total_players,
    )
    print(
        "Con foto oficial:",
        found,
    )
    print(
        "Sin foto / sin resolver:",
        counts.get(
            "missing",
            0,
        ),
    )
    print(
        "Errores:",
        counts.get(
            "error",
            0,
        ),
    )
    print(
        "Cobertura:",
        f"{coverage:.2%}",
    )

    if missing_examples:
        print()
        print(
            "Ejemplos pendientes:"
        )

        for row in missing_examples:
            print(
                " -",
                row[0],
            )

    print("=" * 72)


if __name__ == "__main__":
    main()
