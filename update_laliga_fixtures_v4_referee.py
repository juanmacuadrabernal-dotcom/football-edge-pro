from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from update_laliga_fixtures_v3 import (
    main as base_update,
    DB_PATH,
    HEADERS,
)


DETAIL_URLS = [
    "https://www.fotmob.com/api/matchDetails",
    "https://www.fotmob.com/api/data/matchDetails",
]

MAX_FIXTURES_TO_CHECK = 30
MAX_WORKERS = 4
TIMEOUT = 15
RETRIES = 2


def _norm(value):
    if value is None:
        return ""

    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _clean_text(value):
    if value is None:
        return None

    if isinstance(value, dict):
        for key in (
            "text",
            "name",
            "title",
            "value",
        ):
            if key in value and value[key]:
                return str(value[key]).strip()
        return None

    s = str(value).strip()
    return s or None


def _find_referee(payload):
    candidates = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                key_l = str(key).lower()
                new_path = (
                    f"{path}.{key_l}"
                    if path
                    else key_l
                )

                if "referee" in key_l:
                    text = _clean_text(
                        value
                    )

                    if text:
                        candidates.append(
                            text
                        )

                walk(
                    value,
                    new_path,
                )

        elif isinstance(node, list):
            for value in node:
                walk(
                    value,
                    path,
                )

    walk(payload)

    for value in candidates:
        if (
            value
            and len(value) <= 120
            and not value.startswith("{")
            and not value.startswith("[")
        ):
            return value

    return None


def _session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _get_detail(match_id):
    session = _session()
    last_error = None

    for url in DETAIL_URLS:
        for attempt in range(
            1,
            RETRIES + 1,
        ):
            try:
                r = session.get(
                    url,
                    params={
                        "matchId": int(
                            match_id
                        )
                    },
                    timeout=TIMEOUT,
                )

                if r.status_code == 200:
                    payload = r.json()

                    return {
                        "match_id": int(
                            match_id
                        ),
                        "referee_raw": (
                            _find_referee(
                                payload
                            )
                        ),
                        "ok": True,
                        "error": None,
                    }

                last_error = (
                    f"HTTP {r.status_code}"
                )

            except Exception as exc:
                last_error = str(exc)

            if attempt < RETRIES:
                time.sleep(
                    0.7 * attempt
                )

    return {
        "match_id": int(match_id),
        "referee_raw": None,
        "ok": False,
        "error": last_error,
    }


def _build_alias_map(conn):
    try:
        df = pd.read_sql_query(
            """
            SELECT referee, referee_raw
            FROM referee_match_history
            WHERE referee IS NOT NULL
            """,
            conn,
        )
    except Exception:
        return {}

    candidates = {}

    for row in df.itertuples(
        index=False
    ):
        canonical = row.referee

        for value in (
            canonical,
            getattr(
                row,
                "referee_raw",
                None,
            ),
        ):
            if value is None or pd.isna(
                value
            ):
                continue

            key = _norm(value)

            if key:
                candidates.setdefault(
                    key,
                    set(),
                ).add(
                    str(canonical).strip()
                )

    return {
        key: next(iter(names))
        for key, names
        in candidates.items()
        if len(names) == 1
    }


def main():
    # 1) Conservamos el actualizador V8 que ya funciona.
    base_update()

    print()
    print("=" * 80)
    print("LALIGA EDGE PRO - ENRIQUECER ÁRBITROS DE LA JORNADA")
    print("=" * 80)

    with sqlite3.connect(
        DB_PATH
    ) as conn:
        fixtures = pd.read_sql_query(
            """
            SELECT *
            FROM laliga_upcoming_fixtures
            ORDER BY utc_time
            """,
            conn,
        )

        if fixtures.empty:
            print(
                "No hay fixtures para enriquecer."
            )
            return

        fixtures["utc_time"] = (
            pd.to_datetime(
                fixtures["utc_time"],
                utc=True,
                errors="coerce",
            )
        )

        now = pd.Timestamp.now(
            tz="UTC"
        )

        upcoming = fixtures[
            fixtures["utc_time"]
            >= now
            - pd.Timedelta(hours=12)
        ].head(
            MAX_FIXTURES_TO_CHECK
        ).copy()

        alias_map = (
            _build_alias_map(
                conn
            )
        )

        cols = [
            r[1]
            for r in conn.execute(
                """
                PRAGMA table_info(
                    laliga_upcoming_fixtures
                )
                """
            ).fetchall()
        ]

        if "referee_raw" not in cols:
            conn.execute(
                """
                ALTER TABLE
                laliga_upcoming_fixtures
                ADD COLUMN referee_raw TEXT
                """
            )
            conn.commit()

    print(
        f"Consultando detalle de "
        f"{len(upcoming)} fixtures "
        f"(máx. {MAX_FIXTURES_TO_CHECK})..."
    )

    results = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:
        futures = {
            executor.submit(
                _get_detail,
                int(row.match_id),
            ): int(row.match_id)
            for row in upcoming.itertuples(
                index=False
            )
        }

        done = 0

        for future in as_completed(
            futures
        ):
            done += 1

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "match_id": futures[
                        future
                    ],
                    "referee_raw": None,
                    "ok": False,
                    "error": str(exc),
                }

            results.append(
                result
            )

            if (
                done % 5 == 0
                or done == len(futures)
            ):
                found = sum(
                    1
                    for r in results
                    if r.get(
                        "referee_raw"
                    )
                )

                print(
                    f"  [{done}/{len(futures)}] "
                    f"árbitros encontrados="
                    f"{found}"
                )

    assigned = 0

    with sqlite3.connect(
        DB_PATH
    ) as conn:
        for result in results:
            raw = result.get(
                "referee_raw"
            )

            if not raw:
                continue

            canonical = alias_map.get(
                _norm(raw),
                raw,
            )

            conn.execute(
                """
                UPDATE laliga_upcoming_fixtures
                SET referee = ?,
                    referee_raw = ?
                WHERE match_id = ?
                """,
                (
                    canonical,
                    raw,
                    int(
                        result["match_id"]
                    ),
                ),
            )

            assigned += 1

        conn.commit()

        check = pd.read_sql_query(
            """
            SELECT utc_time,
                   home_team,
                   away_team,
                   referee
            FROM laliga_upcoming_fixtures
            ORDER BY utc_time
            LIMIT 20
            """,
            conn,
        )

    print()
    print(
        f"Árbitros asignados detectados: "
        f"{assigned}"
    )

    if assigned == 0:
        print(
            "FotMob todavía no publica árbitro "
            "para estos partidos. La app usará "
            "perfil neutral y lo indicará como "
            "'pendiente'."
        )
    else:
        print()
        print("PRÓXIMOS PARTIDOS:")
        print("-" * 80)

        for r in check.itertuples(
            index=False
        ):
            ref = (
                r.referee
                if r.referee
                else "PENDIENTE"
            )

            print(
                f"{r.utc_time} | "
                f"{r.home_team} - "
                f"{r.away_team} | "
                f"{ref}"
            )

    print("=" * 80)


if __name__ == "__main__":
    main()
