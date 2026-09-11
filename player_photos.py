# ============================================================
# TENNIS EDGE PRO · PLAYER PHOTOS · ATP OFFICIAL
# ------------------------------------------------------------
# - Fotos oficiales alojadas por ATP Tour.
# - Compatible con jugadores ATP y ATP Challenger.
# - Guarda referencias en tennis_edge.db -> player_photos.
# - NO guarda imágenes binarias dentro de SQLite.
# ============================================================

from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Optional

import requests


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "tennis_edge.db"
ATP_ID_CACHE = BASE_DIR / "atp_player_ids_cache.json"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

USER_AGENT = (
    "TennisEdgePro/1.0 "
    "(ATP official player photo resolver)"
)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
)


def connect():
    return sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
    )


def init_player_photos_table():
    with connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS player_photos (
                player_name TEXT PRIMARY KEY,
                normalized_name TEXT,
                wikidata_id TEXT,
                atp_id TEXT,
                photo_url TEXT,
                profile_url TEXT,
                source TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                checked_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_player_photos_status
            ON player_photos(status)
            """
        )

        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_player_photos_atp_id
            ON player_photos(atp_id)
            """
        )

        con.commit()


def normalize_name(value: str) -> str:
    value = unicodedata.normalize(
        "NFKD",
        str(value or ""),
    )

    value = "".join(
        c
        for c in value
        if not unicodedata.combining(c)
    ).lower()

    value = value.replace(
        "ł",
        "l",
    ).replace(
        "ø",
        "o",
    ).replace(
        "đ",
        "d",
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return " ".join(
        value.split()
    )


def official_headshot_url(
    atp_id: str,
) -> Optional[str]:
    atp_id = str(
        atp_id or ""
    ).strip()

    if not atp_id:
        return None

    return (
        "https://www.atptour.com/"
        "-/media/alias/player-headshot/"
        f"{atp_id.lower()}"
    )


def official_profile_url(
    atp_id: str,
) -> Optional[str]:
    atp_id = str(
        atp_id or ""
    ).strip()

    if not atp_id:
        return None

    return (
        "https://www.atptour.com/"
        "en/players/-/"
        f"{atp_id.lower()}/overview"
    )


def _safe_json_get(
    url,
    params,
    timeout=20,
):
    response = SESSION.get(
        url,
        params=params,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


def _claim_value(
    claims,
    property_id,
):
    for claim in claims.get(
        property_id,
        [],
    ):
        try:
            value = (
                claim["mainsnak"]
                ["datavalue"]
                ["value"]
            )

            if value not in (
                None,
                "",
            ):
                return str(
                    value
                ).strip()

        except Exception:
            continue

    return None


def _headshot_exists(
    url: str,
) -> bool:
    """
    ATP devuelve el headshot como imagen.
    Comprobamos content-type y un tamaño mínimo.
    """

    if not url:
        return False

    try:
        response = SESSION.get(
            url,
            timeout=12,
            stream=True,
            allow_redirects=True,
        )

        if response.status_code != 200:
            return False

        content_type = str(
            response.headers.get(
                "Content-Type",
                "",
            )
        ).lower()

        if not content_type.startswith(
            "image/"
        ):
            return False

        # Leemos un pequeño fragmento para evitar falsos 200 vacíos.
        chunk = next(
            response.iter_content(
                chunk_size=2048
            ),
            b"",
        )

        return len(chunk) >= 300

    except Exception:
        return False


def get_player_photo_record(
    player_name: str,
) -> Optional[dict]:
    init_player_photos_table()

    with connect() as con:
        con.row_factory = sqlite3.Row

        row = con.execute(
            """
            SELECT *
            FROM player_photos
            WHERE player_name = ?
            """,
            (
                str(
                    player_name
                ),
            ),
        ).fetchone()

    return (
        dict(row)
        if row
        else None
    )


def get_player_photo_url(
    player_name: str,
) -> Optional[str]:
    row = get_player_photo_record(
        player_name
    )

    if not row:
        return None

    if row.get(
        "status"
    ) != "found":
        return None

    return (
        row.get(
            "photo_url"
        )
        or None
    )


def _save_record(
    player_name,
    *,
    wikidata_id=None,
    atp_id=None,
    photo_url=None,
    profile_url=None,
    status="missing",
):
    with connect() as con:
        con.execute(
            """
            INSERT INTO player_photos (
                player_name,
                normalized_name,
                wikidata_id,
                atp_id,
                photo_url,
                profile_url,
                source,
                status,
                checked_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(player_name) DO UPDATE SET
                normalized_name=excluded.normalized_name,
                wikidata_id=excluded.wikidata_id,
                atp_id=excluded.atp_id,
                photo_url=excluded.photo_url,
                profile_url=excluded.profile_url,
                source=excluded.source,
                status=excluded.status,
                checked_at=CURRENT_TIMESTAMP
            """,
            (
                str(
                    player_name
                ),
                normalize_name(
                    player_name
                ),
                wikidata_id,
                atp_id,
                photo_url,
                profile_url,
                (
                    "ATP Tour"
                    if photo_url
                    else None
                ),
                status,
            ),
        )

        con.commit()


# ============================================================
# ATP-ID INDEX
# ============================================================

def _load_id_cache():
    if not ATP_ID_CACHE.exists():
        return {}

    try:
        data = json.loads(
            ATP_ID_CACHE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(
            data,
            dict,
        ):
            return data

    except Exception:
        pass

    return {}


def _save_id_cache(
    mapping,
):
    ATP_ID_CACHE.write_text(
        json.dumps(
            mapping,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def download_atp_id_index(
    refresh=False,
):
    """
    Descarga de Wikidata la relación:
      nombre -> ATP Player ID (P536)

    Una única consulta reduce muchísimo el número de peticiones.
    """

    cached = _load_id_cache()

    if (
        cached
        and not refresh
    ):
        return cached

    query = """
    SELECT ?item ?atp ?itemLabel WHERE {
      ?item wdt:P536 ?atp .
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
      }
    }
    """

    try:
        response = SESSION.get(
            WIKIDATA_SPARQL,
            params={
                "query": query,
                "format": "json",
            },
            timeout=60,
        )

        response.raise_for_status()

        payload = response.json()

        mapping = {}

        for row in (
            payload.get(
                "results",
                {},
            )
            .get(
                "bindings",
                [],
            )
        ):
            label = (
                row.get(
                    "itemLabel",
                    {},
                )
                .get(
                    "value",
                    "",
                )
            )

            atp_id = (
                row.get(
                    "atp",
                    {},
                )
                .get(
                    "value",
                    "",
                )
            )

            qid_url = (
                row.get(
                    "item",
                    {},
                )
                .get(
                    "value",
                    "",
                )
            )

            if (
                not label
                or not atp_id
            ):
                continue

            key = normalize_name(
                label
            )

            if not key:
                continue

            qid = (
                qid_url.rsplit(
                    "/",
                    1,
                )[-1]
                if qid_url
                else None
            )

            # Si existe una colisión, guardamos una lista.
            existing = mapping.get(
                key
            )

            candidate = {
                "atp_id": str(
                    atp_id
                ).strip(),
                "wikidata_id": qid,
                "label": label,
            }

            if existing is None:
                mapping[key] = candidate

            elif isinstance(
                existing,
                dict,
            ):
                mapping[key] = [
                    existing,
                    candidate,
                ]

            else:
                existing.append(
                    candidate
                )

        if mapping:
            _save_id_cache(
                mapping
            )

            return mapping

    except Exception:
        pass

    return cached


def _resolve_from_index(
    player_name,
    mapping,
):
    key = normalize_name(
        player_name
    )

    value = mapping.get(
        key
    )

    if not value:
        return None

    if isinstance(
        value,
        dict,
    ):
        return value

    # Nombre duplicado en Wikidata:
    # no elegimos a ciegas.
    return None


# ============================================================
# WIKIDATA FALLBACK POR NOMBRE
# ============================================================

def _resolve_wikidata_search(
    player_name: str,
):
    search = _safe_json_get(
        WIKIDATA_API,
        {
            "action": "wbsearchentities",
            "search": player_name,
            "language": "en",
            "uselang": "en",
            "type": "item",
            "limit": 8,
            "format": "json",
        },
    )

    results = search.get(
        "search",
        [],
    )

    if not results:
        return None

    ids = [
        item.get(
            "id"
        )
        for item in results
        if item.get(
            "id"
        )
    ]

    if not ids:
        return None

    entities = _safe_json_get(
        WIKIDATA_API,
        {
            "action": "wbgetentities",
            "ids": "|".join(
                ids
            ),
            "props": (
                "claims|labels|descriptions"
            ),
            "languages": "en",
            "format": "json",
        },
    ).get(
        "entities",
        {},
    )

    target = normalize_name(
        player_name
    )

    candidates = []

    for position, qid in enumerate(
        ids
    ):
        entity = entities.get(
            qid,
            {},
        )

        claims = entity.get(
            "claims",
            {},
        )

        atp_id = _claim_value(
            claims,
            "P536",
        )

        if not atp_id:
            continue

        label = (
            entity.get(
                "labels",
                {},
            )
            .get(
                "en",
                {},
            )
            .get(
                "value",
                "",
            )
        )

        description = (
            entity.get(
                "descriptions",
                {},
            )
            .get(
                "en",
                {},
            )
            .get(
                "value",
                "",
            )
        )

        exact = (
            normalize_name(
                label
            )
            == target
        )

        tennis = (
            "tennis"
            in str(
                description
            ).lower()
        )

        score = (
            (100 if exact else 0)
            + (25 if tennis else 0)
            - position
        )

        candidates.append(
            (
                score,
                {
                    "atp_id": atp_id,
                    "wikidata_id": qid,
                    "label": label,
                },
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, best = (
        candidates[0]
    )

    # Exigimos coincidencia razonable.
    if best_score < 20:
        return None

    return best


# ============================================================
# PUBLIC RESOLVER
# ============================================================

def resolve_player_photo(
    player_name: str,
    *,
    id_index=None,
    refresh=False,
):
    init_player_photos_table()

    player_name = str(
        player_name or ""
    ).strip()

    if not player_name:
        return None

    current = get_player_photo_record(
        player_name
    )

    if (
        current
        and not refresh
        and current.get(
            "status"
        ) in {
            "found",
            "missing",
        }
    ):
        return current

    if id_index is None:
        id_index = download_atp_id_index(
            refresh=False
        )

    resolved = _resolve_from_index(
        player_name,
        id_index,
    )

    if not resolved:
        try:
            resolved = (
                _resolve_wikidata_search(
                    player_name
                )
            )
        except Exception:
            resolved = None

    if not resolved:
        _save_record(
            player_name,
            status="missing",
        )

        return get_player_photo_record(
            player_name
        )

    atp_id = str(
        resolved.get(
            "atp_id",
            "",
        )
    ).strip()

    qid = resolved.get(
        "wikidata_id"
    )

    photo_url = (
        official_headshot_url(
            atp_id
        )
    )

    profile_url = (
        official_profile_url(
            atp_id
        )
    )

    if (
        photo_url
        and _headshot_exists(
            photo_url
        )
    ):
        _save_record(
            player_name,
            wikidata_id=qid,
            atp_id=atp_id,
            photo_url=photo_url,
            profile_url=profile_url,
            status="found",
        )

    else:
        _save_record(
            player_name,
            wikidata_id=qid,
            atp_id=atp_id,
            photo_url=None,
            profile_url=profile_url,
            status="missing",
        )

    return get_player_photo_record(
        player_name
    )


def ensure_photo(
    player_name: str,
):
    """
    Para la app:
    - si está guardado -> instantáneo
    - si nunca se revisó -> intenta resolverlo una vez
    """

    row = get_player_photo_record(
        player_name
    )

    if (
        row
        and row.get(
            "status"
        ) in {
            "found",
            "missing",
        }
    ):
        return row

    return resolve_player_photo(
        player_name
    )
