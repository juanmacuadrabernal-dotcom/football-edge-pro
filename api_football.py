import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

from config import API_BASE_URL, COUNTRY, LEAGUE_NAME, TIMEZONE

load_dotenv()


class APIFootballError(RuntimeError):
    pass


def get_api_key() -> str:
    """
    Busca la API key primero en Streamlit Secrets
    y después en el archivo .env.
    """

    try:
        key = st.secrets.get("API_FOOTBALL_KEY", "")

        if key:
            return str(key).strip()

    except Exception:
        pass

    return os.getenv("API_FOOTBALL_KEY", "").strip()


def api_get(
    endpoint: str,
    params: dict[str, Any] | None = None
) -> dict[str, Any]:

    key = get_api_key()

    if not key:
        raise APIFootballError(
            "Falta API_FOOTBALL_KEY. "
            "Añádela en el archivo .env."
        )

    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"

    headers = {
        "x-apisports-key": key
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params or {},
            timeout=30
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        raise APIFootballError(
            f"Error conectando con API-Football: {exc}"
        ) from exc

    try:
        payload = response.json()

    except Exception:

        raise APIFootballError(
            "API-Football devolvió una respuesta que no es JSON."
        )

    errors = payload.get("errors")

    if errors:
        raise APIFootballError(
            f"API-Football devolvió errores: {errors}"
        )

    return payload


@st.cache_data(
    ttl=60 * 60 * 12,
    show_spinner=False
)
def get_eliteserien_league() -> dict[str, Any]:
    """
    Busca la Eliteserien en API-Football.

    IMPORTANTE:
    API-Football no permite utilizar al mismo tiempo
    'country' y 'search'.

    Por eso buscamos primero por nombre y después
    filtramos nosotros que sea Noruega.
    """

    payload = api_get(
        "leagues",
        {
            "search": LEAGUE_NAME
        }
    )

    results = payload.get("response", [])

    if not results:
        raise APIFootballError(
            "No se encontró la Eliteserien en API-Football."
        )

    # Buscamos coincidencia exacta:
    # nombre = Eliteserien
    # país = Norway

    exact = []

    for item in results:

        league = item.get("league", {})
        country = item.get("country", {})

        league_name = str(
            league.get("name", "")
        ).strip().lower()

        country_name = str(
            country.get("name", "")
        ).strip().lower()

        if (
            league_name == LEAGUE_NAME.lower()
            and country_name == COUNTRY.lower()
        ):
            exact.append(item)

    if exact:
        return exact[0]

    # Segundo intento:
    # si no encontramos coincidencia exacta,
    # buscamos cualquier Eliteserien.

    for item in results:

        league_name = str(
            item.get("league", {}).get("name", "")
        ).strip().lower()

        if league_name == LEAGUE_NAME.lower():
            return item

    raise APIFootballError(
        "API-Football respondió, pero no encontramos "
        "la Eliteserien de Noruega."
    )


def get_available_seasons() -> list[int]:
    """
    Devuelve todas las temporadas disponibles
    para la Eliteserien.
    """

    league = get_eliteserien_league()

    seasons = league.get("seasons", [])

    available = []

    for season in seasons:

        year = season.get("year")

        if year is not None:

            try:
                available.append(int(year))
            except Exception:
                pass

    return sorted(set(available))


def get_current_season() -> int:
    """
    Detecta automáticamente la temporada actual.
    """

    league = get_eliteserien_league()

    seasons = league.get("seasons", [])

    current = []

    for season in seasons:

        if (
            season.get("current") is True
            and season.get("year") is not None
        ):

            current.append(
                int(season["year"])
            )

    if current:
        return max(current)

    available = get_available_seasons()

    if not available:
        raise APIFootballError(
            "No se pudo determinar la temporada actual."
        )

    return max(available)


@st.cache_data(
    ttl=60 * 30,
    show_spinner=False
)
def get_current_round(
    season: int
) -> str | None:
    """
    Obtiene automáticamente la jornada actual.
    """

    league_id = (
        get_eliteserien_league()
        ["league"]
        ["id"]
    )

    payload = api_get(
        "fixtures/rounds",
        {
            "league": league_id,
            "season": season,
            "current": "true"
        }
    )

    rounds = payload.get("response", [])

    if not rounds:
        return None

    return rounds[0]


@st.cache_data(
    ttl=60 * 10,
    show_spinner=False
)
def get_round_fixtures(
    season: int,
    round_name: str
) -> list[dict[str, Any]]:
    """
    Descarga todos los partidos de una jornada.
    """

    league_id = (
        get_eliteserien_league()
        ["league"]
        ["id"]
    )

    payload = api_get(
        "fixtures",
        {
            "league": league_id,
            "season": season,
            "round": round_name,
            "timezone": TIMEZONE
        }
    )

    return payload.get("response", [])


@st.cache_data(
    ttl=60 * 30,
    show_spinner=False
)
def get_season_fixtures(
    season: int
) -> list[dict[str, Any]]:
    """
    Descarga todos los partidos de una temporada.
    """

    league_id = (
        get_eliteserien_league()
        ["league"]
        ["id"]
    )

    payload = api_get(
        "fixtures",
        {
            "league": league_id,
            "season": season,
            "timezone": TIMEZONE
        }
    )

    return payload.get("response", [])


@st.cache_data(
    ttl=60 * 60,
    show_spinner=False
)
def get_fixture_statistics(
    fixture_id: int
) -> list[dict[str, Any]]:
    """
    Estadísticas del partido:

    - córners
    - tiros
    - tiros a puerta
    - posesión
    - faltas
    - tarjetas
    - offsides
    - etc.
    """

    payload = api_get(
        "fixtures/statistics",
        {
            "fixture": fixture_id
        }
    )

    return payload.get("response", [])


@st.cache_data(
    ttl=60 * 60,
    show_spinner=False
)
def get_fixture_events(
    fixture_id: int
) -> list[dict[str, Any]]:
    """
    Eventos del partido:

    - goles
    - tarjetas
    - sustituciones
    - VAR
    - etc.
    """

    payload = api_get(
        "fixtures/events",
        {
            "fixture": fixture_id
        }
    )

    return payload.get("response", [])


@st.cache_data(
    ttl=60 * 60,
    show_spinner=False
)
def get_fixture_lineups(
    fixture_id: int
) -> list[dict[str, Any]]:
    """
    Obtiene las alineaciones disponibles.
    """

    payload = api_get(
        "fixtures/lineups",
        {
            "fixture": fixture_id
        }
    )

    return payload.get("response", [])


@st.cache_data(
    ttl=60 * 60,
    show_spinner=False
)
def get_fixture_players(
    fixture_id: int
) -> list[dict[str, Any]]:
    """
    Estadísticas individuales de jugadores
    cuando estén disponibles.
    """

    payload = api_get(
        "fixtures/players",
        {
            "fixture": fixture_id
        }
    )

    return payload.get("response", [])