
from __future__ import annotations

import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://api.sportsgameodds.com/v2"
PROJECT_DIR = Path(__file__).resolve().parent

# SportsGameOdds publica Bet365 y Pinnacle.
# Winamax no figura actualmente en su catálogo público de bookmakers.
BOOKMAKER_IDS = {
    "Bet365": "bet365",
    "Pinnacle": "pinnacle",
}

DEFAULT_SOT_BOOKMAKERS = ("Bet365", "Pinnacle")


def _read_dotenv_value(name: str) -> str:
    path = PROJECT_DIR / ".env"
    if not path.exists():
        return ""
    try:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def get_sgo_api_key() -> str:
    value = os.getenv("SPORTSGAMEODDS_API_KEY", "").strip()
    if value:
        return value
    try:
        value = str(st.secrets.get("SPORTSGAMEODDS_API_KEY", "")).strip()
        if value:
            return value
    except Exception:
        pass
    return _read_dotenv_value("SPORTSGAMEODDS_API_KEY")


def _norm(value: Any) -> str:
    s = str(value or "").lower().strip()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    aliases = {
        "ath madrid": "atletico madrid",
        "atletico de madrid": "atletico madrid",
        "ath bilbao": "athletic bilbao",
        "athletic club": "athletic bilbao",
        "sociedad": "real sociedad",
        "espanol": "espanyol",
        "deportivo a coruna": "deportivo la coruna",
        "deportivo de la coruna": "deportivo la coruna",
        "malaga cf": "malaga",
        "deportivo alaves": "alaves",
        "real betis balompie": "betis",
        "rc celta de vigo": "celta",
    }
    return aliases.get(s, s)


def _team_score(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.96
    return SequenceMatcher(None, na, nb).ratio()


def _api_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    key = get_sgo_api_key()
    if not key:
        return {
            "ok": False,
            "message": "Falta SPORTSGAMEODDS_API_KEY.",
            "data": None,
        }

    params = dict(params)
    params["apiKey"] = key

    try:
        response = requests.get(
            BASE_URL + path,
            params=params,
            timeout=30,
        )
    except Exception as exc:
        return {
            "ok": False,
            "message": f"SportsGameOdds: error de conexión: {exc}",
            "data": None,
        }

    try:
        payload = response.json()
    except Exception:
        payload = None

    if response.status_code != 200:
        msg = payload if payload is not None else response.text[:500]
        return {
            "ok": False,
            "message": f"SportsGameOdds {response.status_code}: {msg}",
            "data": payload,
        }

    return {
        "ok": True,
        "message": "",
        "data": payload,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def sot_market_catalog() -> dict[str, Any]:
    # Descubrimos los market IDs reales en vez de asumir el periodID.
    result = _api_get(
        "/markets",
        {
            "sportID": "SOCCER",
            "statID": "shots_onGoal",
            "betTypeID": "ou",
            "sideID": "over",
            "isSupported": "true",
            "limit": 10000,
        },
    )
    if not result.get("ok"):
        return {
            "ok": False,
            "message": result.get("message", ""),
            "markets": [],
        }

    payload = result.get("data")
    if isinstance(payload, dict):
        markets = payload.get("data") or payload.get("markets") or []
    elif isinstance(payload, list):
        markets = payload
    else:
        markets = []

    clean = []
    for m in markets:
        if not isinstance(m, dict):
            continue
        if str(m.get("statID")) != "shots_onGoal":
            continue
        if str(m.get("betTypeID")) != "ou":
            continue
        if str(m.get("sideID")) != "over":
            continue
        if str(m.get("statEntityID")) not in {"all", "home", "away"}:
            continue
        clean.append(m)

    return {"ok": True, "message": "", "markets": clean}


def _event_names(event: dict[str, Any]) -> tuple[str, str]:
    teams = event.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}

    def display(team):
        names = team.get("names") or {}
        return (
            names.get("display")
            or names.get("short")
            or team.get("name")
            or team.get("teamID")
            or ""
        )

    return str(display(home)), str(display(away))


def _find_event(events: list[dict[str, Any]], match: dict[str, Any]) -> dict[str, Any] | None:
    target = pd.to_datetime(match.get("utc_time"), utc=True, errors="coerce")
    best = None
    best_score = -1.0

    for event in events:
        home, away = _event_names(event)

        direct = (
            _team_score(match.get("home_team", ""), home)
            + _team_score(match.get("away_team", ""), away)
        ) / 2.0
        swapped = (
            _team_score(match.get("home_team", ""), away)
            + _team_score(match.get("away_team", ""), home)
        ) / 2.0

        score = max(direct, swapped)
        if score < 0.75:
            continue

        event_time = pd.to_datetime(
            event.get("startTime") or event.get("startsAt"),
            utc=True,
            errors="coerce",
        )
        if not pd.isna(target) and not pd.isna(event_time):
            hours = abs((event_time - target).total_seconds()) / 3600
            if hours > 18:
                continue
            score += max(0, 0.10 - hours * 0.005)

        if score > best_score:
            best_score = score
            best = event

    return best


def _american_to_decimal(value: Any) -> float | None:
    try:
        x = float(str(value).replace("+", "").strip())
    except Exception:
        return None

    if x == 0:
        return None
    if x > 0:
        return 1.0 + x / 100.0
    return 1.0 + 100.0 / abs(x)


def _line_from_row(row: dict[str, Any]) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", str(row.get("label", "")))
    return float(match.group(1)) if match else None


def _entity_from_row(row: dict[str, Any]) -> str:
    key = str(row.get("key", ""))
    if key.startswith("home_"):
        return "home"
    if key.startswith("away_"):
        return "away"
    return "all"


def _market_ids_for_rows(
    catalog: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[str]:
    entities = {_entity_from_row(row) for row in rows}
    out = []
    for market in catalog:
        if str(market.get("statEntityID")) not in entities:
            continue
        odd_id = market.get("oddID")
        if odd_id and odd_id not in out:
            out.append(str(odd_id))
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_sot_events(odd_ids: tuple[str, ...]) -> dict[str, Any]:
    if not odd_ids:
        return {"ok": False, "message": "No hay market IDs SOT compatibles.", "events": []}

    result = _api_get(
        "/events",
        {
            "sportID": "SOCCER",
            "oddsAvailable": "true",
            "started": "false",
            "live": "false",
            "oddIDs": ",".join(odd_ids),
            "includeOpposingOdds": "true",
            "includeAltLines": "true",
            "bookmakerID": "bet365,pinnacle",
            "limit": 100,
        },
    )
    if not result.get("ok"):
        return {
            "ok": False,
            "message": result.get("message", ""),
            "events": [],
        }

    payload = result.get("data")
    if isinstance(payload, dict):
        events = payload.get("data") or payload.get("events") or []
    elif isinstance(payload, list):
        events = payload
    else:
        events = []

    return {"ok": True, "message": "", "events": events}


def _collect_book_lines(
    odd: dict[str, Any],
    bookmaker_id: str,
) -> list[tuple[float, float]]:
    """
    Returns [(line, decimal_odds), ...] including main line + alt lines.
    """
    out = []
    by_book = odd.get("byBookmaker") or {}
    book = by_book.get(bookmaker_id)
    if not isinstance(book, dict):
        return out

    if book.get("available") is True:
        line = book.get("overUnder")
        price = _american_to_decimal(book.get("odds"))
        try:
            line = float(line)
        except Exception:
            line = None
        if line is not None and price and price > 1.0:
            out.append((line, price))

    for alt in book.get("altLines") or []:
        if not isinstance(alt, dict):
            continue
        if alt.get("available") is not True:
            continue
        try:
            line = float(alt.get("overUnder"))
        except Exception:
            continue
        price = _american_to_decimal(alt.get("odds"))
        if price and price > 1.0:
            out.append((line, price))

    return out


def populate_sot_odds(
    match: dict[str, Any],
    sot_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    kickoff = pd.to_datetime(match.get("utc_time"), utc=True, errors="coerce")
    if pd.isna(kickoff):
        return {"ok": False, "message": "Hora de partido inválida.", "prices": {}}

    if kickoff <= pd.Timestamp.now(tz="UTC"):
        return {
            "ok": False,
            "locked": True,
            "message": "Partido iniciado: SOT live bloqueado.",
            "prices": {},
        }

    catalog = sot_market_catalog()
    if not catalog.get("ok"):
        return {
            "ok": False,
            "message": catalog.get("message", "No se pudo descubrir SOT."),
            "prices": {},
        }

    odd_ids = _market_ids_for_rows(catalog["markets"], sot_rows)
    fetched = fetch_sot_events(tuple(odd_ids))
    if not fetched.get("ok"):
        return {
            "ok": False,
            "message": fetched.get("message", "No se pudieron cargar SOT."),
            "prices": {},
        }

    event = _find_event(fetched.get("events", []), match)
    if event is None:
        return {
            "ok": False,
            "message": "SportsGameOdds no encontró este fixture con SOT disponible.",
            "prices": {},
        }

    prices: dict[str, dict[str, float]] = {}
    odds = event.get("odds") or {}

    # Index markets by entity.
    catalog_by_id = {
        str(m.get("oddID")): m
        for m in catalog.get("markets", [])
        if m.get("oddID")
    }

    for row in sot_rows:
        target_line = _line_from_row(row)
        if target_line is None:
            continue

        entity = _entity_from_row(row)

        candidate_ids = [
            odd_id
            for odd_id, market in catalog_by_id.items()
            if str(market.get("statEntityID")) == entity
        ]

        for odd_id in candidate_ids:
            odd = odds.get(odd_id)
            if not isinstance(odd, dict):
                continue

            for visible_name, bookmaker_id in (
                ("Bet365", "bet365"),
                ("Pinnacle", "pinnacle"),
            ):
                lines = _collect_book_lines(odd, bookmaker_id)
                exact = [
                    price for line, price in lines
                    if abs(line - target_line) < 1e-9
                ]
                if exact:
                    prices.setdefault(row["id"], {})[visible_name] = max(exact)

    return {
        "ok": True,
        "message": "",
        "prices": prices,
        "provider": "SportsGameOdds",
        "event_id": event.get("eventID"),
        "books": ["Bet365", "Pinnacle"],
        "note": (
            "SOT usa SportsGameOdds. Bet365 y Pinnacle están soportados. "
            "Winamax no figura actualmente en el catálogo público de este proveedor."
        ),
    }
