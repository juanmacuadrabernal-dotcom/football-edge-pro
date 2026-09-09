
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

BASE_URL = "https://api.the-odds-api.com/v4"
PROJECT_DIR = Path(__file__).resolve().parent

SPORT_KEYS = {
    "laliga": "soccer_spain_la_liga",
    "eliteserien": "soccer_norway_eliteserien",
}

# Máximo 3 casas, tal como acordamos.
DEFAULT_BOOKMAKERS = ("William Hill", "Bet Victor", "Pinnacle")
BOOKMAKER_KEYS = {
    "William Hill": "williamhill",
    "Bet Victor": "betvictor",
    "Pinnacle": "pinnacle",
    "Unibet": "unibet_uk",
}


def _read_dotenv_value(name: str) -> str:
    path = PROJECT_DIR / ".env"
    if not path.exists():
        return ""
    try:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == name:
                return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def get_api_key() -> str:
    value = os.getenv("THE_ODDS_API_KEY", "").strip()
    if value:
        return value

    try:
        value = str(st.secrets.get("THE_ODDS_API_KEY", "")).strip()
        if value:
            return value
    except Exception:
        pass

    return _read_dotenv_value("THE_ODDS_API_KEY")


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
        "vallecano": "rayo vallecano",
        "espanol": "espanyol",
        "deportivo a coruna": "deportivo la coruna",
        "deportivo de la coruna": "deportivo la coruna",
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


def _bookmaker_params(selected_bookmakers: tuple[str, ...]) -> str:
    keys = []
    for name in selected_bookmakers[:3]:
        key = BOOKMAKER_KEYS.get(name)
        if key and key not in keys:
            keys.append(key)
    if not keys:
        keys = [
            BOOKMAKER_KEYS["William Hill"],
            BOOKMAKER_KEYS["Bet Victor"],
            BOOKMAKER_KEYS["Pinnacle"],
        ]
    return ",".join(keys)


def _api_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    key = get_api_key()
    if not key:
        return {
            "ok": False,
            "message": "Falta THE_ODDS_API_KEY.",
            "data": None,
        }

    params = dict(params)
    params["apiKey"] = key
    params.setdefault("oddsFormat", "decimal")
    params.setdefault("dateFormat", "iso")

    try:
        response = requests.get(
            BASE_URL + path,
            params=params,
            timeout=25,
        )
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Error de conexión: {exc}",
            "data": None,
        }

    try:
        payload = response.json()
    except Exception:
        payload = None

    if response.status_code != 200:
        if isinstance(payload, dict):
            msg = payload.get("message") or payload.get("error_code") or str(payload)
        else:
            msg = response.text[:500]
        return {
            "ok": False,
            "message": f"The Odds API {response.status_code}: {msg}",
            "data": payload,
            "remaining": response.headers.get("x-requests-remaining"),
        }

    return {
        "ok": True,
        "message": "",
        "data": payload,
        "remaining": response.headers.get("x-requests-remaining"),
        "used": response.headers.get("x-requests-used"),
        "last": response.headers.get("x-requests-last"),
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_league_featured(
    league_key: str,
    selected_bookmakers: tuple[str, ...],
) -> dict[str, Any]:
    sport = SPORT_KEYS.get(league_key)
    if not sport:
        return {"ok": False, "message": "Liga no soportada.", "events": []}

    result = _api_get(
        f"/sports/{sport}/odds",
        {
            "bookmakers": _bookmaker_params(selected_bookmakers),
            "markets": "h2h,totals",
        },
    )
    return {
        "ok": result.get("ok", False),
        "message": result.get("message", ""),
        "events": result.get("data") or [],
        "remaining": result.get("remaining"),
    }


def _find_event(events, match) -> dict[str, Any] | None:
    target = pd.to_datetime(match.get("utc_time"), utc=True, errors="coerce")
    best = None
    best_score = -1.0

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        direct = (
            _team_score(match.get("home_team", ""), home)
            + _team_score(match.get("away_team", ""), away)
        ) / 2.0

        swapped = (
            _team_score(match.get("home_team", ""), away)
            + _team_score(match.get("away_team", ""), home)
        ) / 2.0

        score = max(direct, swapped)
        if score < 0.78:
            continue

        event_time = pd.to_datetime(
            event.get("commence_time"),
            utc=True,
            errors="coerce",
        )
        if not pd.isna(target) and not pd.isna(event_time):
            hours = abs((event_time - target).total_seconds()) / 3600.0
            if hours > 18:
                continue
            score += max(0.0, 0.10 - hours * 0.005)

        if score > best_score:
            best_score = score
            best = event

    return best


@st.cache_data(ttl=300, show_spinner=False)
def fetch_event_extras(
    league_key: str,
    event_id: str,
    selected_bookmakers: tuple[str, ...],
) -> dict[str, Any]:
    sport = SPORT_KEYS.get(league_key)
    if not sport:
        return {"ok": False, "message": "Liga no soportada.", "event": None}

    # Mercados que encajan con nuestros modelos de fútbol.
    result = _api_get(
        f"/sports/{sport}/events/{event_id}/odds",
        {
            "bookmakers": _bookmaker_params(selected_bookmakers),
            "markets": ",".join([
                "alternate_totals",
                "alternate_totals_corners",
                "alternate_totals_cards",
            ]),
        },
    )

    return {
        "ok": result.get("ok", False),
        "message": result.get("message", ""),
        "event": result.get("data"),
        "remaining": result.get("remaining"),
    }


def _canonical_book(title: str, key: str) -> str | None:
    text = _norm(f"{title} {key}")
    if "william hill" in text or key == "williamhill":
        return "William Hill"
    if "bet victor" in text or "betvictor" in text:
        return "Bet Victor"
    if "pinnacle" in text:
        return "Pinnacle"
    if "unibet" in text:
        return "Unibet"
    return None


def _line_from_row(row: dict[str, Any]) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", str(row.get("label", "")))
    return float(m.group(1)) if m else None


def _price_from_market(market: dict[str, Any], line: float) -> float | None:
    prices = []
    for outcome in market.get("outcomes", []) or []:
        name = str(outcome.get("name", "")).strip().lower()
        point = outcome.get("point")
        try:
            point = float(point)
            price = float(outcome.get("price"))
        except Exception:
            continue
        if name == "over" and abs(point - line) < 1e-6 and price > 1.0:
            prices.append(price)
    return max(prices) if prices else None


def _parse_rows_from_event(
    event: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    selected_bookmakers: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    if not isinstance(event, dict):
        return {}

    selected = set(selected_bookmakers[:3])
    out: dict[str, dict[str, float]] = {}

    for bookmaker in event.get("bookmakers", []) or []:
        cname = _canonical_book(
            str(bookmaker.get("title", "")),
            str(bookmaker.get("key", "")),
        )
        if not cname or cname not in selected:
            continue

        markets = {
            str(m.get("key", "")): m
            for m in bookmaker.get("markets", []) or []
        }

        for row in rows:
            cat = row.get("cat")
            key = row.get("key")
            price = None

            if cat == "1x2":
                market = markets.get("h2h")
                if market:
                    target = (
                        "Draw" if key == "draw"
                        else event.get("home_team") if key == "home"
                        else event.get("away_team")
                    )
                    for outcome in market.get("outcomes", []) or []:
                        if _team_score(str(outcome.get("name", "")), str(target)) >= 0.92:
                            try:
                                candidate = float(outcome.get("price"))
                            except Exception:
                                continue
                            if candidate > 1.0:
                                price = candidate
                                break

            elif cat == "goals":
                line = _line_from_row(row)
                for mk in ("alternate_totals", "totals"):
                    market = markets.get(mk)
                    if market and line is not None:
                        price = _price_from_market(market, line)
                        if price:
                            break

            elif cat == "corners":
                line = _line_from_row(row)
                market = markets.get("alternate_totals_corners")
                if market and line is not None:
                    price = _price_from_market(market, line)

            elif cat == "cards":
                line = _line_from_row(row)
                market = markets.get("alternate_totals_cards")
                if market and line is not None:
                    price = _price_from_market(market, line)

            # El proveedor no documenta total/equipo SOT; se deja manual
            # para no inventar una cuota ni cruzarla con player props.
            elif cat == "sot":
                price = None

            if price:
                out.setdefault(row["id"], {})[cname] = price

    return out


def populate_match_odds(
    league_key: str,
    match: dict[str, Any],
    market_rows: list[dict[str, Any]],
    selected_bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
) -> dict[str, Any]:
    kickoff = pd.to_datetime(match.get("utc_time"), utc=True, errors="coerce")
    if pd.isna(kickoff):
        return {"ok": False, "message": "Hora de partido inválida.", "prices": {}}

    if kickoff <= pd.Timestamp.now(tz="UTC"):
        return {
            "ok": False,
            "locked": True,
            "message": "Partido iniciado: cuotas live bloqueadas.",
            "prices": {},
        }

    featured = fetch_league_featured(
        league_key,
        tuple(selected_bookmakers[:3]),
    )
    if not featured.get("ok"):
        return {
            "ok": False,
            "message": featured.get("message", "Error cargando cuotas."),
            "prices": {},
        }

    event = _find_event(featured.get("events", []), match)
    if not event:
        return {
            "ok": False,
            "message": "The Odds API todavía no tiene este partido/mercado disponible.",
            "prices": {},
            "remaining": featured.get("remaining"),
        }

    # Primero 1X2 y total principal ya devueltos por la consulta de liga.
    prices = _parse_rows_from_event(
        event,
        market_rows,
        tuple(selected_bookmakers[:3]),
    )

    # Luego líneas alternativas de goles, córners y tarjetas SOLO para
    # el partido seleccionado. Así no gastamos créditos en toda la jornada.
    extras = fetch_event_extras(
        league_key,
        str(event.get("id")),
        tuple(selected_bookmakers[:3]),
    )
    if extras.get("ok") and extras.get("event"):
        extra_prices = _parse_rows_from_event(
            extras["event"],
            market_rows,
            tuple(selected_bookmakers[:3]),
        )
        for market_id, book_prices in extra_prices.items():
            prices.setdefault(market_id, {}).update(book_prices)

    return {
        "ok": True,
        "message": extras.get("message", "") if not extras.get("ok") else "",
        "prices": prices,
        "remaining": extras.get("remaining") or featured.get("remaining"),
        "event_id": event.get("id"),
        "provider": "The Odds API",
        "sot_note": (
            "SOT total/equipo no está documentado por este proveedor; "
            "se mantiene cuota manual solo para SOT."
        ),
    }
