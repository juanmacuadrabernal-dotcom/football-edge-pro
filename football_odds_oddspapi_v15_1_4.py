
from __future__ import annotations

import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://api.oddspapi.io/v4"
PROJECT_DIR = Path(__file__).resolve().parent

# Máximo 3 casas, tal como acordamos.
DEFAULT_BOOKMAKERS = ("Bet365", "Winamax", "Pinnacle")

BOOKMAKER_SLUGS = {
    "Bet365": "bet365.es",
    "Winamax": "winamax.es",
    "Pinnacle": "pinnacle",
}

# OddsPapi: fútbol = sportId 10
SOCCER_SPORT_ID = 10


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
    value = os.getenv("ODDSPAPI_API_KEY", "").strip()
    if value:
        return value

    try:
        value = str(st.secrets.get("ODDSPAPI_API_KEY", "")).strip()
        if value:
            return value
    except Exception:
        pass

    return _read_dotenv_value("ODDSPAPI_API_KEY")


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
        "atletico madrid": "atletico madrid",
        "ath bilbao": "athletic bilbao",
        "athletic club": "athletic bilbao",
        "sociedad": "real sociedad",
        "real sociedad san sebastian": "real sociedad",
        "vallecano": "rayo vallecano",
        "rayo": "rayo vallecano",
        "espanol": "espanyol",
        "rcd espanyol barcelona": "espanyol",
        "deportivo a coruna": "deportivo la coruna",
        "deportivo de la coruna": "deportivo la coruna",
        "rc deportivo la coruna": "deportivo la coruna",
        "malaga cf": "malaga",
        "celta vigo": "celta",
        "rc celta de vigo": "celta",
        "deportivo alaves": "alaves",
        "real betis balompie": "betis",
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
    ta, tb = set(na.split()), set(nb.split())
    jacc = len(ta & tb) / max(1, len(ta | tb))
    seq = SequenceMatcher(None, na, nb).ratio()
    return max(seq, jacc)


def _bookmaker_param(selected_bookmakers: tuple[str, ...]) -> str:
    slugs = []
    for name in selected_bookmakers[:3]:
        slug = BOOKMAKER_SLUGS.get(name)
        if slug and slug not in slugs:
            slugs.append(slug)
    if not slugs:
        slugs = ["bet365.es", "winamax.es", "pinnacle"]
    return ",".join(slugs[:3])


def _api_get(path: str, params: dict[str, Any], timeout: int = 35) -> dict[str, Any]:
    key = get_api_key()
    if not key:
        return {
            "ok": False,
            "message": "Falta ODDSPAPI_API_KEY.",
            "data": None,
        }

    params = dict(params)
    params["apiKey"] = key

    try:
        response = requests.get(
            BASE_URL + path,
            params=params,
            timeout=timeout,
        )
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Error de conexión con OddsPapi: {exc}",
            "data": None,
        }

    try:
        payload = response.json()
    except Exception:
        payload = None

    if response.status_code == 429:
        retry_ms = 1200
        if isinstance(payload, dict):
            try:
                retry_ms = int(
                    (payload.get("error") or {}).get("retryMs", retry_ms)
                )
            except Exception:
                pass
        time.sleep(min(max(retry_ms / 1000.0 + 0.2, 0.8), 4.0))
        try:
            response = requests.get(
                BASE_URL + path,
                params=params,
                timeout=timeout,
            )
            payload = response.json()
        except Exception as exc:
            return {
                "ok": False,
                "message": f"OddsPapi limitó temporalmente la consulta: {exc}",
                "data": None,
            }

    if response.status_code != 200:
        msg = payload if payload is not None else response.text[:500]
        return {
            "ok": False,
            "message": f"OddsPapi {response.status_code}: {msg}",
            "data": payload,
        }

    return {
        "ok": True,
        "message": "",
        "data": payload,
    }


@st.cache_data(ttl=86400, show_spinner=False)
def market_catalog() -> dict[str, Any]:
    result = _api_get(
        "/markets",
        {
            "sportId": SOCCER_SPORT_ID,
            "language": "en",
        },
        timeout=60,
    )
    if not result.get("ok"):
        return {
            "ok": False,
            "message": result.get("message", "No se pudo cargar catálogo."),
            "markets": [],
            "by_id": {},
        }

    raw = result.get("data") or []
    # El catálogo de OddsPapi puede incluir mercados globales.
    markets = [
        m for m in raw
        if int(m.get("sportId") or SOCCER_SPORT_ID) == SOCCER_SPORT_ID
    ]

    by_id = {str(m.get("marketId")): m for m in markets}
    return {
        "ok": True,
        "message": "",
        "markets": markets,
        "by_id": by_id,
    }


def _wanted_market(
    catalog: dict[str, Any],
    family: str,
    line: float | None = None,
) -> dict[str, Any] | None:
    exact_names = {
        "1x2": {"Full Time Result"},
        "goals": {"Over Under Full Time"},
        "corners": {"Corners - Over Under Full Time", "Corners Over Under Full Time"},
        "cards": {"Bookings - Over Under Full Time", "Bookings Over Under Full Time"},
    }
    wanted_names = exact_names.get(family, set())

    candidates = []
    for market in catalog.get("markets", []):
        name = str(market.get("marketName") or "")
        if name not in wanted_names:
            continue

        period = str(market.get("period") or "fulltime").lower()
        if period not in {"fulltime", "full_time", "ft", ""}:
            continue

        if family != "1x2":
            hcap = market.get("handicap")
            try:
                hcap = float(hcap)
            except Exception:
                continue
            if line is None or abs(hcap - float(line)) > 1e-9:
                continue

        candidates.append(market)

    if not candidates:
        return None

    # Si hubiera más de una variante, preferimos mercado estándar sin player prop.
    candidates.sort(
        key=lambda m: (
            bool(m.get("playerProp", False)),
            int(m.get("marketId") or 10**9),
        )
    )
    return candidates[0]


def _outcome_name(
    market_def: dict[str, Any],
    outcome_id: str,
) -> str:
    for outcome in market_def.get("outcomes", []) or []:
        if str(outcome.get("outcomeId")) == str(outcome_id):
            return str(outcome.get("outcomeName") or outcome_id)
    return str(outcome_id)


def _extract_quote(outcome: dict[str, Any]) -> float | None:
    players = outcome.get("players", {}) or {}
    quote = players.get("0")

    if isinstance(quote, list):
        # Compatibilidad defensiva.
        quote = next(
            (
                x for x in reversed(quote)
                if isinstance(x, dict) and x.get("active") is not False
            ),
            None,
        )

    if not isinstance(quote, dict):
        return None
    if quote.get("active") is False:
        return None

    try:
        price = float(quote.get("price"))
    except Exception:
        return None

    return price if price > 1.0 else None


def _market_prices(
    book: dict[str, Any],
    market_def: dict[str, Any],
) -> dict[str, float]:
    if not isinstance(book, dict):
        return {}
    if book.get("bookmakerIsActive") is False:
        return {}
    if book.get("suspended") is True:
        return {}

    market_id = str(market_def.get("marketId"))
    market = (book.get("markets") or {}).get(market_id)
    if not isinstance(market, dict):
        return {}
    if market.get("marketActive") is False:
        return {}

    out = {}
    for outcome_id, outcome in (market.get("outcomes") or {}).items():
        price = _extract_quote(outcome)
        if price is None:
            continue
        label = _outcome_name(market_def, str(outcome_id))
        out[label] = price

    return out


def _pick_over(prices: dict[str, float]) -> float | None:
    for label, price in prices.items():
        n = _norm(label)
        if n == "over" or n.startswith("over "):
            return price
    # OddsPapi puede devolver el outcome id como fallback; para mercados
    # O/U el primer outcome del catálogo es el Over.
    return None


def _pick_1x2(prices: dict[str, float], key: str) -> float | None:
    wanted = {
        "home": {"1", "home"},
        "draw": {"x", "draw"},
        "away": {"2", "away"},
    }.get(key, set())

    for label, price in prices.items():
        if _norm(label) in wanted:
            return price
    return None


@st.cache_data(ttl=300, show_spinner=False)
def discover_fixtures(
    kickoff_iso: str,
    selected_bookmakers: tuple[str, ...],
) -> dict[str, Any]:
    kickoff = pd.to_datetime(kickoff_iso, utc=True, errors="coerce")
    if pd.isna(kickoff):
        return {"ok": False, "message": "Hora inválida.", "fixtures": []}

    # Una ventana pequeña reduce ruido y consumo.
    start = (kickoff - pd.Timedelta(hours=18)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (kickoff + pd.Timedelta(hours=18)).strftime("%Y-%m-%dT%H:%M:%SZ")

    result = _api_get(
        "/fixtures",
        {
            "sportId": SOCCER_SPORT_ID,
            "from": start,
            "to": end,
            "statusId": 0,
            "hasOdds": "true",
            "bookmakers": _bookmaker_param(selected_bookmakers),
            "language": "en",
        },
    )

    return {
        "ok": result.get("ok", False),
        "message": result.get("message", ""),
        "fixtures": result.get("data") or [],
    }


def _find_fixture(
    fixtures: list[dict[str, Any]],
    match: dict[str, Any],
) -> dict[str, Any] | None:
    target = pd.to_datetime(match.get("utc_time"), utc=True, errors="coerce")
    best = None
    best_score = -1.0

    for fixture in fixtures:
        home = (
            fixture.get("participant1Name")
            or fixture.get("participant1ShortName")
            or ""
        )
        away = (
            fixture.get("participant2Name")
            or fixture.get("participant2ShortName")
            or ""
        )

        direct = (
            _team_score(match.get("home_team", ""), home)
            + _team_score(match.get("away_team", ""), away)
        ) / 2.0
        swapped = (
            _team_score(match.get("home_team", ""), away)
            + _team_score(match.get("away_team", ""), home)
        ) / 2.0

        score = max(direct, swapped)
        if score < 0.76:
            continue

        ftime = pd.to_datetime(
            fixture.get("startTime"),
            utc=True,
            errors="coerce",
        )
        if not pd.isna(target) and not pd.isna(ftime):
            hours = abs((ftime - target).total_seconds()) / 3600.0
            if hours > 12:
                continue
            score += max(0.0, 0.12 - hours * 0.01)

        if score > best_score:
            best_score = score
            best = fixture

    return best


@st.cache_data(ttl=300, show_spinner=False)
def fetch_fixture_odds(
    fixture_id: str,
    selected_bookmakers: tuple[str, ...],
) -> dict[str, Any]:
    result = _api_get(
        "/odds",
        {
            "fixtureId": fixture_id,
            "bookmakers": _bookmaker_param(selected_bookmakers),
            "language": "en",
            "verbosity": 3,
            "oddsFormat": "decimal",
        },
        timeout=45,
    )
    return {
        "ok": result.get("ok", False),
        "message": result.get("message", ""),
        "event": result.get("data"),
    }


def _line_from_row(row: dict[str, Any]) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", str(row.get("label", "")))
    return float(m.group(1)) if m else None


def _canonical_book(slug: str) -> str | None:
    s = str(slug or "").lower()
    if s in {"bet365.es", "bet365"}:
        return "Bet365"
    if s in {"winamax.es", "winamax.fr", "winamax_fr", "winamax"}:
        return "Winamax"
    if s == "pinnacle":
        return "Pinnacle"
    return None



def _sot_tokens(name: str) -> bool:
    n = _norm(name)
    return (
        ("shot" in n and "target" in n)
        or "shots on goal" in n
        or "shot on goal" in n
    )


def _market_line(market: dict[str, Any]) -> float | None:
    try:
        return float(market.get("handicap"))
    except Exception:
        return None


def _sot_side_from_key(key: str) -> str:
    key = str(key or "")
    if key.startswith("home_"):
        return "home"
    if key.startswith("away_"):
        return "away"
    return "total"


def _sot_market_score(
    market: dict[str, Any],
    row: dict[str, Any],
    home_team: str,
    away_team: str,
) -> float:
    """
    Busca SOLO mercados SOT de partido/equipo.
    Los player props se descartan para no cruzarlos con nuestro modelo.
    """
    name = str(market.get("marketName") or "")
    mtype = str(market.get("marketType") or "")
    text = _norm(f"{name} {mtype}")

    if not _sot_tokens(text):
        return -999.0

    # Nuestro modelo NO es de jugadores.
    if bool(market.get("playerProp", False)):
        return -999.0

    period = str(market.get("period") or "fulltime").lower()
    if period not in {"fulltime", "full_time", "ft", ""}:
        return -999.0

    line = _line_from_row(row)
    hcap = _market_line(market)
    if line is None or hcap is None or abs(float(line) - hcap) > 1e-9:
        return -999.0

    side = _sot_side_from_key(str(row.get("key")))
    home_n = _norm(home_team)
    away_n = _norm(away_team)

    home_markers = (
        "home", "team 1", "team1", "participant 1", "participant1",
        "1st team", "first team"
    )
    away_markers = (
        "away", "team 2", "team2", "participant 2", "participant2",
        "2nd team", "second team"
    )

    has_home = any(x in text for x in home_markers) or (home_n and home_n in text)
    has_away = any(x in text for x in away_markers) or (away_n and away_n in text)
    has_team = "team" in text or has_home or has_away

    score = 10.0

    # Prefer explicit Over/Under totals.
    if "over under" in text or "over/under" in name.lower():
        score += 3.0

    if side == "total":
        # Total de partido: evitar mercados explícitos de un equipo.
        if has_home or has_away:
            return -999.0
        if "total" in text or not has_team:
            score += 5.0

    elif side == "home":
        if has_away:
            return -999.0
        if has_home:
            score += 8.0
        elif "team" in text:
            score += 2.0
        else:
            return -999.0

    elif side == "away":
        if has_home:
            return -999.0
        if has_away:
            score += 8.0
        elif "team" in text:
            score += 2.0
        else:
            return -999.0

    return score


def _wanted_sot_market(
    catalog: dict[str, Any],
    row: dict[str, Any],
    home_team: str,
    away_team: str,
) -> dict[str, Any] | None:
    ranked = []
    for market in catalog.get("markets", []):
        score = _sot_market_score(market, row, home_team, away_team)
        if score > -900:
            ranked.append((score, int(market.get("marketId") or 10**9), market))

    if not ranked:
        return None

    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked[0][2]


def _sot_catalog_candidates(
    catalog: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Devuelve sólo mercados SOT que realmente aparecen en el fixture,
    para diagnóstico visible si el proveedor no ofrece total/equipo.
    """
    present_ids = set()
    for book in (payload.get("bookmakerOdds") or {}).values():
        if not isinstance(book, dict):
            continue
        for mid in (book.get("markets") or {}).keys():
            present_ids.add(str(mid))

    out = []
    by_id = catalog.get("by_id", {}) or {}
    for mid in sorted(present_ids):
        market = by_id.get(str(mid))
        if not isinstance(market, dict):
            continue
        if not _sot_tokens(str(market.get("marketName") or "")):
            continue
        out.append({
            "marketId": market.get("marketId"),
            "marketName": market.get("marketName"),
            "handicap": market.get("handicap"),
            "period": market.get("period"),
            "playerProp": bool(market.get("playerProp", False)),
        })
    return out


def parse_prices(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    selected_bookmakers: tuple[str, ...],
    catalog: dict[str, Any],
    home_team: str,
    away_team: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    books = payload.get("bookmakerOdds", {}) or {}
    selected = set(selected_bookmakers[:3])

    for slug, book in books.items():
        cname = _canonical_book(slug)
        if cname is None or cname not in selected:
            continue

        for row in rows:
            cat = row.get("cat")

            line = _line_from_row(row)
            if cat == "sot":
                market_def = _wanted_sot_market(
                    catalog,
                    row,
                    home_team,
                    away_team,
                )
            else:
                market_def = _wanted_market(
                    catalog,
                    cat,
                    None if cat == "1x2" else line,
                )
            if not market_def:
                continue

            prices = _market_prices(book, market_def)
            if not prices:
                continue

            if cat == "1x2":
                price = _pick_1x2(prices, str(row.get("key")))
            else:
                price = _pick_over(prices)

            if price is not None:
                out.setdefault(row["id"], {})[cname] = float(price)

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
            "message": "Partido iniciado: no se consultan cuotas live.",
            "prices": {},
        }

    catalog = market_catalog()
    if not catalog.get("ok"):
        return {
            "ok": False,
            "message": catalog.get("message", "No se pudo cargar mercados."),
            "prices": {},
        }

    discovery = discover_fixtures(
        str(kickoff.isoformat()),
        tuple(selected_bookmakers[:3]),
    )
    if not discovery.get("ok"):
        return {
            "ok": False,
            "message": discovery.get("message", "No se pudieron buscar partidos."),
            "prices": {},
        }

    fixture = _find_fixture(
        discovery.get("fixtures", []),
        match,
    )
    if fixture is None:
        return {
            "ok": False,
            "message": "OddsPapi todavía no ha publicado este partido en Bet365/Winamax/Pinnacle.",
            "prices": {},
        }

    fixture_id = str(fixture.get("fixtureId"))
    result = fetch_fixture_odds(
        fixture_id,
        tuple(selected_bookmakers[:3]),
    )
    if not result.get("ok"):
        return {
            "ok": False,
            "message": result.get("message", "No se pudieron cargar cuotas."),
            "prices": {},
        }

    payload = result.get("event") or {}
    status_name = str(payload.get("statusName") or "").lower()
    if status_name and status_name not in {"pre-game", "pregame", "scheduled", "not started"}:
        return {
            "ok": False,
            "locked": True,
            "message": "OddsPapi marca el partido como iniciado/no prepartido. Cuotas bloqueadas.",
            "prices": {},
        }

    prices = parse_prices(
        payload,
        market_rows,
        tuple(selected_bookmakers[:3]),
        catalog,
        str(match.get("home_team", "")),
        str(match.get("away_team", "")),
    )

    available_books = []
    for slug in (payload.get("bookmakerOdds") or {}).keys():
        cname = _canonical_book(slug)
        if cname and cname not in available_books:
            available_books.append(cname)

    return {
        "ok": True,
        "message": "",
        "prices": prices,
        "provider": "OddsPapi",
        "fixture_id": fixture_id,
        "updated": payload.get("updatedAt"),
        "available_books": available_books,
        "selected_books": list(selected_bookmakers[:3]),
        "cards_note": (
            "Tarjetas usa el mercado 'Bookings - Over Under Full Time'. "
            "Comprueba siempre las reglas de liquidación de la casa."
        ),
        "side_market_note": (
            "Córners y tarjetas suelen abrir más cerca del inicio; "
            "si aún no aparecen, la línea puede no estar abierta todavía."
        ),
        "sot_candidates": _sot_catalog_candidates(catalog, payload),
        "sot_auto_rows": sum(
            1 for row_id in prices
            if "|sot|" in str(row_id)
        ),
    }
