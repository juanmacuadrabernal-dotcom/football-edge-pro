
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


BASE_URL = "https://v3.football.api-sports.io"
PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_BOOKMAKERS = ("Bet365", "Unibet", "William Hill")


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
        return ""
    return ""


def get_api_key() -> str:
    value = os.getenv("API_FOOTBALL_KEY", "").strip()
    if value:
        return value

    try:
        value = str(st.secrets.get("API_FOOTBALL_KEY", "")).strip()
        if value:
            return value
    except Exception:
        pass

    return _read_dotenv_value("API_FOOTBALL_KEY")


def _norm(value: Any) -> str:
    s = str(value or "").lower().strip()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"[^a-z0-9]+", " ", s)
    tokens = [
        t for t in s.split()
        if t not in {
            "fc", "cf", "cd", "ud", "sad", "club",
            "de", "la", "el", "real", "football", "futbol"
        }
    ]
    aliases = {
        "ath madrid": "atletico madrid",
        "atletico de madrid": "atletico madrid",
        "ath bilbao": "athletic bilbao",
        "athletic club": "athletic bilbao",
        "sociedad": "real sociedad",
        "rayo": "rayo vallecano",
        "vallecano": "rayo vallecano",
        "espanol": "espanyol",
        "deportivo a coruna": "deportivo coruna",
        "deportivo la coruna": "deportivo coruna",
        "deportivo de la coruna": "deportivo coruna",
    }
    out = " ".join(tokens)
    return aliases.get(out, out)


def _team_score(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    jacc = len(ta & tb) / max(1, len(ta | tb))
    seq = SequenceMatcher(None, na, nb).ratio()
    if na in nb or nb in na:
        seq = max(seq, 0.92)
    return max(seq, jacc)


def _api_get(path: str, params: dict[str, Any], timeout: int = 25) -> dict[str, Any]:
    key = get_api_key()
    if not key:
        return {
            "ok": False,
            "error": "MISSING_KEY",
            "message": "Falta API_FOOTBALL_KEY.",
            "response": [],
        }

    try:
        r = requests.get(
            BASE_URL + path,
            headers={"x-apisports-key": key},
            params=params,
            timeout=timeout,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": "NETWORK",
            "message": str(exc),
            "response": [],
        }

    try:
        payload = r.json()
    except Exception:
        payload = {}

    if r.status_code != 200:
        return {
            "ok": False,
            "error": f"HTTP_{r.status_code}",
            "message": str(payload)[:500],
            "response": [],
        }

    errors = payload.get("errors")
    if errors:
        return {
            "ok": False,
            "error": "API_ERROR",
            "message": str(errors),
            "response": payload.get("response", []),
        }

    return {
        "ok": True,
        "response": payload.get("response", []),
        "paging": payload.get("paging", {}),
        "results": payload.get("results", 0),
        "headers": {
            "remaining": r.headers.get("x-ratelimit-requests-remaining")
                         or r.headers.get("x-requests-remaining"),
        },
    }


@st.cache_data(ttl=86400, show_spinner=False)
def _resolve_eliteserien_id() -> int | None:
    result = _api_get(
        "/leagues",
        {"country": "Norway", "search": "Eliteserien", "current": "true"},
    )
    if not result.get("ok"):
        return None
    for item in result.get("response", []):
        league = item.get("league", {})
        if "eliteserien" in str(league.get("name", "")).lower():
            try:
                return int(league["id"])
            except Exception:
                pass
    return None


def league_id(league_key: str) -> int | None:
    if league_key == "laliga":
        return 140
    if league_key == "eliteserien":
        return _resolve_eliteserien_id()
    return None


def season_from_match(league_key: str, kickoff_value: Any) -> int:
    ts = pd.Timestamp(kickoff_value)
    if league_key == "laliga":
        return ts.year if ts.month >= 7 else ts.year - 1
    return ts.year


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_league_prematch_odds(
    league_key: str,
    season: int,
) -> dict[str, Any]:
    lid = league_id(league_key)
    if lid is None:
        return {
            "ok": False,
            "message": "No se pudo resolver el ID de la liga en API-Football.",
            "events": [],
        }

    all_events = []
    page = 1
    remaining = None

    while page <= 4:
        result = _api_get(
            "/odds",
            {
                "league": lid,
                "season": season,
                "page": page,
            },
        )
        if not result.get("ok"):
            return {
                "ok": False,
                "message": result.get("message", "Error consultando cuotas."),
                "events": all_events,
                "remaining": remaining,
            }

        all_events.extend(result.get("response", []))
        remaining = result.get("headers", {}).get("remaining")

        paging = result.get("paging", {}) or {}
        current = int(paging.get("current", page) or page)
        total = int(paging.get("total", current) or current)
        if current >= total:
            break
        page += 1

    return {
        "ok": True,
        "message": "",
        "events": all_events,
        "remaining": remaining,
    }


def _fixture_info(event: dict[str, Any]) -> tuple[str, str, pd.Timestamp | None]:
    fixture = event.get("fixture", {}) or {}
    home = fixture.get("home", {}) or {}
    away = fixture.get("away", {}) or {}

    # API-Football odds payloads can vary slightly; support both shapes.
    home_name = home.get("name") or event.get("home_team") or ""
    away_name = away.get("name") or event.get("away_team") or ""

    if not home_name or not away_name:
        teams = fixture.get("teams", {}) or {}
        home_name = home_name or (teams.get("home", {}) or {}).get("name", "")
        away_name = away_name or (teams.get("away", {}) or {}).get("name", "")

    date_value = (
        fixture.get("date")
        or event.get("commence_time")
        or event.get("update")
    )
    try:
        ts = pd.to_datetime(date_value, utc=True, errors="coerce")
        if pd.isna(ts):
            ts = None
    except Exception:
        ts = None

    return str(home_name), str(away_name), ts


def find_event_for_match(
    events: list[dict[str, Any]],
    home_team: str,
    away_team: str,
    kickoff_value: Any,
) -> dict[str, Any] | None:
    target_ts = pd.to_datetime(kickoff_value, utc=True, errors="coerce")
    best = None
    best_score = -1.0

    for event in events:
        eh, ea, ets = _fixture_info(event)
        direct = (_team_score(home_team, eh) + _team_score(away_team, ea)) / 2
        swapped = (_team_score(home_team, ea) + _team_score(away_team, eh)) / 2
        score = max(direct, swapped)

        if score < 0.72:
            continue

        if ets is not None and not pd.isna(target_ts):
            hours = abs((ets - target_ts).total_seconds()) / 3600
            if hours > 18:
                continue
            score += max(0.0, 0.08 - hours * 0.004)

        if score > best_score:
            best_score = score
            best = event

    return best


def _line_from_row(row: dict[str, Any]) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", str(row.get("label", "")))
    return float(m.group(1)) if m else None


def _is_over_value(value: str, line: float) -> bool:
    v = _norm(value)
    if "over" not in v and "mas" not in v:
        return False
    nums = re.findall(r"\d+(?:\.\d+)?", str(value))
    return any(abs(float(x) - line) < 1e-6 for x in nums)


def _book_allowed(book_name: str, selected: tuple[str, ...]) -> str | None:
    n = _norm(book_name)
    for wanted in selected:
        w = _norm(wanted)
        if n == w or w in n or n in w:
            return wanted
    return None


def _bet_matches_category(bet_name: str, row: dict[str, Any]) -> bool:
    n = _norm(bet_name)
    cat = row.get("cat")

    if cat == "goals":
        if any(x in n for x in ("corner", "card", "booking", "shot")):
            return False
        return (
            "goal" in n
            or "over under" in n
            or "total" in n
        )

    if cat == "corners":
        return "corner" in n

    if cat == "cards":
        return "card" in n or "booking" in n

    if cat == "sot":
        return "shot" in n and ("target" in n or "goal" in n)

    if cat == "1x2":
        return (
            "match winner" in n
            or "winner" == n
            or "1x2" in n
        )

    return False


def _outcome_for_1x2(
    row: dict[str, Any],
    value: str,
    home_team: str,
    away_team: str,
) -> bool:
    v = _norm(value)
    key = row.get("key")
    if key == "draw":
        return v in {"draw", "x"} or "draw" in v
    if key == "home":
        return v in {"home", "1"} or _team_score(value, home_team) >= 0.82
    if key == "away":
        return v in {"away", "2"} or _team_score(value, away_team) >= 0.82
    return False


def _sot_team_match(
    row: dict[str, Any],
    bet_name: str,
    value: str,
    home_team: str,
    away_team: str,
) -> bool:
    key = str(row.get("key", ""))
    if key.startswith("total_"):
        return True

    target_team = home_team if key.startswith("home_") else away_team
    combined = f"{bet_name} {value}"
    return _team_score(target_team, combined) >= 0.55 or _norm(target_team) in _norm(combined)


def prices_for_row(
    event: dict[str, Any],
    row: dict[str, Any],
    home_team: str,
    away_team: str,
    selected_bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
) -> dict[str, float]:
    result: dict[str, float] = {}
    line = _line_from_row(row)

    for bookmaker in event.get("bookmakers", []) or []:
        canonical = _book_allowed(
            str(bookmaker.get("name", "")),
            selected_bookmakers,
        )
        if not canonical:
            continue

        for bet in bookmaker.get("bets", []) or []:
            bet_name = str(bet.get("name", ""))
            if not _bet_matches_category(bet_name, row):
                continue

            for item in bet.get("values", []) or []:
                value = str(item.get("value", ""))
                try:
                    odd = float(str(item.get("odd", "")).replace(",", "."))
                except Exception:
                    continue
                if odd <= 1.0:
                    continue

                matched = False
                if row.get("cat") == "1x2":
                    matched = _outcome_for_1x2(row, value, home_team, away_team)
                elif line is not None:
                    matched = _is_over_value(value, line)
                    if matched and row.get("cat") == "sot":
                        matched = _sot_team_match(
                            row, bet_name, value, home_team, away_team
                        )

                if matched:
                    # Keep the best current price for that bookmaker.
                    result[canonical] = max(result.get(canonical, 0.0), odd)

    return result


def populate_match_odds(
    league_key: str,
    match: dict[str, Any],
    market_rows: list[dict[str, Any]],
    selected_bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
) -> dict[str, Any]:
    kickoff = pd.to_datetime(match.get("utc_time"), utc=True, errors="coerce")
    if pd.isna(kickoff):
        return {"ok": False, "message": "Hora de partido inválida.", "prices": {}}

    now_utc = pd.Timestamp.now(tz="UTC")
    if kickoff <= now_utc:
        return {
            "ok": False,
            "message": "Partido iniciado: no se consultan cuotas live.",
            "prices": {},
            "locked": True,
        }

    season = season_from_match(league_key, kickoff)
    bundle = fetch_league_prematch_odds(league_key, season)
    if not bundle.get("ok"):
        return {
            "ok": False,
            "message": bundle.get("message", "No se pudieron cargar cuotas."),
            "prices": {},
        }

    event = find_event_for_match(
        bundle.get("events", []),
        str(match.get("home_team", "")),
        str(match.get("away_team", "")),
        kickoff,
    )
    if event is None:
        return {
            "ok": False,
            "message": "API-Football todavía no tiene cuotas para este partido.",
            "prices": {},
            "remaining": bundle.get("remaining"),
        }

    prices: dict[str, dict[str, float]] = {}
    for row in market_rows:
        row_prices = prices_for_row(
            event,
            row,
            str(match.get("home_team", "")),
            str(match.get("away_team", "")),
            selected_bookmakers=selected_bookmakers,
        )
        if row_prices:
            prices[row["id"]] = row_prices

    return {
        "ok": True,
        "message": "",
        "prices": prices,
        "remaining": bundle.get("remaining"),
        "bookmakers": list(selected_bookmakers),
        "updated": event.get("update"),
    }
