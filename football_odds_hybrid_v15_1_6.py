
from __future__ import annotations

from football_odds_oddspapi_v15_1_4 import (
    populate_match_odds as populate_oddspapi,
    get_api_key,
    DEFAULT_BOOKMAKERS,
)
from football_sot_sportsgameodds_v15_1_6 import (
    populate_sot_odds,
    get_sgo_api_key,
)


def populate_match_odds(
    league_key,
    match,
    market_rows,
    selected_bookmakers=DEFAULT_BOOKMAKERS,
):
    normal_rows = [
        row for row in market_rows
        if row.get("cat") != "sot"
    ]
    sot_rows = [
        row for row in market_rows
        if row.get("cat") == "sot"
    ]

    normal = populate_oddspapi(
        league_key,
        match,
        normal_rows,
        selected_bookmakers=selected_bookmakers,
    )

    combined = {
        "ok": bool(normal.get("ok")),
        "locked": bool(normal.get("locked", False)),
        "message": normal.get("message", ""),
        "prices": dict(normal.get("prices", {}) or {}),
        "provider": "OddsPapi + SportsGameOdds",
        "available_books": normal.get("available_books", []),
        "selected_books": normal.get("selected_books", list(selected_bookmakers)),
    }

    if sot_rows:
        if get_sgo_api_key():
            sot = populate_sot_odds(league_key, match, sot_rows)
            if sot.get("locked"):
                combined["locked"] = True

            for row_id, book_prices in (sot.get("prices") or {}).items():
                combined["prices"].setdefault(row_id, {}).update(book_prices)

            combined["sot_ok"] = bool(sot.get("ok"))
            combined["sot_message"] = sot.get("message", "")
            combined["sot_provider"] = sot.get("provider", "SportsGameOdds")
            combined["sot_books"] = sot.get("books", [])
            combined["sot_note"] = sot.get("note", "")
        else:
            combined["sot_ok"] = False
            combined["sot_message"] = "Falta SPORTSGAMEODDS_API_KEY."
            combined["sot_note"] = (
                "Para SOT automático falta configurar SportsGameOdds. "
                "Goles, tarjetas, córners y 1X2 siguen funcionando con OddsPapi."
            )

    return combined
