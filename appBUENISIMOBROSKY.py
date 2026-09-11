from datetime import datetime

import pandas as pd
import streamlit as st

from api_football import (
    APIFootballError,
    get_api_key,
    get_current_round,
    get_current_season,
    get_eliteserien_league,
    get_round_fixtures,
)
from config import APP_NAME, DATA_START_SEASON
from database import get_database_summary, init_db, upsert_matches
from sync_data import sync_matches_from_2023


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🇳🇴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.2rem 1.35rem;
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 18px;
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2rem;
        }

        .hero p {
            margin: .35rem 0 0 0;
            opacity: .72;
        }

        .fixture-card {
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: .8rem;
        }

        .muted {
            opacity: .65;
            font-size: .92rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.20);
            padding: .75rem;
            border-radius: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

st.markdown(
    f"""
    <div class="hero">
        <h1>🇳🇴 {APP_NAME}</h1>
        <p>Eliteserien · datos de equipos desde {DATA_START_SEASON} · árbitros: histórico disponible</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not get_api_key():
    st.error("Falta configurar tu API key de API-Football.")
    st.code('API_FOOTBALL_KEY="TU_API_KEY"', language="text")
    st.info(
        "En local puedes copiar .env.example como .env. "
        "En Streamlit Cloud usa Secrets."
    )
    st.stop()

try:
    league_info = get_eliteserien_league()
    season = get_current_season()
    current_round = get_current_round(season)

except APIFootballError as exc:
    st.error(str(exc))
    st.stop()

league = league_info.get("league", {})

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Liga", league.get("name", "Eliteserien"))

with col2:
    st.metric("Temporada", season)

with col3:
    st.metric("Jornada", current_round or "—")

with col4:
    st.metric("Histórico equipos", f"{DATA_START_SEASON} → {season}")

st.divider()

tab_jornada, tab_picks, tab_datos = st.tabs(
    ["⚽ Jornada", "💎 Mejores picks", "🗄️ Datos"]
)

with tab_jornada:
    st.subheader("Partidos de la jornada")

    if not current_round:
        st.warning("La API no indica una jornada activa en este momento.")
    else:
        try:
            fixtures = get_round_fixtures(season, current_round)
            upsert_matches(fixtures)
        except APIFootballError as exc:
            st.error(str(exc))
            fixtures = []

        if not fixtures:
            st.info("No hay partidos encontrados para la jornada actual.")

        for item in fixtures:
            fixture = item.get("fixture", {})
            teams = item.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            status = fixture.get("status", {})
            league_data = item.get("league", {})

            kickoff = fixture.get("date", "")
            try:
                dt = datetime.fromisoformat(kickoff)
                kickoff_text = dt.strftime("%d/%m · %H:%M")
            except Exception:
                kickoff_text = kickoff or "Hora por confirmar"

            referee = fixture.get("referee") or "Árbitro todavía sin asignar"
            venue = (fixture.get("venue") or {}).get("name") or "Estadio por confirmar"

            st.markdown(
                f"""
                <div class="fixture-card">
                    <div class="muted">{kickoff_text} · {venue}</div>
                    <h3 style="margin:.45rem 0">{home.get("name","Local")} vs {away.get("name","Visitante")}</h3>
                    <div class="muted">👨‍⚖️ {referee} · Estado: {status.get("short","—")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            cols = st.columns([1, 1, 2])
            with cols[0]:
                if st.button(
                    "Analizar",
                    key=f"analyse_{fixture.get('id')}",
                    use_container_width=True,
                ):
                    st.info(
                        "El motor de predicción será el siguiente módulo: "
                        "goles, córners, tarjetas, árbitro y EV."
                    )

            with cols[1]:
                if st.button(
                    "Ver datos",
                    key=f"data_{fixture.get('id')}",
                    use_container_width=True,
                ):
                    st.json(
                        {
                            "fixture_id": fixture.get("id"),
                            "round": league_data.get("round"),
                            "referee": referee,
                            "home": home.get("name"),
                            "away": away.get("name"),
                        }
                    )

with tab_picks:
    st.subheader("💎 Mejores picks de la jornada")
    st.info(
        "Aquí aparecerá el ranking automático cuando conectemos los modelos. "
        "No recomendaremos una apuesta solo por tener alta probabilidad: "
        "también exigiremos valor esperado positivo."
    )

    placeholder = pd.DataFrame(
        [
            {
                "Partido": "—",
                "Mercado": "Esperando modelos",
                "Prob. modelo": "—",
                "Cuota": "—",
                "EV": "—",
            }
        ]
    )
    st.dataframe(placeholder, use_container_width=True, hide_index=True)

with tab_datos:
    st.subheader("🗄️ Base de datos")

    summary = get_database_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Partidos guardados", summary["matches"])
    c2.metric("Finalizados", summary["finished"])
    c3.metric("Árbitros detectados", summary["referees"])

    st.caption(
        "La primera descarga guarda calendario/resultados. "
        "Después añadiremos estadísticas detalladas partido a partido "
        "(córners, tarjetas, tiros, faltas, etc.) de forma controlada."
    )

    if st.button(
        f"⬇️ Sincronizar partidos desde {DATA_START_SEASON}",
        type="primary",
        use_container_width=True,
    ):
        progress = st.progress(0)
        status_box = st.empty()

        try:
            def report(value, text):
                progress.progress(min(max(float(value), 0.0), 1.0))
                status_box.write(text)

            total, seasons = sync_matches_from_2023(report)
            progress.progress(1.0)
            status_box.success(
                f"Sincronización terminada: {total} registros procesados "
                f"({', '.join(map(str, seasons))})."
            )
            st.rerun()

        except APIFootballError as exc:
            st.error(str(exc))
