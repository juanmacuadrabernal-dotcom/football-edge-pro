from __future__ import annotations

import pandas as pd
import streamlit as st

from prediction_engine import PredictionEngine


st.set_page_config(
    page_title="Eliteserien Edge Pro",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource
def get_engine():
    return PredictionEngine()


@st.cache_data(ttl=300)
def get_predictions():
    engine = get_engine()
    return engine.predict_next_round()


def fair_odds(prob):
    if prob <= 0:
        return None
    return 1.0 / prob


def ev_value(prob, odds):
    if odds is None or odds <= 1.0:
        return None
    return prob * odds - 1.0


def ev_label(ev, min_ev):
    if ev is None:
        return "⚪ Introduce cuota"

    if ev >= min_ev:
        return "🟢 BET candidato"

    if ev >= 0:
        return "🟡 Valor insuficiente"

    return "🔴 NO BET"


st.title("⚽ Eliteserien Edge Pro")
st.caption(
    "Modelos validados con backtesting walk-forward. "
    "El 1X2 se muestra solo como contexto porque no superó "
    "las cuotas de cierre en el backtest de ROI."
)

with st.sidebar:
    st.header("Filtros")

    min_ev_percent = st.slider(
        "EV mínimo para BET",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
    )

    min_ev = min_ev_percent / 100.0

    st.markdown(
        "**Mercados activos**  \n"
        "⚽ Over 2.5 goles  \n"
        "🚩 Over 9.5 / 10.5 / 11.5 córners  \n"
        "🟨 Over 3.5 / 4.5 / 5.5 amarillas"
    )

try:
    predictions = get_predictions()
except Exception as exc:
    st.error(
        "No he podido cargar el motor de predicción."
    )
    st.exception(exc)
    st.stop()

if not predictions:
    st.warning(
        "No encuentro partidos futuros en fotmob_matches. "
        "Ejecuta primero el actualizador de FotMob."
    )
    st.stop()

first = predictions[0]

round_text = (
    str(first["round"])
    if pd.notna(first["round"])
    else "próxima"
)

st.subheader(
    f"Jornada {round_text} · {len(predictions)} partidos"
)

tab_matches, tab_best, tab_1x2 = st.tabs(
    [
        "📊 Jornada",
        "⭐ Mejores picks",
        "ℹ️ 1X2 contexto",
    ]
)

entered_picks = []

MARKETS = [
    (
        "goals",
        "over_2_5",
        "⚽ Over 2.5 goles",
        "FUERTE",
    ),
    (
        "corners",
        "over_9_5",
        "🚩 Over 9.5 córners",
        "APROBADO",
    ),
    (
        "corners",
        "over_10_5",
        "🚩 Over 10.5 córners",
        "FUERTE",
    ),
    (
        "corners",
        "over_11_5",
        "🚩 Over 11.5 córners",
        "FUERTE",
    ),
    (
        "cards",
        "over_3_5",
        "🟨 Over 3.5 amarillas",
        "SECUNDARIO",
    ),
    (
        "cards",
        "over_4_5",
        "🟨 Over 4.5 amarillas",
        "APROBADO",
    ),
    (
        "cards",
        "over_5_5",
        "🟨 Over 5.5 amarillas",
        "FUERTE",
    ),
]


with tab_matches:
    for match in predictions:
        kickoff = pd.Timestamp(
            match["utc_time"]
        ).tz_convert("Europe/Madrid")

        with st.container(border=True):
            st.markdown(
                f"### {match['home_team']} — {match['away_team']}"
            )

            info1, info2, info3 = st.columns(3)

            info1.caption(
                kickoff.strftime("%d/%m/%Y · %H:%M")
            )

            referee = match.get("referee")

            info2.caption(
                f"Árbitro: {referee}"
                if referee is not None and not pd.isna(referee)
                else "Árbitro: pendiente"
            )

            g = match["goals"]

            info3.caption(
                f"xG-model goles: "
                f"{g['lambda_home']:.2f} - {g['lambda_away']:.2f}"
            )

            header = st.columns(
                [3.1, 1.2, 1.2, 1.4, 1.8]
            )

            header[0].markdown("**Mercado**")
            header[1].markdown("**Modelo**")
            header[2].markdown("**Cuota justa**")
            header[3].markdown("**Cuota mercado**")
            header[4].markdown("**Decisión**")

            for category, key, label, strength in MARKETS:
                prob = float(
                    match[category][key]
                )

                fair = fair_odds(prob)

                cols = st.columns(
                    [3.1, 1.2, 1.2, 1.4, 1.8]
                )

                cols[0].write(
                    f"{label} · {strength}"
                )

                cols[1].write(
                    f"{prob*100:.1f}%"
                )

                cols[2].write(
                    f"{fair:.2f}"
                    if fair is not None
                    else "—"
                )

                widget_key = (
                    f"odds_{match['match_id']}_{category}_{key}"
                )

                odds = cols[3].number_input(
                    "Cuota",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.01,
                    key=widget_key,
                    label_visibility="collapsed",
                )

                ev = ev_value(
                    prob,
                    odds,
                )

                cols[4].write(
                    ev_label(
                        ev,
                        min_ev,
                    )
                )

                if ev is not None:
                    cols[4].caption(
                        f"EV {ev*100:+.1f}%"
                    )

                    entered_picks.append(
                        {
                            "match_id": match["match_id"],
                            "partido": (
                                f"{match['home_team']} - "
                                f"{match['away_team']}"
                            ),
                            "mercado": label,
                            "strength": strength,
                            "prob": prob,
                            "fair_odds": fair,
                            "market_odds": odds,
                            "ev": ev,
                            "approved": ev >= min_ev,
                        }
                    )


with tab_best:
    valid = [
        p for p in entered_picks
        if p["approved"]
    ]

    if not entered_picks:
        st.info(
            "Introduce cuotas en la pestaña Jornada "
            "para que aparezcan aquí."
        )

    elif not valid:
        st.warning(
            f"NO BET: ninguna cuota introducida alcanza "
            f"EV mínimo de {min_ev_percent}%."
        )

    else:
        valid = sorted(
            valid,
            key=lambda x: x["ev"],
            reverse=True,
        )

        st.success(
            f"{len(valid)} oportunidades superan "
            f"el filtro EV ≥ {min_ev_percent}%."
        )

        table = pd.DataFrame(
            [
                {
                    "Partido": p["partido"],
                    "Mercado": p["mercado"],
                    "Nivel": p["strength"],
                    "Prob. modelo": f"{p['prob']*100:.1f}%",
                    "Cuota justa": f"{p['fair_odds']:.2f}",
                    "Cuota": f"{p['market_odds']:.2f}",
                    "EV": f"{p['ev']*100:+.1f}%",
                }
                for p in valid
            ]
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Que un pick tenga EV positivo según el modelo "
            "no garantiza que vaya a ganar."
        )


with tab_1x2:
    st.warning(
        "El modelo 1X2 mejoró el baseline predictivo, "
        "pero dio ROI negativo contra cuotas de cierre. "
        "Por eso NO genera picks de apuesta."
    )

    rows = []

    for match in predictions:
        g = match["goals"]

        rows.append(
            {
                "Partido": (
                    f"{match['home_team']} - "
                    f"{match['away_team']}"
                ),
                "Local": f"{g['home_win_raw']*100:.1f}%",
                "Empate": f"{g['draw_raw']*100:.1f}%",
                "Visitante": f"{g['away_win_raw']*100:.1f}%",
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )
