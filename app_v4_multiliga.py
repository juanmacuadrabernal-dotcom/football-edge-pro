from __future__ import annotations

import pandas as pd
import streamlit as st

from prediction_engine_v3 import PredictionEngine as EliteserienEngine

try:
    from prediction_engine_laliga import PredictionEngineLaLiga
except Exception:
    PredictionEngineLaLiga = None


st.set_page_config(
    page_title="Football Edge Pro",
    page_icon="⚽",
    layout="wide",
)


LEAGUES = {
    "🇳🇴 Eliteserien": {
        "key": "eliteserien",
        "title": "Eliteserien Edge Pro",
    },
    "🇪🇸 LaLiga": {
        "key": "laliga",
        "title": "LaLiga Edge Pro",
    },
}


@st.cache_resource
def get_engine(league_key):
    if league_key == "eliteserien":
        return EliteserienEngine()

    if league_key == "laliga":
        if PredictionEngineLaLiga is None:
            raise RuntimeError(
                "Falta prediction_engine_laliga.py"
            )
        return PredictionEngineLaLiga()

    raise ValueError(league_key)


@st.cache_data(ttl=300)
def get_predictions(league_key):
    engine = get_engine(league_key)
    return engine.predict_next_round()


def fair_odds(prob):
    if prob <= 0:
        return None
    return 1.0 / prob


def min_bet_odds(prob, min_ev):
    if prob <= 0:
        return None
    return (1.0 + min_ev) / prob


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


with st.sidebar:
    st.header("⚽ Football Edge Pro")

    league_label = st.selectbox(
        "Liga",
        options=list(LEAGUES.keys()),
        index=0,
    )

    league = LEAGUES[league_label]
    league_key = league["key"]

    st.divider()

    min_ev_percent = st.slider(
        "EV mínimo para BET",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
    )

    min_ev = min_ev_percent / 100.0

    st.markdown(
        "**Mercados del motor**  \n"
        "⚽ Over 2.5 goles · fuerte  \n"
        "⚽ Over 3.5 goles · secundario  \n"
        "🚩 Over 9.5 / 10.5 / 11.5 córners  \n"
        "🟨 Over 3.5 / 4.5 / 5.5 amarillas"
    )


st.title(f"⚽ {league['title']}")
st.caption(
    f"{league_label} · Modelos independientes por competición. "
    "Nunca reutilizamos las probabilidades de otra liga."
)

if league_key == "eliteserien":
    st.success("✅ NORUEGA · MOTOR ACTIVO")
else:
    st.success("🇪🇸 LALIGA · MOTOR INDEPENDIENTE")


try:
    predictions = get_predictions(
        league_key
    )
except FileNotFoundError as exc:
    if league_key == "laliga":
        st.warning(
            "LaLiga todavía no tiene los modelos generados en tu PC."
        )
        st.code(
            "python build_laliga_phase1.py",
            language="text",
        )
        st.caption(
            "Ese comando descarga el histórico, construye features "
            "y entrena goles, córners y tarjetas."
        )
        st.stop()

    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(
        "No he podido cargar el motor de esta liga."
    )
    st.exception(exc)
    st.stop()


if not predictions:
    if league_key == "laliga":
        st.warning(
            "Los modelos están creados, pero no encuentro próximos "
            "partidos SP1 en el archivo de fixtures. "
            "Vuelve a ejecutar: python setup_laliga_data.py"
        )
    else:
        st.warning(
            "No encuentro partidos futuros. "
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
        "goals",
        "over_3_5",
        "⚽ Over 3.5 goles",
        "SECUNDARIO",
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
                f"Modelo goles: "
                f"{g['lambda_home']:.2f} - {g['lambda_away']:.2f}"
            )

            header = st.columns(
                [2.8, 1.0, 1.0, 1.2, 1.2, 1.7]
            )

            header[0].markdown("**Mercado**")
            header[1].markdown("**Modelo**")
            header[2].markdown("**Justa**")
            header[3].markdown("**Mín. BET**")
            header[4].markdown("**Mercado**")
            header[5].markdown("**Decisión**")

            for category, key, label, strength in MARKETS:
                if key not in match.get(category, {}):
                    continue

                prob = float(
                    match[category][key]
                )

                fair = fair_odds(prob)
                minimum = min_bet_odds(
                    prob,
                    min_ev,
                )

                cols = st.columns(
                    [2.8, 1.0, 1.0, 1.2, 1.2, 1.7]
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

                cols[3].write(
                    f"🎯 {minimum:.2f}"
                    if minimum is not None
                    else "—"
                )

                widget_key = (
                    f"{league_key}_odds_"
                    f"{match['match_id']}_{category}_{key}"
                )

                odds = cols[4].number_input(
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

                cols[5].write(
                    ev_label(
                        ev,
                        min_ev,
                    )
                )

                if ev is not None:
                    cols[5].caption(
                        f"EV {ev*100:+.1f}%"
                    )

                    entered_picks.append(
                        {
                            "partido": (
                                f"{match['home_team']} - "
                                f"{match['away_team']}"
                            ),
                            "mercado": label,
                            "strength": strength,
                            "prob": prob,
                            "fair_odds": fair,
                            "min_odds": minimum,
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
            "Introduce las cuotas de tu casa en Jornada. "
            "El scanner ordenará automáticamente el valor."
        )
    elif not valid:
        st.warning(
            f"NO BET: ninguna cuota introducida alcanza "
            f"EV ≥ {min_ev_percent}%."
        )
    else:
        valid = sorted(
            valid,
            key=lambda x: x["ev"],
            reverse=True,
        )

        st.success(
            f"{len(valid)} oportunidades superan "
            f"EV ≥ {min_ev_percent}%."
        )

        table = pd.DataFrame(
            [
                {
                    "Partido": p["partido"],
                    "Mercado": p["mercado"],
                    "Nivel": p["strength"],
                    "Modelo": f"{p['prob']*100:.1f}%",
                    "Justa": f"{p['fair_odds']:.2f}",
                    "Mín. BET": f"{p['min_odds']:.2f}",
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


with tab_1x2:
    if league_key == "eliteserien":
        st.warning(
            "En Eliteserien el 1X2 mejoró el baseline predictivo, "
            "pero dio ROI negativo contra cuotas de cierre. "
            "Solo contexto."
        )
    else:
        st.info(
            "En LaLiga todavía no hemos validado ROI del 1X2. "
            "Estas probabilidades son contexto y NO generan picks."
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
