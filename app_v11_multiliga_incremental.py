from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from prediction_engine_v3 import PredictionEngine as EliteserienEngine

try:
    from prediction_engine_laliga_v4_cards_referee import PredictionEngineLaLiga
except Exception:
    PredictionEngineLaLiga = None


st.set_page_config(
    page_title="Football Edge Pro",
    page_icon="⚽",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent


def run_laliga_updater():
    updater = BASE_DIR / "update_laliga_fixtures_v4_referee.py"

    if not updater.exists():
        return False, (
            "Falta update_laliga_fixtures_v4_referee.py "
            "en la carpeta del proyecto."
        )

    result = subprocess.run(
        [sys.executable, str(updater)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )

    output = (
        (result.stdout or "")
        + ("\n" + result.stderr if result.stderr else "")
    ).strip()

    return result.returncode == 0, output



def run_laliga_full_updater():
    updater = BASE_DIR / "update_laliga_all_v2_incremental.py"

    if not updater.exists():
        return False, (
            "Falta update_laliga_all_v2_incremental.py en la carpeta del proyecto."
        )

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [sys.executable, str(updater)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        output = (
            (exc.stdout or "")
            + ("\n" + exc.stderr if exc.stderr else "")
        ).strip()
        return False, (
            "La actualización superó 30 minutos.\n" + output[-8000:]
        )

    output = (
        (result.stdout or "")
        + ("\n" + result.stderr if result.stderr else "")
    ).strip()

    return result.returncode == 0, output


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

    if league_key == "laliga":
        if st.button(
            "🚀 ACTUALIZAR TODO",
            type="primary",
            use_container_width=True,
            help=(
                "Actualización incremental: solo busca partidos y árbitros nuevos, "
                "reentrena solo si hay resultados nuevos y descarga la próxima jornada."
            ),
        ):
            with st.spinner(
                "Buscando SOLO partidos nuevos de la temporada actual..."
            ):
                ok, output = run_laliga_full_updater()

            if ok:
                st.session_state["laliga_update_message"] = (
                    "✅ ACTUALIZACIÓN INCREMENTAL: solo se procesó lo nuevo."
                )
                st.session_state["laliga_full_update_output"] = output[-12000:]
                get_predictions.clear()
                get_engine.clear()
                st.rerun()
            else:
                st.error(
                    "No se pudo completar la actualización general. "
                    "La copia de seguridad se restaura automáticamente si falla un paso."
                )
                if output:
                    st.code(
                        output[-12000:],
                        language="text",
                    )

        if st.button(
            "🔄 Solo próxima jornada",
            use_container_width=True,
            help="Solo refresca fixtures y árbitros asignados; no reentrena modelos.",
        ):
            with st.spinner(
                "Descargando jornada y comprobando árbitros..."
            ):
                ok, output = run_laliga_updater()

            if ok:
                st.session_state["laliga_update_message"] = (
                    "✅ Próxima jornada y árbitros actualizados."
                )
                get_predictions.clear()
                get_engine.clear()
                st.rerun()
            else:
                st.error(
                    "No se pudo actualizar la jornada de LaLiga."
                )
                if output:
                    st.code(
                        output[-5000:],
                        language="text",
                    )
    else:
        if st.button(
            "🔄 Recargar jornada",
            use_container_width=True,
        ):
            get_predictions.clear()
            get_engine.clear()
            st.rerun()

    st.divider()

    min_ev_percent = st.slider(
        "EV mínimo para BET",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
    )

    min_ev = min_ev_percent / 100.0

    if league_key == "eliteserien":
        st.markdown(
            "**Mercados del motor**  \n"
            "⚽ Over 2.5 goles · fuerte  \n"
            "⚽ Over 3.5 goles · secundario  \n"
            "🚩 Over 9.5 / 10.5 / 11.5 córners  \n"
            "🟨 Tarjetas · según disponibilidad"
        )
    else:
        st.markdown(
            "**Mercados LaLiga**  \n"
            "⚽ O1.5 / O2.5 / O3.5 goles  \n"
            "🚩 O8.5 / O10.5 córners  \n"
            "🟨 Cards V3 + árbitro · O2.5 / O3.5 / O4.5 / O5.5  \n"
            "🎯 Tiros a puerta totales y por equipo"
        )


st.title(f"⚽ {league['title']}")

if (
    league_key == "laliga"
    and "laliga_update_message" in st.session_state
):
    st.success(
        st.session_state.pop("laliga_update_message")
    )


if (
    league_key == "laliga"
    and "laliga_full_update_output" in st.session_state
):
    with st.expander("📋 Ver resumen técnico de la última actualización"):
        st.code(
            st.session_state.pop("laliga_full_update_output"),
            language="text",
        )
st.caption("Football Edge Pro · Multiliga v10 · actualización automática + Cards V3")
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
            "No hay una jornada futura disponible para el motor. "
            "Puedes descargar solo la próxima jornada o usar ACTUALIZAR TODO."
        )

        if st.button(
            "🇪🇸 Descargar próxima jornada de LaLiga",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "Buscando partidos y designaciones arbitrales en FotMob..."
            ):
                ok, output = run_laliga_updater()

            if ok:
                st.session_state["laliga_update_message"] = (
                    "✅ Próxima jornada descargada correctamente."
                )
                get_predictions.clear()
                get_engine.clear()
                st.rerun()
            else:
                st.error(
                    "El actualizador no pudo cargar los partidos."
                )
                if output:
                    st.code(
                        output[-5000:],
                        language="text",
                    )

        st.caption(
            "Ya no necesitas ejecutar ningún comando en CMD "
            "para actualizar la jornada."
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

if league_key == "eliteserien":
    MARKETS = [
        ("goals", "over_2_5", "⚽ Over 2.5 goles", "FUERTE"),
        ("goals", "over_3_5", "⚽ Over 3.5 goles", "SECUNDARIO"),
        ("corners", "over_9_5", "🚩 Over 9.5 córners", "APROBADO"),
        ("corners", "over_10_5", "🚩 Over 10.5 córners", "FUERTE"),
        ("corners", "over_11_5", "🚩 Over 11.5 córners", "FUERTE"),
        ("cards", "over_3_5", "🟨 Over 3.5 amarillas", "SECUNDARIO"),
        ("cards", "over_4_5", "🟨 Over 4.5 amarillas", "APROBADO"),
        ("cards", "over_5_5", "🟨 Over 5.5 amarillas", "FUERTE"),
    ]
else:
    MARKETS = [
        ("goals", "over_1_5", "⚽ Over 1.5 goles", "FUERTE"),
        ("goals", "over_2_5", "⚽ Over 2.5 goles", "APROBADO"),
        ("goals", "over_3_5", "⚽ Over 3.5 goles", "FUERTE"),
        ("corners", "over_8_5", "🚩 Over 8.5 córners", "FUERTE"),
        ("corners", "over_10_5", "🚩 Over 10.5 córners", "APROBADO"),
        ("cards", "over_2_5", "🟨 Over 2.5 amarillas", "FUERTE V3"),
        ("cards", "over_3_5", "🟨 Over 3.5 amarillas", "FUERTE V3"),
        ("cards", "over_4_5", "🟨 Over 4.5 amarillas", "FUERTE V3"),
        ("cards", "over_5_5", "🟨 Over 5.5 amarillas", "FUERTE V3"),
    ]



def render_sot_market(
    match,
    label,
    prob,
    strength,
    key_suffix,
    entered_picks,
):
    fair = fair_odds(prob)
    minimum = min_bet_odds(prob, min_ev)

    cols = st.columns(
        [2.8, 1.0, 1.0, 1.2, 1.2, 1.7]
    )

    cols[0].write(
        f"{label} · {strength}"
    )
    cols[1].write(f"{prob*100:.1f}%")
    cols[2].write(f"{fair:.2f}")
    cols[3].write(f"🎯 {minimum:.2f}")

    odds = cols[4].number_input(
        "Cuota",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.01,
        key=(
            f"laliga_sot_{match['match_id']}_"
            f"{key_suffix}"
        ),
        label_visibility="collapsed",
    )

    ev = ev_value(prob, odds)

    cols[5].write(
        ev_label(ev, min_ev)
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

            if league_key == "laliga":
                cards_v3 = match.get("cards", {})
                ref_info = cards_v3.get(
                    "referee_info",
                    {},
                )

                if ref_info.get("assigned"):
                    ref_name = (
                        ref_info.get("name")
                        or referee
                        or "Árbitro"
                    )
                    profile = ref_info.get(
                        "profile",
                        "NEUTRO",
                    )

                    ref_avg = ref_info.get(
                        "shrunk_yellow"
                    )
                    league_avg = ref_info.get(
                        "league_avg_yellow"
                    )
                    delta = ref_info.get(
                        "delta_vs_league",
                        0.0,
                    )
                    matches_ref = int(
                        ref_info.get(
                            "matches",
                            0,
                        )
                    )
                    last10 = ref_info.get(
                        "last10_yellow"
                    )

                    effect_mu = float(
                        cards_v3.get(
                            "referee_effect_mu",
                            0.0,
                        )
                    )

                    effect_o45 = float(
                        cards_v3.get(
                            "referee_effect_probability",
                            {},
                        ).get(
                            "over_4_5",
                            0.0,
                        )
                    )

                    ref_avg_txt = (
                        f"{ref_avg:.2f}"
                        if ref_avg is not None
                        and not pd.isna(ref_avg)
                        else "—"
                    )

                    league_avg_txt = (
                        f"{league_avg:.2f}"
                        if league_avg is not None
                        and not pd.isna(league_avg)
                        else "—"
                    )

                    last10_txt = (
                        f"{last10:.2f}"
                        if last10 is not None
                        and not pd.isna(last10)
                        else "—"
                    )

                    icon = (
                        "🔥"
                        if effect_mu >= 0.15
                        else "🧊"
                        if effect_mu <= -0.15
                        else "⚖️"
                    )

                    st.info(
                        f"🧑‍⚖️ **{ref_name}** · **{profile}** · "
                        f"{ref_avg_txt} amarillas/p "
                        f"(LaLiga {league_avg_txt}, {delta:+.2f}) · "
                        f"N={matches_ref} · últimos 10: {last10_txt}  \n"
                        f"{icon} **Impacto Cards V3:** "
                        f"{effect_mu:+.2f} tarjetas esperadas · "
                        f"Over 4.5 {effect_o45*100:+.1f} puntos porcentuales"
                    )
                else:
                    st.caption(
                        "🧑‍⚖️ Árbitro pendiente · Cards V3 utiliza "
                        "un perfil arbitral neutral hasta que FotMob "
                        "publique la designación."
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


            if league_key == "laliga" and "shots_on_target" in match:
                sot = match["shots_on_target"]

                with st.expander(
                    "🎯 Remates a puerta · Totales y por equipo",
                    expanded=False,
                ):
                    st.caption(
                        f"Modelo SOT esperado: "
                        f"{match['home_team']} {sot['lambda_home']:.2f} · "
                        f"{match['away_team']} {sot['lambda_away']:.2f}"
                    )

                    st.markdown("**Total del partido**")

                    total_line = st.selectbox(
                        "Línea total SOT",
                        options=[7.5, 8.5, 9.5, 10.5, 11.5],
                        index=1,
                        key=f"sot_total_line_{match['match_id']}",
                    )

                    total_key = (
                        "total_over_"
                        + str(total_line).replace(".", "_")
                    )

                    total_strength = {
                        7.5: "APROBADO",
                        8.5: "FUERTE",
                        9.5: "APROBADO",
                        10.5: "FUERTE",
                        11.5: "SECUNDARIO",
                    }[total_line]

                    render_sot_market(
                        match,
                        f"🎯 Total partido Over {total_line} SOT",
                        float(sot[total_key]),
                        total_strength,
                        f"total_{str(total_line).replace('.', '_')}",
                        entered_picks,
                    )

                    st.divider()
                    st.markdown(
                        f"**{match['home_team']} · tiros a puerta**"
                    )

                    home_line = st.selectbox(
                        "Línea equipo local",
                        options=[2.5, 3.5, 4.5, 5.5, 6.5],
                        index=1,
                        key=f"sot_home_line_{match['match_id']}",
                    )

                    hk = (
                        "home_over_"
                        + str(home_line).replace(".", "_")
                    )

                    home_strength = (
                        "SECUNDARIO"
                        if home_line == 6.5
                        else "FUERTE"
                    )

                    render_sot_market(
                        match,
                        (
                            f"🎯 {match['home_team']} "
                            f"Over {home_line} SOT"
                        ),
                        float(sot[hk]),
                        home_strength,
                        f"home_{str(home_line).replace('.', '_')}",
                        entered_picks,
                    )

                    st.divider()
                    st.markdown(
                        f"**{match['away_team']} · tiros a puerta**"
                    )

                    away_line = st.selectbox(
                        "Línea equipo visitante",
                        options=[2.5, 3.5, 4.5, 5.5, 6.5],
                        index=1,
                        key=f"sot_away_line_{match['match_id']}",
                    )

                    ak = (
                        "away_over_"
                        + str(away_line).replace(".", "_")
                    )

                    away_strength = (
                        "SECUNDARIO"
                        if away_line == 6.5
                        else "FUERTE"
                    )

                    render_sot_market(
                        match,
                        (
                            f"🎯 {match['away_team']} "
                            f"Over {away_line} SOT"
                        ),
                        float(sot[ak]),
                        away_strength,
                        f"away_{str(away_line).replace('.', '_')}",
                        entered_picks,
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
