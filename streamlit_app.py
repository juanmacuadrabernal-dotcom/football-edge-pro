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


def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0b1220 0%, #111827 45%, #0f172a 100%);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
            max-width: 1380px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(148,163,184,0.15);
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stCaption {
            color: #e5e7eb;
        }
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(96,165,250,0.20);
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.18);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        .hero-card {
            position: relative;
            overflow: hidden;
            background: radial-gradient(circle at top right, rgba(59,130,246,.28), transparent 28%),
                        linear-gradient(135deg, rgba(15,23,42,0.96), rgba(17,24,39,0.95));
            border: 1px solid rgba(96,165,250,0.24);
            border-radius: 24px;
            padding: 24px 26px 18px 26px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.24);
            margin-bottom: 14px;
        }
        .hero-topline {
            display: inline-block;
            background: rgba(59,130,246,0.14);
            border: 1px solid rgba(96,165,250,0.18);
            color: #93c5fd;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: .04em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .hero-title {
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 0;
        }
        .hero-subtitle {
            color: #cbd5e1;
            font-size: 0.98rem;
            margin-top: 6px;
            margin-bottom: 16px;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 6px;
        }
        .chip {
            background: rgba(30,41,59,0.95);
            border: 1px solid rgba(148,163,184,0.20);
            color: #e2e8f0;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 600;
        }
        .chip.status {
            background: rgba(16,185,129,0.12);
            border-color: rgba(16,185,129,0.24);
            color: #a7f3d0;
        }
        .subtle-note {
            color: #94a3b8;
            font-size: 0.84rem;
            margin-top: 12px;
        }
        [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(15,23,42,0.45);
            padding: 8px;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.14);
            margin-bottom: 10px;
        }
        [data-baseweb="tab"] {
            height: 44px;
            border-radius: 12px;
            padding-left: 18px;
            padding-right: 18px;
            font-weight: 700;
            background: transparent;
        }
        [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(37,99,235,.32), rgba(59,130,246,.12));
            color: #eff6ff !important;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(96,165,250,0.24);
            box-shadow: 0 10px 22px rgba(0,0,0,0.15);
            font-weight: 700;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: white;
        }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.15);
        }
        div[data-testid="stExpander"] {
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.15);
            background: rgba(15,23,42,0.50);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 20px !important;
            border: 1px solid rgba(148,163,184,0.14) !important;
            background: rgba(15,23,42,0.58);
            box-shadow: 0 12px 28px rgba(0,0,0,0.14);
        }
        .section-title {
            color: #f8fafc;
            font-size: 1.2rem;
            font-weight: 800;
            margin-top: .2rem;
            margin-bottom: .35rem;
        }
        .section-text {
            color: #cbd5e1;
            margin-bottom: 0.8rem;
        }
        .sidebar-card {
            background: rgba(15,23,42,0.8);
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 18px;
            padding: 16px 16px 14px 16px;
            margin-bottom: 12px;
        }
        .sidebar-title {
            color: #f8fafc;
            font-weight: 800;
            font-size: 1.05rem;
            margin-bottom: 4px;
        }
        .sidebar-sub {
            color: #94a3b8;
            font-size: 0.84rem;
        }
        .fixture-card {
            background: linear-gradient(145deg, rgba(17,24,39,.96), rgba(15,23,42,.92));
            border: 1px solid rgba(148,163,184,.15);
            border-radius: 20px;
            padding: 16px 17px;
            min-height: 132px;
            box-shadow: 0 14px 34px rgba(0,0,0,.18);
            margin-bottom: 8px;
        }
        .fixture-time {
            color: #60a5fa;
            font-weight: 800;
            font-size: .78rem;
            letter-spacing: .04em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .fixture-team {
            color: #f8fafc;
            font-weight: 800;
            font-size: 1rem;
            line-height: 1.45;
        }
        .fixture-meta {
            color: #94a3b8;
            font-size: .78rem;
            margin-top: 9px;
        }
        .mobile-nav-card {
            background: linear-gradient(135deg, rgba(37,99,235,.12), rgba(15,23,42,.78));
            border: 1px solid rgba(96,165,250,.18);
            border-radius: 18px;
            padding: 12px 15px 4px 15px;
            margin: 12px 0 16px 0;
        }
        .mobile-nav-label {
            color: #dbeafe;
            font-weight: 800;
            font-size: .9rem;
            margin-bottom: 2px;
        }
        .market-note {
            background: rgba(15,23,42,.72);
            border-left: 3px solid #3b82f6;
            border-radius: 12px;
            padding: 11px 14px;
            color: #cbd5e1;
            margin: 8px 0 14px 0;
        }
        .odds-hero {
            background: radial-gradient(circle at 85% 15%, rgba(16,185,129,.16), transparent 32%),
                        linear-gradient(135deg, rgba(15,23,42,.96), rgba(17,24,39,.94));
            border: 1px solid rgba(16,185,129,.18);
            border-radius: 20px;
            padding: 18px 20px;
            margin: 12px 0 14px 0;
        }
        .odds-title {
            color: #f8fafc;
            font-size: 1.15rem;
            font-weight: 850;
            margin-bottom: 4px;
        }
        .odds-subtitle {
            color: #94a3b8;
            font-size: .86rem;
        }
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stMultiSelect"] > div,
        div[data-testid="stNumberInput"] > div {
            border-radius: 14px;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: .65rem;
                padding-left: .75rem;
                padding-right: .75rem;
                padding-bottom: 1.4rem;
            }
            .hero-card {
                border-radius: 18px;
                padding: 18px 16px 15px 16px;
                margin-bottom: 8px;
            }
            .hero-title {
                font-size: 1.55rem;
                line-height: 1.12;
            }
            .hero-subtitle {
                font-size: .88rem;
            }
            .chip-row {
                gap: 7px;
            }
            .chip {
                font-size: .75rem;
                padding: 6px 9px;
            }
            div[data-testid="stMetric"] {
                border-radius: 14px;
                padding: 10px 11px;
            }
            .fixture-card {
                min-height: auto;
                border-radius: 16px;
                padding: 13px 14px;
            }
            .section-title {
                font-size: 1.08rem;
            }
            [data-baseweb="tab-list"] {
                display: none !important;
            }
            section[data-testid="stSidebar"] {
                min-width: 280px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def league_market_chips(league_key):
    if league_key == "eliteserien":
        return [
            "⚽ O2.5 / O3.5 goles",
            "🚩 O9.5 / O10.5 / O11.5 córners",
            "🟨 Tarjetas",
        ]

    return [
        "⚽ O1.5 / O2.5 / O3.5 goles",
        "🚩 O8.5 / O10.5 córners",
        "🟨 Cards V3 + árbitro",
        "🎯 Tiros a puerta",
    ]


def render_hero_header(league_label, league_title, league_key):
    status = (
        "✅ NORUEGA · MOTOR ACTIVO"
        if league_key == "eliteserien"
        else "🇪🇸 LALIGA · MOTOR INDEPENDIENTE"
    )
    chips = "".join(
        f'<span class="chip">{c}</span>' for c in league_market_chips(league_key)
    )
    html = f"""
    <div class="hero-card">
        <div class="hero-topline">Football Edge Pro · Multiliga v14</div>
        <h1 class="hero-title">⚽ {league_title}</h1>
        <div class="hero-subtitle">{league_label} · dashboard deportivo premium · estilo bet app · pensado para PC y móvil.</div>
        <div class="chip-row">
            <span class="chip status">{status}</span>
            {chips}
        </div>
        <div class="subtle-note">Modelos independientes por competición. Las probabilidades de una liga nunca se reutilizan en otra.</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_fixture_strip(predictions):
    if not predictions:
        return

    st.markdown(
        '<div class="section-title">🔥 Próximos partidos</div>'
        '<div class="section-text">Vista rápida de la jornada, estilo marcador deportivo.</div>',
        unsafe_allow_html=True,
    )

    visible = predictions[:3]
    cols = st.columns(len(visible), gap="large")

    for col, match in zip(cols, visible):
        kickoff = pd.Timestamp(
            match["utc_time"]
        ).tz_convert("Europe/Madrid")

        referee = match.get("referee")
        if referee is None or pd.isna(referee):
            referee_text = "Árbitro pendiente"
        else:
            referee_text = f"🧑‍⚖️ {referee}"

        html = f"""
        <div class="fixture-card">
            <div class="fixture-time">{kickoff.strftime('%a %d/%m · %H:%M')}</div>
            <div class="fixture-team">{match['home_team']}</div>
            <div class="fixture-team">{match['away_team']}</div>
            <div class="fixture-meta">{referee_text}</div>
        </div>
        """
        col.markdown(
            html,
            unsafe_allow_html=True,
        )


NAV_PAGES = {
    "scanner": "🔎 Scanner jornada",
    "matches": "📊 Partido a partido",
    "picks": "⭐ Picks / valor",
    "one_x_two": "💶 1X2 + cuotas justas",
}


def mobile_navigation():
    try:
        requested = st.query_params.get(
            "page",
            "scanner",
        )
    except Exception:
        requested = "scanner"

    if requested not in NAV_PAGES:
        requested = "scanner"

    keys = list(NAV_PAGES.keys())
    labels = list(NAV_PAGES.values())
    index = keys.index(requested)

    st.markdown(
        """
        <div class="mobile-nav-card">
            <div class="mobile-nav-label">☰ MENÚ · Ir a</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_label = st.selectbox(
        "☰ MENÚ · Ir a",
        labels,
        index=index,
        key="football_edge_mobile_nav",
        label_visibility="collapsed",
    )

    selected_key = keys[
        labels.index(selected_label)
    ]

    try:
        if st.query_params.get(
            "page"
        ) != selected_key:
            st.query_params["page"] = (
                selected_key
            )
    except Exception:
        pass

    return selected_key


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
    updater = BASE_DIR / "update_laliga_all_v3_incremental_fotmob.py"

    if not updater.exists():
        return False, (
            "Falta update_laliga_all_v3_incremental_fotmob.py en la carpeta del proyecto."
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


inject_custom_css()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">⚽ Football Edge Pro</div>
            <div class="sidebar-sub">Escáner de mercados, ranking por EV y modelos separados por liga.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
                "Consultando resultados nuevos en FotMob (solo temporada actual)..."
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


render_hero_header(league_label, league['title'], league_key)

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

st.markdown(
    f"<div class=\"section-title\">Jornada {round_text} · {len(predictions)} partidos</div>"
    f"<div class=\"section-text\">Scanner, vista partido a partido, picks manuales y contexto 1X2 en una sola app.</div>",
    unsafe_allow_html=True,
)

render_fixture_strip(
    predictions
)

active_page = mobile_navigation()

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



def scanner_rows_for_predictions(
    predictions,
    league_key,
    min_ev,
):
    rows = []

    def add_row(
        match,
        category,
        market_key,
        label,
        strength,
        probability,
        ref_profile="",
        ref_effect_pp=None,
    ):
        probability = float(probability)
        kickoff = pd.Timestamp(
            match["utc_time"]
        ).tz_convert("Europe/Madrid")

        rows.append(
            {
                "ID": (
                    f"{match['match_id']}|"
                    f"{category}|{market_key}"
                ),
                "Hora": kickoff.strftime(
                    "%d/%m %H:%M"
                ),
                "Partido": (
                    f"{match['home_team']} - "
                    f"{match['away_team']}"
                ),
                "Tipo": category,
                "Mercado": label,
                "Nivel": strength,
                "ProbRaw": probability,
                "Modelo %": probability * 100.0,
                "Justa": fair_odds(probability),
                "Mín. BET": min_bet_odds(
                    probability,
                    min_ev,
                ),
                "Árbitro": ref_profile,
                "Impacto árbitro pp": (
                    float(ref_effect_pp) * 100.0
                    if ref_effect_pp is not None
                    else None
                ),
            }
        )

    for match in predictions:
        for category, key, label, strength in MARKETS:
            market_data = match.get(
                category,
                {},
            )

            if key not in market_data:
                continue

            ref_profile = ""
            ref_effect = None

            if (
                league_key == "laliga"
                and category == "cards"
            ):
                cards = match.get(
                    "cards",
                    {},
                )

                ref_info = cards.get(
                    "referee_info",
                    {},
                )

                if ref_info.get("assigned"):
                    name = (
                        ref_info.get("name")
                        or match.get("referee")
                        or "Árbitro"
                    )
                    profile = ref_info.get(
                        "profile",
                        "NEUTRO",
                    )
                    ref_profile = (
                        f"{name} · {profile}"
                    )
                else:
                    ref_profile = "Pendiente · neutral"

                ref_effect = (
                    cards.get(
                        "referee_effect_probability",
                        {},
                    ).get(
                        key,
                        0.0,
                    )
                )

            add_row(
                match,
                category,
                key,
                label,
                strength,
                market_data[key],
                ref_profile,
                ref_effect,
            )

        if (
            league_key == "laliga"
            and "shots_on_target" in match
        ):
            sot = match[
                "shots_on_target"
            ]

            total_strength = {
                7.5: "APROBADO",
                8.5: "FUERTE",
                9.5: "APROBADO",
                10.5: "FUERTE",
                11.5: "SECUNDARIO",
            }

            for line, strength in total_strength.items():
                key = (
                    "total_over_"
                    + str(line).replace(
                        ".",
                        "_",
                    )
                )

                if key in sot:
                    add_row(
                        match,
                        "sot_total",
                        key,
                        (
                            f"🎯 Total Over "
                            f"{line} tiros a puerta"
                        ),
                        strength,
                        sot[key],
                    )

            for side, team_key in (
                ("home", "home_team"),
                ("away", "away_team"),
            ):
                team = match[
                    team_key
                ]

                for line in (
                    2.5,
                    3.5,
                    4.5,
                    5.5,
                    6.5,
                ):
                    key = (
                        f"{side}_over_"
                        + str(line).replace(
                            ".",
                            "_",
                        )
                    )

                    if key not in sot:
                        continue

                    strength = (
                        "SECUNDARIO"
                        if line == 6.5
                        else "FUERTE"
                    )

                    add_row(
                        match,
                        (
                            "sot_local"
                            if side == "home"
                            else "sot_visitante"
                        ),
                        key,
                        (
                            f"🎯 {team} Over "
                            f"{line} tiros a puerta"
                        ),
                        strength,
                        sot[key],
                    )

    return rows


def scanner_type_label(value):
    return {
        "goals": "⚽ Goles",
        "corners": "🚩 Córners",
        "cards": "🟨 Amarillas",
        "sot_total": "🎯 SOT total",
        "sot_local": "🎯 SOT local",
        "sot_visitante": "🎯 SOT visitante",
    }.get(
        value,
        value,
    )


def scanner_strength_rank(value):
    value = str(value).upper()

    if "FUERTE" in value:
        return 3

    if "APROBADO" in value:
        return 2

    return 1


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


if active_page == "scanner":
    st.markdown(
        "<div class=\"section-title\">🔎 Scanner de jornada</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class=\"section-text\">Filtra, ordena y mete tus cuotas. La app prioriza claridad visual y ranking de oportunidades.</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Ordenado automáticamente de mejor a peor. "
        "Sin cuotas: por probabilidad del modelo. "
        "Con cuotas: por valor esperado (EV)."
    )

    scanner_rows = scanner_rows_for_predictions(
        predictions,
        league_key,
        min_ev,
    )

    scanner_df = pd.DataFrame(
        scanner_rows
    )

    if scanner_df.empty:
        st.info(
            "No hay mercados disponibles para escanear."
        )
    else:
        scanner_df["Tipo mercado"] = (
            scanner_df["Tipo"]
            .map(scanner_type_label)
        )

        scanner_df["NivelRank"] = (
            scanner_df["Nivel"]
            .map(scanner_strength_rank)
        )

        # ---------------------------------------------------------
        # FILTROS
        # ---------------------------------------------------------
        with st.container():
            c1, c2, c3 = st.columns(
                [1.6, 1.6, 1.0],
                gap="large",
            )

            all_types = list(
                dict.fromkeys(
                    scanner_df[
                        "Tipo mercado"
                    ].tolist()
                )
            )

            selected_types = c1.multiselect(
                "Mercados",
                options=all_types,
                default=all_types,
                key=f"scanner_types_{league_key}",
            )

            all_levels = list(
                dict.fromkeys(
                    scanner_df[
                        "Nivel"
                    ].tolist()
                )
            )

            selected_levels = c2.multiselect(
                "Nivel",
                options=all_levels,
                default=all_levels,
                key=f"scanner_levels_{league_key}",
            )

            min_model_prob = c3.slider(
                "Prob. mínima",
                min_value=0,
                max_value=90,
                value=0,
                step=5,
                key=f"scanner_prob_{league_key}",
            )

        filtered = scanner_df[
            scanner_df[
                "Tipo mercado"
            ].isin(
                selected_types
            )
            & scanner_df[
                "Nivel"
            ].isin(
                selected_levels
            )
            & (
                scanner_df[
                    "Modelo %"
                ] >= min_model_prob
            )
        ].copy()

        if filtered.empty:
            st.warning(
                "Los filtros actuales no dejan ningún mercado."
            )
        else:
            odds_store = (
                st.session_state.setdefault(
                    f"scanner_odds_{league_key}",
                    {},
                )
            )

            filtered["Cuota"] = (
                filtered["ID"]
                .map(odds_store)
                .fillna(0.0)
                .astype(float)
            )

            filtered["EV actual"] = (
                filtered["ProbRaw"]
                * filtered["Cuota"]
                - 1.0
            )

            filtered["Tiene cuota"] = (
                filtered["Cuota"] > 1.0
            )

            # Si ya hay cuotas, las oportunidades con cuota van arriba
            # ordenadas por EV. El resto sigue por probabilidad.
            if filtered["Tiene cuota"].any():
                filtered = filtered.sort_values(
                    [
                        "Tiene cuota",
                        "EV actual",
                        "Modelo %",
                        "NivelRank",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                        False,
                    ],
                )
            else:
                # Sin cuota todavía: ranking predictivo puro.
                filtered = filtered.sort_values(
                    [
                        "Modelo %",
                        "NivelRank",
                        "Hora",
                    ],
                    ascending=[
                        False,
                        False,
                        True,
                    ],
                )

            filtered = filtered.reset_index(
                drop=True
            )

            filtered.insert(
                0,
                "#",
                range(
                    1,
                    len(filtered) + 1,
                ),
            )

            # ---------------------------------------------------------
            # RESUMEN
            # ---------------------------------------------------------
            quoted_now = int(
                filtered["Tiene cuota"].sum()
            )

            candidates_now = int(
                (
                    filtered["Tiene cuota"]
                    & (
                        filtered["EV actual"]
                        >= min_ev
                    )
                ).sum()
            )

            m1, m2, m3 = st.columns(
                3,
                gap="large",
            )

            m1.metric(
                "Mercados",
                len(filtered),
            )

            m2.metric(
                "Con cuota",
                quoted_now,
            )

            m3.metric(
                f"EV ≥ {min_ev_percent}%",
                candidates_now,
            )

            st.markdown(
                "### 🏆 De mejor a peor"
            )

            if quoted_now:
                st.caption(
                    "Los mercados con cuota aparecen primero y se "
                    "ordenan por EV. Los que aún no tienen cuota "
                    "quedan después por probabilidad."
                )
            else:
                st.caption(
                    "Todavía no hay cuotas: ranking ordenado por "
                    "probabilidad del modelo. Esto NO implica valor "
                    "de apuesta."
                )

            # ---------------------------------------------------------
            # TABLA PRINCIPAL LIMPIA
            # ---------------------------------------------------------
            main_cols = [
                "ID",
                "#",
                "Hora",
                "Partido",
                "Mercado",
                "Nivel",
                "Modelo %",
                "Mín. BET",
                "Cuota",
            ]

            filter_signature = (
                str(len(filtered))
                + "_"
                + str(min_model_prob)
                + "_"
                + str(
                    sum(
                        ord(ch)
                        for ch in "|".join(
                            selected_types
                            + selected_levels
                        )
                    )
                )
                + "_clean"
            )

            edited = st.data_editor(
                filtered[
                    main_cols
                ],
                use_container_width=True,
                hide_index=True,
                height=620,
                key=(
                    f"scanner_editor_clean_"
                    f"{league_key}_"
                    f"{filter_signature}"
                ),
                disabled=[
                    col
                    for col in main_cols
                    if col != "Cuota"
                ],
                column_config={
                    "ID": None,
                    "#": st.column_config.NumberColumn(
                        "#",
                        width="small",
                        format="%d",
                    ),
                    "Hora": st.column_config.TextColumn(
                        "Hora",
                        width="small",
                    ),
                    "Partido": st.column_config.TextColumn(
                        "Partido",
                        width="medium",
                    ),
                    "Mercado": st.column_config.TextColumn(
                        "Mercado",
                        width="large",
                    ),
                    "Nivel": st.column_config.TextColumn(
                        "Nivel",
                        width="small",
                    ),
                    "Modelo %": st.column_config.NumberColumn(
                        "Modelo",
                        format="%.1f%%",
                        width="small",
                    ),
                    "Mín. BET": st.column_config.NumberColumn(
                        "Mín. BET",
                        format="%.2f",
                        width="small",
                        help=(
                            "Cuota mínima para alcanzar "
                            f"EV ≥ {min_ev_percent}%."
                        ),
                    ),
                    "Cuota": st.column_config.NumberColumn(
                        "Tu cuota",
                        min_value=0.0,
                        max_value=100.0,
                        step=0.01,
                        format="%.2f",
                        width="small",
                        help=(
                            "Escribe aquí la cuota decimal real."
                        ),
                    ),
                },
            )

            # Guardar cuotas editadas
            for _, row in edited.iterrows():
                row_id = row["ID"]
                odds = pd.to_numeric(
                    row["Cuota"],
                    errors="coerce",
                )

                if (
                    pd.notna(odds)
                    and float(odds) > 0
                ):
                    odds_store[
                        row_id
                    ] = float(odds)
                else:
                    odds_store.pop(
                        row_id,
                        None,
                    )

            # ---------------------------------------------------------
            # RANKING DE VALOR
            # ---------------------------------------------------------
            prob_map = dict(
                zip(
                    scanner_df["ID"],
                    scanner_df["ProbRaw"],
                )
            )

            quoted = edited[
                pd.to_numeric(
                    edited["Cuota"],
                    errors="coerce",
                ) > 1.0
            ].copy()

            st.markdown("")
            st.markdown(
                "### ⭐ Ranking de valor"
            )

            if quoted.empty:
                st.info(
                    "Introduce cuotas en la columna «Tu cuota». "
                    "Cuando haya alguna, aparecerán aquí ordenadas "
                    "automáticamente de mayor a menor EV."
                )
            else:
                quoted["ProbRaw"] = (
                    quoted["ID"]
                    .map(prob_map)
                    .astype(float)
                )

                quoted["EV"] = (
                    quoted["ProbRaw"]
                    * pd.to_numeric(
                        quoted["Cuota"],
                        errors="coerce",
                    )
                    - 1.0
                )

                quoted["EV %"] = (
                    quoted["EV"]
                    * 100.0
                )

                quoted["Decisión"] = quoted[
                    "EV"
                ].apply(
                    lambda x: (
                        "🟢 BET candidato"
                        if x >= min_ev
                        else (
                            "🟡 Poco valor"
                            if x >= 0
                            else "🔴 NO BET"
                        )
                    )
                )

                ranked = quoted.sort_values(
                    [
                        "EV",
                        "Modelo %",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                ).reset_index(
                    drop=True
                )

                ranked.insert(
                    0,
                    "Puesto",
                    range(
                        1,
                        len(ranked) + 1,
                    ),
                )

                candidates = ranked[
                    ranked["EV"] >= min_ev
                ].copy()

                if candidates.empty:
                    st.warning(
                        f"NO BET: ninguna de las {len(ranked)} "
                        "cuotas introducidas alcanza "
                        f"EV ≥ {min_ev_percent}%."
                    )
                else:
                    st.success(
                        f"🟢 {len(candidates)} candidatos superan "
                        f"EV ≥ {min_ev_percent}%."
                    )

                st.dataframe(
                    ranked[
                        [
                            "Puesto",
                            "Partido",
                            "Mercado",
                            "Modelo %",
                            "Mín. BET",
                            "Cuota",
                            "EV %",
                            "Decisión",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                    height=min(
                        520,
                        42 * (
                            len(ranked) + 1
                        ),
                    ),
                    column_config={
                        "Puesto": st.column_config.NumberColumn(
                            "#",
                            format="%d",
                            width="small",
                        ),
                        "Partido": st.column_config.TextColumn(
                            "Partido",
                            width="medium",
                        ),
                        "Mercado": st.column_config.TextColumn(
                            "Mercado",
                            width="large",
                        ),
                        "Modelo %": st.column_config.NumberColumn(
                            "Modelo",
                            format="%.1f%%",
                            width="small",
                        ),
                        "Mín. BET": st.column_config.NumberColumn(
                            "Mín. BET",
                            format="%.2f",
                            width="small",
                        ),
                        "Cuota": st.column_config.NumberColumn(
                            "Cuota",
                            format="%.2f",
                            width="small",
                        ),
                        "EV %": st.column_config.NumberColumn(
                            "EV",
                            format="%+.1f%%",
                            width="small",
                        ),
                        "Decisión": st.column_config.TextColumn(
                            "Decisión",
                            width="medium",
                        ),
                    },
                )

                st.caption(
                    "BET candidato = supera el EV mínimo configurado. "
                    "No implica beneficio garantizado."
                )

            # ---------------------------------------------------------
            # DETALLES SECUNDARIOS, FUERA DE LA VISTA PRINCIPAL
            # ---------------------------------------------------------
            with st.expander(
                "📋 Ver cuota justa, tipo y detalles del árbitro",
                expanded=False,
            ):
                detail_cols = [
                    "#",
                    "Hora",
                    "Partido",
                    "Tipo mercado",
                    "Mercado",
                    "Nivel",
                    "Modelo %",
                    "Justa",
                    "Mín. BET",
                ]

                if (
                    league_key == "laliga"
                    and (
                        filtered["Tipo"]
                        == "cards"
                    ).any()
                ):
                    detail_cols += [
                        "Árbitro",
                        "Impacto árbitro pp",
                    ]

                st.dataframe(
                    filtered[
                        detail_cols
                    ],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "#": st.column_config.NumberColumn(
                            "#",
                            format="%d",
                            width="small",
                        ),
                        "Modelo %": st.column_config.NumberColumn(
                            "Modelo",
                            format="%.1f%%",
                        ),
                        "Justa": st.column_config.NumberColumn(
                            "Cuota justa",
                            format="%.2f",
                        ),
                        "Mín. BET": st.column_config.NumberColumn(
                            "Mín. BET",
                            format="%.2f",
                        ),
                        "Impacto árbitro pp": st.column_config.NumberColumn(
                            "Impacto ref.",
                            format="%+.1f pp",
                        ),
                    },
                )


if active_page == "matches":
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


if active_page == "picks":
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


if active_page == "one_x_two":
    st.markdown(
        '<div class="section-title">💶 1X2 · Probabilidad y cuota justa</div>'
        '<div class="section-text">Compara lo que estima el modelo con la cuota real de tu casa de apuestas.</div>',
        unsafe_allow_html=True,
    )

    if league_key == "eliteserien":
        st.warning(
            "⚠️ En Eliteserien el 1X2 mejoró el baseline predictivo, "
            "pero dio ROI negativo contra cuotas de cierre. "
            "Este apartado es COMPARADOR / CONTEXTO, no genera picks."
        )
    else:
        st.info(
            "ℹ️ En LaLiga todavía no hemos validado ROI histórico del 1X2. "
            "Las cuotas justas son una referencia estadística, no una señal automática de apuesta."
        )

    probability_rows = []
    price_rows = []

    for match in predictions:
        g = match["goals"]

        outcomes = [
            (
                "1 · " + match["home_team"],
                float(g["home_win_raw"]),
            ),
            (
                "X · Empate",
                float(g["draw_raw"]),
            ),
            (
                "2 · " + match["away_team"],
                float(g["away_win_raw"]),
            ),
        ]

        probability_rows.append(
            {
                "Partido": (
                    f"{match['home_team']} - "
                    f"{match['away_team']}"
                ),
                "1 Local": (
                    float(g["home_win_raw"])
                    * 100.0
                ),
                "X Empate": (
                    float(g["draw_raw"])
                    * 100.0
                ),
                "2 Visitante": (
                    float(g["away_win_raw"])
                    * 100.0
                ),
            }
        )

        for selection, prob in outcomes:
            price_rows.append(
                {
                    "ID": (
                        f"{match['match_id']}|"
                        f"{selection}"
                    ),
                    "Partido": (
                        f"{match['home_team']} - "
                        f"{match['away_team']}"
                    ),
                    "Selección": selection,
                    "ProbRaw": prob,
                    "Prob. modelo %": (
                        prob * 100.0
                    ),
                    "Cuota justa": fair_odds(
                        prob
                    ),
                    "Cuota mín. EV": min_bet_odds(
                        prob,
                        min_ev,
                    ),
                }
            )

    st.markdown(
        "### 📊 Probabilidades del modelo"
    )

    st.dataframe(
        pd.DataFrame(
            probability_rows
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "1 Local": st.column_config.NumberColumn(
                "1 · Local",
                format="%.1f%%",
            ),
            "X Empate": st.column_config.NumberColumn(
                "X · Empate",
                format="%.1f%%",
            ),
            "2 Visitante": st.column_config.NumberColumn(
                "2 · Visitante",
                format="%.1f%%",
            ),
        },
    )

    st.markdown(
        """
        <div class="odds-hero">
            <div class="odds-title">💸 ¿Qué cuota debería tener cada selección?</div>
            <div class="odds-subtitle">
                Cuota justa = 1 / probabilidad del modelo. Si la casa ofrece una cuota más alta,
                el precio es mejor que el estimado por el modelo; aun así, 1X2 sigue siendo contexto
                hasta validar ROI por liga.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    price_df = pd.DataFrame(
        price_rows
    )

    odds_store = (
        st.session_state.setdefault(
            f"one_x_two_odds_{league_key}",
            {},
        )
    )

    price_df["Cuota casa"] = (
        price_df["ID"]
        .map(odds_store)
        .fillna(0.0)
        .astype(float)
    )

    edited_1x2 = st.data_editor(
        price_df[
            [
                "ID",
                "Partido",
                "Selección",
                "Prob. modelo %",
                "Cuota justa",
                "Cuota mín. EV",
                "Cuota casa",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=min(
            650,
            39 * (
                len(price_df) + 2
            ),
        ),
        key=f"one_x_two_editor_{league_key}",
        disabled=[
            "ID",
            "Partido",
            "Selección",
            "Prob. modelo %",
            "Cuota justa",
            "Cuota mín. EV",
        ],
        column_config={
            "ID": None,
            "Partido": st.column_config.TextColumn(
                "Partido",
                width="medium",
            ),
            "Selección": st.column_config.TextColumn(
                "Selección",
                width="medium",
            ),
            "Prob. modelo %": st.column_config.NumberColumn(
                "Modelo",
                format="%.1f%%",
                width="small",
            ),
            "Cuota justa": st.column_config.NumberColumn(
                "Cuota justa",
                format="%.2f",
                width="small",
            ),
            "Cuota mín. EV": st.column_config.NumberColumn(
                f"Mín. EV {min_ev_percent}%",
                format="%.2f",
                width="small",
            ),
            "Cuota casa": st.column_config.NumberColumn(
                "Casa",
                min_value=0.0,
                max_value=100.0,
                step=0.01,
                format="%.2f",
                width="small",
                help=(
                    "Introduce aquí la cuota decimal "
                    "de tu casa de apuestas."
                ),
            ),
        },
    )

    for _, row in edited_1x2.iterrows():
        row_id = row["ID"]
        odds = pd.to_numeric(
            row["Cuota casa"],
            errors="coerce",
        )

        if (
            pd.notna(odds)
            and float(odds) > 0
        ):
            odds_store[
                row_id
            ] = float(odds)
        else:
            odds_store.pop(
                row_id,
                None,
            )

    prob_map = dict(
        zip(
            price_df["ID"],
            price_df["ProbRaw"],
        )
    )

    compared = edited_1x2[
        pd.to_numeric(
            edited_1x2["Cuota casa"],
            errors="coerce",
        ) > 1.0
    ].copy()

    if not compared.empty:
        compared["ProbRaw"] = (
            compared["ID"]
            .map(prob_map)
            .astype(float)
        )

        compared["Diferencia teórica %"] = (
            compared["ProbRaw"]
            * pd.to_numeric(
                compared["Cuota casa"],
                errors="coerce",
            )
            - 1.0
        ) * 100.0

        compared["Precio"] = compared[
            "Diferencia teórica %"
        ].apply(
            lambda x: (
                "🟢 Por encima de justa"
                if x > 0
                else "🔴 Por debajo de justa"
            )
        )

        compared = compared.sort_values(
            "Diferencia teórica %",
            ascending=False,
        )

        st.markdown(
            "### 🧮 Comparación con la casa"
        )

        st.dataframe(
            compared[
                [
                    "Partido",
                    "Selección",
                    "Prob. modelo %",
                    "Cuota justa",
                    "Cuota casa",
                    "Diferencia teórica %",
                    "Precio",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Prob. modelo %": st.column_config.NumberColumn(
                    "Modelo",
                    format="%.1f%%",
                ),
                "Cuota justa": st.column_config.NumberColumn(
                    "Justa",
                    format="%.2f",
                ),
                "Cuota casa": st.column_config.NumberColumn(
                    "Casa",
                    format="%.2f",
                ),
                "Diferencia teórica %": st.column_config.NumberColumn(
                    "Dif. teórica",
                    format="%+.1f%%",
                ),
            },
        )

        st.caption(
            "La diferencia teórica usa probabilidad × cuota − 1. "
            "No convierte el 1X2 en mercado aprobado: en Noruega el backtest fue negativo "
            "y en LaLiga todavía no hemos validado ROI."
        )

