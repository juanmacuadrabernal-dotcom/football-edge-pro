
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from prediction_engine_v3 import PredictionEngine as EliteserienEngine

try:
    from prediction_engine_laliga_v4_cards_referee import PredictionEngineLaLiga
except Exception:
    PredictionEngineLaLiga = None


st.set_page_config(
    page_title="Football Edge Pro v15",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
TZ = "Europe/Madrid"


# -----------------------------------------------------------------------------
# ESTILO
# -----------------------------------------------------------------------------

def inject_css():
    st.markdown(
        """
        <style>
        :root{
            --bg:#07111f;
            --panel:#0b1728;
            --panel2:#0e1c30;
            --line:rgba(148,163,184,.16);
            --muted:#8fa4bf;
            --text:#f7fbff;
            --blue:#2477ff;
            --green:#2ee67b;
            --yellow:#ffbf3c;
            --red:#ff5b63;
        }
        .stApp{
            background:
              radial-gradient(circle at 74% 0%, rgba(22,79,145,.20), transparent 30%),
              linear-gradient(180deg,#07111f 0%,#071421 48%,#06101d 100%);
        }
        .block-container{
            max-width:1460px;
            padding-top:1rem;
            padding-bottom:3rem;
        }
        section[data-testid="stSidebar"]{
            background:linear-gradient(180deg,#07111f 0%,#091625 100%);
            border-right:1px solid var(--line);
        }
        section[data-testid="stSidebar"] > div{
            padding-top:.7rem;
        }
        h1,h2,h3,h4,p,span,label{letter-spacing:-.01em}
        .brand{
            border-bottom:1px solid var(--line);
            padding:8px 4px 18px 4px;
            margin-bottom:10px;
        }
        .brandline{display:flex;align-items:center;gap:11px}
        .brandlogo{
            font-size:26px;font-weight:900;color:var(--green);
            letter-spacing:-4px;
        }
        .brandtitle{font-size:20px;font-weight:900;color:var(--text)}
        .brandsub{font-size:12px;color:var(--muted);margin-top:2px}
        .page-title{
            color:var(--text);font-size:32px;font-weight:900;line-height:1.1;
            letter-spacing:-.035em;margin:3px 0 4px 0;
        }
        .page-sub{color:var(--muted);font-size:14px;margin-bottom:14px}
        .top-status{
            border:1px solid var(--line);background:rgba(8,21,36,.82);
            padding:10px 14px;border-radius:13px;color:#d8e7f7;font-size:12px;
        }
        .green-dot{
            display:inline-block;width:10px;height:10px;border-radius:50%;
            background:var(--green);box-shadow:0 0 14px rgba(46,230,123,.55);
            margin-right:8px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]{
            border:1px solid var(--line)!important;
            background:linear-gradient(145deg,rgba(11,23,40,.92),rgba(8,20,35,.94))!important;
            border-radius:16px!important;
            box-shadow:0 12px 30px rgba(0,0,0,.14);
        }
        .fixture-head{
            display:flex;justify-content:space-between;align-items:center;gap:10px
        }
        .fixture-name{font-weight:850;font-size:17px;color:var(--text)}
        .fixture-meta{font-size:12px;color:#a7bbd1;margin-top:4px}
        .fixture-interest{font-size:12px;color:var(--yellow);font-weight:800;margin-top:8px}
        .status-pre{
            display:inline-block;color:#8ff7ba;background:rgba(20,170,88,.14);
            border:1px solid rgba(46,230,123,.42);padding:4px 9px;border-radius:999px;
            font-size:10px;font-weight:900;
        }
        .status-live{
            display:inline-block;color:#ffb0b5;background:rgba(255,91,99,.13);
            border:1px solid rgba(255,91,99,.38);padding:4px 9px;border-radius:999px;
            font-size:10px;font-weight:900;
        }
        .match-hero{
            border:1px solid var(--line);border-radius:18px;
            background:
              radial-gradient(circle at 50% -30%,rgba(36,119,255,.17),transparent 45%),
              linear-gradient(145deg,rgba(11,23,40,.97),rgba(7,18,31,.98));
            padding:22px 24px;margin:5px 0 10px 0;
        }
        .versus{
            display:grid;grid-template-columns:1fr auto 1fr;gap:18px;
            align-items:center;text-align:center
        }
        .team-name{font-size:23px;font-weight:900;color:var(--text)}
        .team-badge{
            width:54px;height:54px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;margin:0 auto 8px auto;font-size:21px;font-weight:900;
            color:white;background:linear-gradient(145deg,#163150,#0b1d31);
            border:1px solid rgba(120,170,220,.25)
        }
        .kickoff{font-size:30px;font-weight:950;color:white}
        .kick-meta{font-size:12px;color:var(--muted);margin-top:4px}
        .analysis-band{
            padding:15px 18px;border:1px solid var(--line);border-radius:15px;
            background:rgba(9,22,38,.87);margin:12px 0;
        }
        .analysis-band-title{font-size:18px;font-weight:900;color:white}
        .analysis-band-sub{font-size:12px;color:var(--muted);margin-top:2px}
        .market-card{
            border:1px solid var(--line);border-radius:15px;padding:14px 15px;
            background:linear-gradient(145deg,#0b1829,#091523);
            height:100%;
        }
        .market-card.good{border-color:rgba(46,230,123,.65);box-shadow:0 0 0 1px rgba(46,230,123,.08)}
        .market-title{color:white;font-size:15px;font-weight:850;margin-bottom:10px}
        .market-row{
            display:flex;justify-content:space-between;gap:10px;
            color:#a9bbcf;font-size:12px;margin:6px 0
        }
        .market-row strong{color:#eef7ff;font-size:13px}
        .ev-good{color:var(--green)!important}
        .ev-bad{color:var(--red)!important}
        .tag{
            display:inline-block;border-radius:7px;padding:4px 8px;
            font-size:10px;font-weight:900;margin-top:7px;
        }
        .tag-strong{background:rgba(238,76,43,.15);color:#ff9a79;border:1px solid rgba(255,122,85,.25)}
        .tag-approved{background:rgba(46,230,123,.13);color:#83f3ad;border:1px solid rgba(46,230,123,.25)}
        .tag-secondary{background:rgba(255,191,60,.13);color:#ffd477;border:1px solid rgba(255,191,60,.25)}
        .tag-unvalidated{background:rgba(255,91,99,.12);color:#ff9ba1;border:1px solid rgba(255,91,99,.25)}
        .ref-card{
            padding:13px 15px;border:1px solid var(--line);border-radius:13px;
            background:rgba(7,19,33,.72);margin-bottom:10px
        }
        .ref-title{font-weight:900;color:white}
        .ref-meta{font-size:12px;color:#a9bdd4;margin-top:3px}
        .detail-box{
            border:1px solid var(--line);border-radius:13px;padding:14px;
            background:rgba(8,20,34,.72);height:100%
        }
        .detail-title{font-size:12px;color:#91a7c0;text-transform:uppercase;font-weight:850;margin-bottom:8px}
        .big-number{font-size:25px;font-weight:950;color:white}
        .opportunity{
            border:1px solid rgba(46,230,123,.35);border-radius:15px;
            background:linear-gradient(145deg,rgba(13,36,41,.74),rgba(8,21,34,.92));
            padding:13px 15px;margin-bottom:9px
        }
        .op-head{display:flex;justify-content:space-between;gap:8px;align-items:center}
        .op-title{font-weight:900;color:white}
        .op-ev{font-weight:950;color:var(--green)}
        .op-sub{font-size:12px;color:#a9bdd3;margin-top:5px}
        div[data-testid="stMetric"]{
            border:1px solid var(--line);border-radius:14px;background:rgba(9,22,38,.82);
            padding:11px 13px
        }
        .stButton>button,.stDownloadButton>button{
            border-radius:11px;font-weight:800;min-height:42px;
            border:1px solid rgba(100,150,205,.24)
        }
        .stButton>button[kind="primary"]{
            background:linear-gradient(135deg,#1d6cff,#2c83ff);
            border-color:#4092ff;color:white;
            box-shadow:0 8px 22px rgba(20,100,255,.22)
        }
        div[data-testid="stExpander"]{
            border:1px solid var(--line);border-radius:14px;background:rgba(7,18,31,.65)
        }
        div[data-baseweb="select"]>div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input{
            background:#0a1727!important;border-color:rgba(148,163,184,.20)!important;
        }
        .small-note{font-size:11px;color:#7f95ae}
        @media(max-width:760px){
            .block-container{padding:.7rem .65rem 2rem .65rem}
            .page-title{font-size:25px}
            .versus{gap:7px}
            .team-name{font-size:17px}
            .team-badge{width:45px;height:45px;font-size:17px}
            .kickoff{font-size:23px}
            .match-hero{padding:16px 10px}
            .stButton>button{min-height:46px}
            section[data-testid="stSidebar"]{min-width:280px}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# DATOS / MOTORES
# -----------------------------------------------------------------------------

LEAGUES = {
    "🇪🇸 LaLiga": {"key": "laliga", "short": "LaLiga"},
    "🇳🇴 Eliteserien": {"key": "eliteserien", "short": "Eliteserien"},
}


@st.cache_resource
def get_engine(league_key: str):
    if league_key == "eliteserien":
        return EliteserienEngine()
    if league_key == "laliga":
        if PredictionEngineLaLiga is None:
            raise RuntimeError("No se pudo importar el motor de LaLiga.")
        return PredictionEngineLaLiga()
    raise ValueError(league_key)


@st.cache_data(ttl=300)
def get_predictions(league_key: str):
    return get_engine(league_key).predict_next_round()


def run_script(filename: str, timeout: int = 1800):
    script = BASE_DIR / filename
    if not script.exists():
        return False, f"Falta {filename} en la carpeta del proyecto."
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "La actualización superó el tiempo máximo."
    output = ((r.stdout or "") + ("\n" + r.stderr if r.stderr else "")).strip()
    return r.returncode == 0, output


def fair_odds(prob):
    try:
        p = float(prob)
    except Exception:
        return None
    return 1.0 / p if p > 0 else None


def min_odds(prob, min_ev):
    try:
        p = float(prob)
    except Exception:
        return None
    return (1.0 + min_ev) / p if p > 0 else None


def ev_value(prob, odds):
    try:
        p, o = float(prob), float(odds)
    except Exception:
        return None
    return p * o - 1.0 if o > 1.0 else None


def madrid_time(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(TZ)


def now_madrid():
    return pd.Timestamp.now(tz=TZ)


def is_prematch(match):
    return now_madrid() < madrid_time(match["utc_time"])


def initials(name):
    words = [w for w in str(name).replace("-", " ").split() if w]
    if not words:
        return "FC"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[-1][0]).upper()


def last_data_update(league_key):
    path = BASE_DIR / "data" / ("laliga.db" if league_key == "laliga" else "eliteserien.db")
    if not path.exists():
        return "sin dato"
    try:
        ts = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").tz_convert(TZ)
        return ts.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return "sin dato"


def strength_tag(level):
    t = str(level).upper()
    if "FUERTE" in t:
        return '<span class="tag tag-strong">🔥 FUERTE</span>'
    if "APROBADO" in t:
        return '<span class="tag tag-approved">✅ APROBADO</span>'
    if "NO VALIDADO" in t:
        return '<span class="tag tag-unvalidated">⚠️ NO VALIDADO</span>'
    return '<span class="tag tag-secondary">⚠️ SECUNDARIO</span>'


def markets_for_match(match, league_key):
    rows = []

    def add(category, key, label, level, prob, extra=None):
        if prob is None:
            return
        rows.append({
            "id": f"{match['match_id']}|{category}|{key}",
            "match_id": match["match_id"],
            "category": category,
            "key": key,
            "label": label,
            "level": level,
            "prob": float(prob),
            "extra": extra or {},
        })

    goals = match.get("goals", {})
    if league_key == "laliga":
        for line, level in [(1.5, "FUERTE"), (2.5, "APROBADO"), (3.5, "FUERTE")]:
            key = f"over_{str(line).replace('.', '_')}"
            if key in goals:
                add("goals", key, f"Más de {line} goles", level, goals[key])
    else:
        for line, level in [(2.5, "FUERTE"), (3.5, "SECUNDARIO")]:
            key = f"over_{str(line).replace('.', '_')}"
            if key in goals:
                add("goals", key, f"Más de {line} goles", level, goals[key])

    corners = match.get("corners", {})
    if league_key == "laliga":
        lines = [(8.5, "FUERTE"), (10.5, "APROBADO")]
    else:
        lines = [(9.5, "APROBADO"), (10.5, "FUERTE"), (11.5, "FUERTE")]
    for line, level in lines:
        key = f"over_{str(line).replace('.', '_')}"
        if key in corners:
            add("corners", key, f"Más de {line} córners", level, corners[key])

    cards = match.get("cards", {})
    if league_key == "laliga":
        card_lines = [(2.5, "FUERTE"), (3.5, "FUERTE"), (4.5, "FUERTE"), (5.5, "FUERTE")]
    else:
        card_lines = [(3.5, "SECUNDARIO"), (4.5, "APROBADO"), (5.5, "FUERTE")]
    ref_info = cards.get("referee_info", {}) if isinstance(cards, dict) else {}
    for line, level in card_lines:
        key = f"over_{str(line).replace('.', '_')}"
        if key in cards:
            add(
                "cards", key, f"Más de {line} tarjetas", level, cards[key],
                {"referee_info": ref_info,
                 "ref_effect": cards.get("referee_effect_probability", {}).get(key, 0.0)}
            )

    if league_key == "laliga":
        sot = match.get("shots_on_target", {})
        total_strength = {7.5:"APROBADO",8.5:"FUERTE",9.5:"APROBADO",10.5:"FUERTE",11.5:"SECUNDARIO"}
        for line, level in total_strength.items():
            key = f"total_over_{str(line).replace('.', '_')}"
            if key in sot:
                add("sot", key, f"Más de {line} SOT totales", level, sot[key])
        for side, team_key in [("home","home_team"),("away","away_team")]:
            for line in (2.5,3.5,4.5,5.5,6.5):
                key = f"{side}_over_{str(line).replace('.', '_')}"
                if key in sot:
                    level = "SECUNDARIO" if line == 6.5 else "FUERTE"
                    add("sot", key, f"{match[team_key]} +{line} SOT", level, sot[key])

    # 1X2: siempre contexto, no mercado aprobado.
    if all(k in goals for k in ("home_win_raw", "draw_raw", "away_win_raw")):
        add("1x2", "home", f"{match['home_team']} gana", "NO VALIDADO", goals["home_win_raw"])
        add("1x2", "draw", "Empate", "NO VALIDADO", goals["draw_raw"])
        add("1x2", "away", f"{match['away_team']} gana", "NO VALIDADO", goals["away_win_raw"])

    return rows


def categories_for_league(league_key):
    cats = [
        ("goals", "⚽", "Goles"),
        ("cards", "🟨", "Tarjetas"),
        ("corners", "🚩", "Córners"),
    ]
    if league_key == "laliga":
        cats.append(("sot", "🎯", "Remates a puerta"))
    cats.append(("1x2", "🏆", "1X2"))
    return cats


def odds_store():
    return st.session_state.setdefault("v15_odds", {})


def bookmaker_names():
    return st.session_state.setdefault("v15_bookmakers", ["Bet365", "Unibet", "Betano"])


def best_market_odds(market_id):
    values = odds_store().get(market_id, {})
    valid = [(b, float(v)) for b, v in values.items() if v is not None and float(v) > 1.0]
    if not valid:
        return None, None
    return max(valid, key=lambda x: x[1])


def market_analysis(row, min_ev):
    book, odd = best_market_odds(row["id"])
    ev = ev_value(row["prob"], odd) if odd else None
    return {
        **row,
        "fair": fair_odds(row["prob"]),
        "minimum": min_odds(row["prob"], min_ev),
        "book": book,
        "odds": odd,
        "ev": ev,
    }


def match_interest_count(match, league_key, min_ev):
    rows = [market_analysis(r, min_ev) for r in markets_for_match(match, league_key) if r["category"] != "1x2"]
    with_odds = [r for r in rows if r["ev"] is not None]
    if with_odds:
        return sum(r["ev"] >= min_ev for r in with_odds)
    # Sin proveedor de cuotas aún: sirve solo para priorizar qué mirar, NO es valor.
    return sum(r["prob"] >= 0.55 and "SECUNDARIO" not in r["level"] for r in rows)


def match_by_id(predictions, match_id):
    for m in predictions:
        if str(m["match_id"]) == str(match_id):
            return m
    return None


# -----------------------------------------------------------------------------
# NAVEGACIÓN
# -----------------------------------------------------------------------------

def set_page(page, match_id=None, market=None):
    st.session_state["v15_page"] = page
    if match_id is not None:
        st.session_state["v15_match_id"] = str(match_id)
    if market is not None:
        st.session_state["v15_market"] = market
    st.rerun()


def sidebar(league_key):
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
              <div class="brandline">
                <div class="brandlogo">▮▮▮</div>
                <div>
                  <div class="brandtitle">Football Edge Pro</div>
                  <div class="brandsub">Datos. Modelos. Valor.</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page_labels = {
            "jornada": "🏠  Jornada",
            "oportunidades": "🔥  Mejores oportunidades",
            "historico": "🕘  Histórico",
            "ajustes": "⚙️  Ajustes",
        }
        current = st.session_state.get("v15_page", "jornada")
        if current == "partido":
            current = "jornada"

        chosen = st.radio(
            "Navegación",
            list(page_labels),
            format_func=lambda x: page_labels[x],
            index=list(page_labels).index(current if current in page_labels else "jornada"),
            label_visibility="collapsed",
        )
        if chosen != current:
            st.session_state["v15_page"] = chosen
            st.session_state.pop("v15_market", None)
            st.rerun()

        st.markdown("#### LIGA ACTUAL")
        labels = list(LEAGUES.keys())
        current_label = next(k for k,v in LEAGUES.items() if v["key"] == league_key)
        selected = st.selectbox("Liga", labels, index=labels.index(current_label), label_visibility="collapsed")
        new_key = LEAGUES[selected]["key"]
        if new_key != league_key:
            st.session_state["v15_league"] = new_key
            st.session_state["v15_page"] = "jornada"
            st.session_state.pop("v15_match_id", None)
            st.rerun()

        st.markdown("<br><div class='small-note'>Football Edge Pro v15<br>Analistas, no apostadores.</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CABECERA / JORNADA
# -----------------------------------------------------------------------------

def top_header(league_key, predictions):
    first = predictions[0]
    rnd = first.get("round")
    rnd = str(rnd) if pd.notna(rnd) else "próxima"
    c1, c2 = st.columns([4, 1.6])
    with c1:
        league_name = "LaLiga" if league_key == "laliga" else "Eliteserien"
        st.markdown(f'<div class="page-title">{league_name} – Jornada {rnd}</div>', unsafe_allow_html=True)
        times = [madrid_time(m["utc_time"]) for m in predictions]
        if times:
            st.markdown(
                f'<div class="page-sub">📅 {min(times).strftime("%d")} – {max(times).strftime("%d %B %Y")} · {len(predictions)} partidos</div>',
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown(
            f'<div class="top-status"><span class="green-dot"></span><b>Solo cuotas prepartido</b><br>'
            f'<span style="color:#8298b0">Datos: {last_data_update(league_key)}</span></div>',
            unsafe_allow_html=True,
        )


def render_fixture_card(match, league_key, min_ev):
    kick = madrid_time(match["utc_time"])
    pre = is_prematch(match)
    status = '<span class="status-pre">PREPARTIDO</span>' if pre else '<span class="status-live">INICIADO</span>'
    interest = match_interest_count(match, league_key, min_ev)

    st.markdown(
        f"""
        <div class="fixture-head">
          <div>
            <div class="fixture-name">{match['home_team']} <span style="color:#5f7792">vs</span> {match['away_team']}</div>
            <div class="fixture-meta">{kick.strftime('%a, %d %b · %H:%M')}</div>
          </div>
          <div>{status}</div>
        </div>
        <div class="fixture-interest">{interest} mercados interesantes para revisar ›</div>
        """,
        unsafe_allow_html=True,
    )


def render_jornada(predictions, league_key, min_ev):
    top_header(league_key, predictions)

    filter_choice = st.radio(
        "Filtro",
        ["Todos", "Hoy", "Mañana"],
        horizontal=True,
        label_visibility="collapsed",
        key=f"filter_{league_key}",
    )

    today = now_madrid().date()
    tomorrow = (now_madrid() + pd.Timedelta(days=1)).date()
    shown = []
    for m in predictions:
        d = madrid_time(m["utc_time"]).date()
        if filter_choice == "Hoy" and d != today:
            continue
        if filter_choice == "Mañana" and d != tomorrow:
            continue
        shown.append(m)

    if not shown:
        st.info("No hay partidos en ese filtro.")
        return

    for i in range(0, len(shown), 2):
        cols = st.columns(2, gap="medium")
        for col, match in zip(cols, shown[i:i+2]):
            with col:
                with st.container(border=True):
                    render_fixture_card(match, league_key, min_ev)
                    if st.button(
                        "Analizar partido  →",
                        key=f"open_{league_key}_{match['match_id']}",
                        use_container_width=True,
                        type="primary" if i == 0 else "secondary",
                    ):
                        set_page("partido", match["match_id"])


# -----------------------------------------------------------------------------
# PARTIDO
# -----------------------------------------------------------------------------

def countdown_text(match):
    diff = madrid_time(match["utc_time"]) - now_madrid()
    if diff.total_seconds() <= 0:
        return "Partido iniciado"
    minutes = int(diff.total_seconds() // 60)
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {mins}m"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def render_match_hero(match):
    kick = madrid_time(match["utc_time"])
    pre = is_prematch(match)
    status = '<span class="status-pre">PREPARTIDO</span>' if pre else '<span class="status-live">INICIADO</span>'
    st.markdown(
        f"""
        <div class="match-hero">
          <div style="text-align:right;margin-bottom:8px">{status}</div>
          <div class="versus">
            <div>
              <div class="team-badge">{initials(match['home_team'])}</div>
              <div class="team-name">{match['home_team']}</div>
            </div>
            <div>
              <div class="kick-meta">{kick.strftime('%A, %d %B %Y')}</div>
              <div class="kickoff">{kick.strftime('%H:%M')}</div>
              <div class="kick-meta">Tiempo para el inicio: {countdown_text(match)}</div>
            </div>
            <div>
              <div class="team-badge">{initials(match['away_team'])}</div>
              <div class="team-name">{match['away_team']}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def category_name(category):
    return {
        "goals":"⚽ Goles",
        "cards":"🟨 Tarjetas",
        "corners":"🚩 Córners",
        "sot":"🎯 Remates a puerta",
        "1x2":"🏆 1X2",
    }.get(category, category)


def odds_inputs(row, locked):
    names = bookmaker_names()[:3]
    saved = odds_store().setdefault(row["id"], {})
    cols = st.columns(len(names)) if names else []
    for col, name in zip(cols, names):
        default = float(saved.get(name, 0.0) or 0.0)
        with col:
            val = st.number_input(
                name,
                min_value=0.0,
                max_value=100.0,
                value=default,
                step=0.01,
                format="%.2f",
                key=f"od_{row['id']}_{name}",
                disabled=locked,
            )
            if val > 1.0:
                saved[name] = float(val)
            else:
                saved.pop(name, None)


def market_card(row, min_ev):
    a = market_analysis(row, min_ev)
    good = a["ev"] is not None and a["ev"] >= min_ev
    ev_text = "—" if a["ev"] is None else f"{a['ev']*100:+.1f}%"
    best = "Esperando cuota" if a["odds"] is None else f"{a['odds']:.2f} ({a['book']})"
    css_ev = "ev-good" if good else ("ev-bad" if a["ev"] is not None and a["ev"] < 0 else "")
    st.markdown(
        f"""
        <div class="market-card {'good' if good else ''}">
          <div class="market-title">{row['label']}</div>
          <div class="market-row"><span>Probabilidad modelo</span><strong>{row['prob']*100:.1f}%</strong></div>
          <div class="market-row"><span>Cuota justa</span><strong>{a['fair']:.2f}</strong></div>
          <div class="market-row"><span>Cuota mínima EV</span><strong>{a['minimum']:.2f}</strong></div>
          <div class="market-row"><span>Mejor cuota</span><strong>{best}</strong></div>
          <div class="market-row"><span>Valor esperado</span><strong class="{css_ev}">{ev_text}</strong></div>
          {strength_tag(row['level'])}
        </div>
        """,
        unsafe_allow_html=True,
    )


def detailed_analysis(match, row, min_ev):
    a = market_analysis(row, min_ev)
    with st.expander(f"📊 Ver análisis completo · {row['label']}", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="detail-title">MODELO</div>', unsafe_allow_html=True)
            st.metric("Probabilidad", f"{row['prob']*100:.1f}%")
            st.metric("Cuota justa", f"{a['fair']:.2f}")
        with c2:
            st.markdown('<div class="detail-title">PRECIO</div>', unsafe_allow_html=True)
            st.metric("Cuota mínima", f"{a['minimum']:.2f}")
            st.metric("Mejor cuota", "—" if a["odds"] is None else f"{a['odds']:.2f}")
        with c3:
            st.markdown('<div class="detail-title">VALOR</div>', unsafe_allow_html=True)
            st.metric("EV", "—" if a["ev"] is None else f"{a['ev']*100:+.1f}%")
            st.markdown(strength_tag(row["level"]), unsafe_allow_html=True)

        if row["category"] == "cards":
            ref = row.get("extra", {}).get("referee_info", {}) or {}
            assigned = ref.get("assigned", False)
            refname = ref.get("name") or match.get("referee") or "Pendiente"
            profile = ref.get("profile", "PENDIENTE")
            avg = ref.get("avg_yellow")
            delta = ref.get("delta_vs_league")
            st.markdown("#### 🧑‍⚖️ Árbitro")
            st.write(f"**{refname}** · {profile}")
            if assigned:
                if avg is not None and not pd.isna(avg):
                    st.write(f"Media histórica: **{float(avg):.2f} tarjetas**")
                if delta is not None and not pd.isna(delta):
                    st.write(f"Impacto frente a la media de liga: **{float(delta):+.2f} tarjetas**")
                ref_eff = row.get("extra", {}).get("ref_effect")
                if ref_eff is not None:
                    st.write(f"Impacto sobre esta probabilidad: **{float(ref_eff)*100:+.1f} pp**")
            else:
                st.caption("Árbitro todavía no asignado. El motor usa comportamiento neutral de liga.")

        elif row["category"] == "goals":
            g = match.get("goals", {})
            st.markdown("#### ⚽ Lectura del modelo")
            st.write(
                f"Goles esperados del modelo: **{float(g.get('lambda_home', 0)):.2f} "
                f"{match['home_team']} + {float(g.get('lambda_away', 0)):.2f} {match['away_team']}**."
            )

        elif row["category"] == "corners":
            c = match.get("corners", {})
            st.markdown("#### 🚩 Lectura del modelo")
            st.write(
                f"Córners esperados: **{float(c.get('lambda_home', 0)):.2f} local + "
                f"{float(c.get('lambda_away', 0)):.2f} visitante**."
            )

        elif row["category"] == "sot":
            s = match.get("shots_on_target", {})
            st.markdown("#### 🎯 Lectura del modelo")
            st.write(
                f"SOT esperados: **{float(s.get('lambda_home', 0)):.2f} {match['home_team']} + "
                f"{float(s.get('lambda_away', 0)):.2f} {match['away_team']}**."
            )

        elif row["category"] == "1x2":
            st.warning(
                "1X2 es contexto analítico. No lo tratamos como mercado aprobado: "
                "Noruega tuvo ROI histórico negativo y LaLiga aún no tiene ROI validado."
            )

        st.markdown("#### 📝 Ficha rápida para Telegram")
        txt = (
            f"{match['home_team']} vs {match['away_team']}\n"
            f"Mercado: {row['label']}\n"
            f"Modelo: {row['prob']*100:.1f}%\n"
            f"Cuota justa: {a['fair']:.2f}\n"
            f"Cuota mínima EV: {a['minimum']:.2f}\n"
            f"Mejor cuota: {'—' if a['odds'] is None else f'{a['odds']:.2f} ({a['book']})'}\n"
            f"EV: {'—' if a['ev'] is None else f'{a['ev']*100:+.1f}%'}\n"
            f"Nivel: {row['level']}"
        )
        st.code(txt, language=None)


def render_category(match, league_key, category, min_ev):
    if st.button("← Volver al partido", key=f"back_market_{match['match_id']}"):
        st.session_state["v15_market"] = None
        st.rerun()

    st.markdown(f'<div class="page-title">{category_name(category)}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-sub">{match["home_team"]} vs {match["away_team"]} · análisis prepartido</div>',
        unsafe_allow_html=True,
    )

    if category == "1x2":
        st.warning("⚠️ 1X2 se mantiene como contexto. No genera señal automática de BET.")

    if category == "cards":
        cards = match.get("cards", {})
        ref = cards.get("referee_info", {}) if isinstance(cards, dict) else {}
        refname = ref.get("name") or match.get("referee") or "Pendiente"
        profile = ref.get("profile", "PENDIENTE")
        avg = ref.get("avg_yellow")
        avg_txt = "sin histórico" if avg is None or pd.isna(avg) else f"{float(avg):.2f} tarjetas"
        st.markdown(
            f"""
            <div class="ref-card">
              <div class="ref-title">🧑‍⚖️ {refname} <span class="tag tag-secondary">{profile}</span></div>
              <div class="ref-meta">Media histórica: {avg_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    rows = [r for r in markets_for_match(match, league_key) if r["category"] == category]
    if not rows:
        st.info("Este motor no tiene mercados disponibles en esta categoría.")
        return

    locked = not is_prematch(match)
    if locked:
        st.error("🔴 Partido iniciado. Las cuotas quedan bloqueadas: no se buscan nuevos errores de precio en live.")
    else:
        st.caption("Cuotas prepartido. Hasta conectar la API, puedes introducir las 1–3 casas elegidas como respaldo manual.")

    for row in rows:
        with st.container(border=True):
            market_card(row, min_ev)
            if not locked:
                odds_inputs(row, locked=False)
            detailed_analysis(match, row, min_ev)


def render_search_value(match, league_key, min_ev):
    if st.button("← Volver al partido", key=f"back_value_{match['match_id']}"):
        st.session_state["v15_market"] = None
        st.rerun()

    st.markdown('<div class="page-title">🔍 Buscar errores de cuota</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-sub">{match["home_team"]} vs {match["away_team"]} · todos los mercados en un solo escaneo</div>',
        unsafe_allow_html=True,
    )

    rows = [
        market_analysis(r, min_ev)
        for r in markets_for_match(match, league_key)
        if r["category"] != "1x2"
    ]
    quoted = [r for r in rows if r["ev"] is not None]
    value = sorted([r for r in quoted if r["ev"] >= min_ev], key=lambda x: x["ev"], reverse=True)

    if not is_prematch(match):
        st.error("🔴 Partido iniciado. El escáner queda cerrado para nuevas cuotas.")
    elif not quoted:
        st.info(
            "Todavía no hay cuotas cargadas para este partido. La V15 ya está preparada "
            "para el conector automático de 1–3 casas; mientras lo conectamos, entra en un mercado "
            "y usa las cuotas manuales de respaldo."
        )
        # Aun sin cuotas, ordenamos por probabilidad como guía analítica.
        review = sorted(rows, key=lambda x: (("FUERTE" in x["level"]), x["prob"]), reverse=True)[:5]
        st.markdown("### Mercados que el modelo prioriza para revisar")
        for r in review:
            st.markdown(
                f"""
                <div class="opportunity">
                  <div class="op-head"><div class="op-title">{r['label']}</div><div>{strength_tag(r['level'])}</div></div>
                  <div class="op-sub">Modelo {r['prob']*100:.1f}% · justa {r['fair']:.2f} · mínima EV {r['minimum']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    st.markdown(
        f"""
        <div class="analysis-band">
          <div class="analysis-band-title">🔥 {len(value)} oportunidades superan EV ≥ {min_ev*100:.0f}%</div>
          <div class="analysis-band-sub">Ordenadas por valor esperado. La app filtra; la decisión final sigue siendo del analista.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not value:
        st.success("✅ NO BET claro: ninguna cuota cargada supera nuestro EV mínimo.")
    else:
        for rank, r in enumerate(value, 1):
            st.markdown(
                f"""
                <div class="opportunity">
                  <div class="op-head">
                    <div class="op-title">{rank}. {r['label']}</div>
                    <div class="op-ev">EV {r['ev']*100:+.1f}%</div>
                  </div>
                  <div class="op-sub">
                    Modelo {r['prob']*100:.1f}% · justa {r['fair']:.2f} ·
                    {r['odds']:.2f} en {r['book']} · {r['level']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_match(match, league_key, min_ev):
    if st.button("← Volver a la jornada", key=f"back_match_{match['match_id']}"):
        set_page("jornada")

    render_match_hero(match)

    selected_market = st.session_state.get("v15_market")
    if selected_market == "all":
        render_search_value(match, league_key, min_ev)
        return
    if selected_market:
        render_category(match, league_key, selected_market, min_ev)
        return

    if not is_prematch(match):
        st.error("🔴 Partido iniciado · la búsqueda de valor queda cerrada. Solo consulta el análisis ya calculado.")

    if st.button(
        "🔍 Buscar errores de cuota en este partido",
        key=f"scan_{match['match_id']}",
        type="primary",
        use_container_width=True,
        disabled=not is_prematch(match),
    ):
        st.session_state["v15_market"] = "all"
        st.rerun()

    st.markdown("### ¿Qué quieres analizar?")
    cats = categories_for_league(league_key)
    cols = st.columns(3)
    for idx, (cat, icon, label) in enumerate(cats):
        with cols[idx % 3]:
            if st.button(
                f"{icon}\n\n{label}",
                key=f"cat_{match['match_id']}_{cat}",
                use_container_width=True,
            ):
                st.session_state["v15_market"] = cat
                st.rerun()

    # Resumen analítico rápido.
    st.markdown("### Lectura rápida del partido")
    c1, c2, c3 = st.columns(3)
    g = match.get("goals", {})
    c = match.get("corners", {})
    cards = match.get("cards", {})
    with c1:
        st.metric("⚽ Goles esperados", f"{float(g.get('lambda_home',0))+float(g.get('lambda_away',0)):.2f}")
    with c2:
        st.metric("🚩 Córners esperados", f"{float(c.get('lambda_home',0))+float(c.get('lambda_away',0)):.2f}")
    with c3:
        st.metric("🟨 Tarjetas esperadas", f"{float(cards.get('mu_total',0)):.2f}" if "mu_total" in cards else "—")

    ref = match.get("referee")
    if ref is not None and not pd.isna(ref):
        st.caption(f"🧑‍⚖️ Árbitro asignado: {ref}")


# -----------------------------------------------------------------------------
# OPORTUNIDADES / HISTÓRICO / AJUSTES
# -----------------------------------------------------------------------------

def render_global_opportunities(predictions, league_key, min_ev):
    st.markdown('<div class="page-title">🔥 Mejores oportunidades</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Todos los partidos de la jornada · solo cuotas prepartido · ordenadas por EV</div>', unsafe_allow_html=True)

    rows = []
    for match in predictions:
        if not is_prematch(match):
            continue
        for r in markets_for_match(match, league_key):
            if r["category"] == "1x2":
                continue
            a = market_analysis(r, min_ev)
            if a["ev"] is not None:
                a["match"] = f"{match['home_team']} vs {match['away_team']}"
                rows.append(a)

    value = sorted([r for r in rows if r["ev"] >= min_ev], key=lambda x: x["ev"], reverse=True)
    if not rows:
        st.info(
            "No hay cuotas cargadas todavía. Cuando conectemos la API de las 1–3 casas elegidas, "
            "esta pantalla será el ranking automático de la jornada."
        )
        return
    if not value:
        st.success("✅ Ninguna cuota cargada supera el EV mínimo. NO BET es una decisión válida.")
        return

    for rank, r in enumerate(value, 1):
        st.markdown(
            f"""
            <div class="opportunity">
              <div class="op-head">
                <div class="op-title">{rank}. {r['label']} <span style="color:#7890aa">· {r['match']}</span></div>
                <div class="op-ev">EV {r['ev']*100:+.1f}%</div>
              </div>
              <div class="op-sub">Modelo {r['prob']*100:.1f}% · cuota {r['odds']:.2f} ({r['book']}) · justa {r['fair']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history():
    st.markdown('<div class="page-title">🕘 Histórico</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Zona reservada para el Pick Tracker y rendimiento real de nuestras selecciones.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.info(
            "La interfaz está preparada, pero no inventamos resultados: todavía no hay un registro persistente "
            "de picks V15. Cuando activemos el Pick Tracker, aquí veremos apuesta, cuota prepartido guardada, "
            "resultado, ROI, yield y rendimiento por mercado."
        )


def render_settings(league_key, min_ev_percent):
    st.markdown('<div class="page-title">⚙️ Ajustes</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Configuración de análisis. Máximo 3 casas y siempre prepartido.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Configuración general")
        st.write(f"**Liga activa:** {'LaLiga' if league_key=='laliga' else 'Eliteserien'}")
        st.write(f"**EV mínimo actual:** {min_ev_percent}%")
        st.checkbox("Solo cuotas prepartido", value=True, disabled=True)
        st.caption("Cuando empieza el partido, la V15 bloquea la introducción/búsqueda de nuevas cuotas.")

    with st.container(border=True):
        st.markdown("### Casas de apuestas (máximo 3)")
        names = bookmaker_names()
        new_names = []
        for i in range(3):
            new_names.append(
                st.text_input(
                    f"Casa {i+1}",
                    value=names[i] if i < len(names) else "",
                    key=f"book_name_{i}",
                ).strip()
            )
        if st.button("Guardar casas", use_container_width=True):
            st.session_state["v15_bookmakers"] = [x for x in new_names if x][:3]
            st.success("Casas guardadas.")

        st.caption(
            "Conector automático de cuotas: preparado para la siguiente fase. "
            "Antes de activarlo elegiremos un proveedor con cobertura real de goles, tarjetas, córners y SOT. "
            "Nada de cuotas live."
        )

    with st.container(border=True):
        st.markdown("### Actualización de datos")
        st.info(
            "La actualización persistente se hace desde los BAT locales que ya tenemos. "
            "Así se actualizan datos/modelos, se suben a GitHub y Streamlit redepliega con la nueva versión."
        )
        st.caption(
            "No actualizamos la base directamente desde Streamlit Cloud porque sus cambios locales no deben usarse como almacenamiento persistente."
        )


# -----------------------------------------------------------------------------
# APP
# -----------------------------------------------------------------------------

inject_css()

st.session_state.setdefault("v15_page", "jornada")
st.session_state.setdefault("v15_league", "laliga")
st.session_state.setdefault("v15_min_ev", 5)

league_key = st.session_state["v15_league"]
sidebar(league_key)

# Sidebar puede haber cambiado liga.
league_key = st.session_state.get("v15_league", league_key)

# EV en un control pequeño arriba de la app; ajustes lo explica, no invade la pantalla.
with st.sidebar:
    st.markdown("---")
    min_ev_percent = st.slider(
        "EV mínimo",
        min_value=0,
        max_value=20,
        value=int(st.session_state.get("v15_min_ev", 5)),
        step=1,
    )
    st.session_state["v15_min_ev"] = min_ev_percent
    min_ev = min_ev_percent / 100.0

try:
    predictions = get_predictions(league_key)
except Exception as exc:
    st.error("No se ha podido cargar el motor de la liga.")
    st.exception(exc)
    st.stop()

if not predictions:
    st.warning("No hay una próxima jornada disponible para analizar.")
    st.stop()

page = st.session_state.get("v15_page", "jornada")

if page == "jornada":
    render_jornada(predictions, league_key, min_ev)

elif page == "partido":
    match = match_by_id(predictions, st.session_state.get("v15_match_id"))
    if match is None:
        st.session_state["v15_page"] = "jornada"
        st.warning("Ese partido ya no pertenece a la próxima jornada.")
    else:
        render_match(match, league_key, min_ev)

elif page == "oportunidades":
    render_global_opportunities(predictions, league_key, min_ev)

elif page == "historico":
    render_history()

elif page == "ajustes":
    render_settings(league_key, min_ev_percent)
