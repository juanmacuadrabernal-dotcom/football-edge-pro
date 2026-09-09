
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from prediction_engine_v3 import PredictionEngine as EliteserienEngine
from football_odds_oddspapi import populate_match_odds, get_api_key, DEFAULT_BOOKMAKERS

try:
    from prediction_engine_laliga_v4_cards_referee import PredictionEngineLaLiga
except Exception:
    PredictionEngineLaLiga = None


st.set_page_config(
    page_title="Football Edge Pro v15.1",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
TZ = "Europe/Madrid"

LEAGUES = {
    "🇪🇸 LaLiga": "laliga",
    "🇳🇴 Eliteserien": "eliteserien",
}


# =========================================================
# VISUAL
# =========================================================

def css():
    st.markdown(
        """
        <style>
        :root{
            --bg:#07111d;
            --panel:#0b1828;
            --panel2:#0e2034;
            --line:#183047;
            --line2:#234663;
            --text:#f7fbff;
            --muted:#94a8bd;
            --blue:#2d7dff;
            --green:#29e578;
            --yellow:#ffc13b;
            --red:#ff5f6d;
        }

        html, body, [class*="css"]{font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif}
        .stApp{
            background:
                radial-gradient(circle at 62% -5%, rgba(30,103,186,.18), transparent 27%),
                linear-gradient(180deg,#07111d,#071521 58%,#06101b);
            color:var(--text);
        }

        /* FIX CABECERA CORTADA */
        .block-container{
            max-width:1500px;
            padding-top:4.6rem !important;
            padding-left:1.35rem;
            padding-right:1.35rem;
            padding-bottom:3rem;
        }

        header[data-testid="stHeader"]{
            background:rgba(7,17,29,.94);
            backdrop-filter:blur(8px);
        }

        section[data-testid="stSidebar"]{
            background:linear-gradient(180deg,#07101c,#091827);
            border-right:1px solid rgba(120,165,205,.15);
        }
        section[data-testid="stSidebar"] > div{padding-top:1.3rem}
        section[data-testid="stSidebar"] .stRadio label{
            padding:.28rem .18rem;
            border-radius:9px;
        }

        .brand{
            padding:8px 4px 18px;
            border-bottom:1px solid rgba(148,163,184,.15);
            margin-bottom:12px;
        }
        .brand-flex{display:flex;gap:11px;align-items:center}
        .brand-mark{
            width:27px;height:31px;display:flex;align-items:flex-end;gap:3px
        }
        .brand-mark i{display:block;width:6px;background:var(--green);border-radius:2px}
        .brand-mark i:nth-child(1){height:16px}
        .brand-mark i:nth-child(2){height:26px}
        .brand-mark i:nth-child(3){height:21px}
        .brand-title{font-size:20px;font-weight:900;color:white}
        .brand-sub{font-size:11px;color:var(--muted);margin-top:1px}

        .title{
            font-size:31px;font-weight:950;color:white;letter-spacing:-.045em;
            line-height:1.08;margin-bottom:4px
        }
        .subtitle{font-size:13px;color:#8fa5bb;margin-bottom:12px}
        .statusbox{
            border:1px solid var(--line);border-radius:12px;padding:10px 13px;
            background:linear-gradient(145deg,#0a1727,#091522);
            font-size:11px;color:#cfe1f0;line-height:1.7
        }
        .green-dot{
            display:inline-block;width:9px;height:9px;border-radius:50%;
            background:var(--green);margin-right:7px;box-shadow:0 0 12px rgba(41,229,120,.6)
        }

        /* fixture strip */
        .fixture-card{
            min-height:168px;
            border:1px solid var(--line);
            border-radius:13px;
            background:linear-gradient(160deg,#0c1b2c,#081522);
            padding:13px 10px 11px;
            text-align:center;
            position:relative;
            box-shadow:0 10px 22px rgba(0,0,0,.13);
        }
        .fixture-card.selected{
            border:2px solid var(--green);
            box-shadow:0 0 0 1px rgba(41,229,120,.16),0 0 24px rgba(41,229,120,.08);
        }
        .mini-teams{
            display:flex;justify-content:center;align-items:center;gap:8px;margin:3px 0 9px
        }
        .mini-badge{
            width:38px;height:38px;border-radius:50%;
            display:flex;align-items:center;justify-content:center;
            background:linear-gradient(145deg,#17395b,#0b2138);
            border:1px solid rgba(115,169,220,.28);
            color:#fff;font-weight:950;font-size:13px;
            box-shadow:inset 0 0 10px rgba(255,255,255,.02)
        }
        .mini-vs{font-size:11px;color:#5d7891;font-weight:800}
        .fixture-name{font-size:13px;font-weight:850;color:#fff;line-height:1.25}
        .fixture-time{font-size:11px;color:#a4b7c9;margin-top:5px}
        .pill-pre{
            display:inline-block;margin-top:8px;padding:3px 9px;border-radius:999px;
            border:1px solid rgba(41,229,120,.55);background:rgba(41,229,120,.11);
            color:#8ef5b9;font-size:9px;font-weight:950
        }
        .pill-started{
            display:inline-block;margin-top:8px;padding:3px 9px;border-radius:999px;
            border:1px solid rgba(255,95,109,.46);background:rgba(255,95,109,.10);
            color:#ff9ca5;font-size:9px;font-weight:950
        }
        .fixture-interest{font-size:10px;color:var(--yellow);font-weight:900;margin-top:7px}

        /* selected match hero */
        .hero{
            border:1px solid var(--line);border-radius:16px;
            background:
                radial-gradient(circle at 50% -30%,rgba(45,125,255,.18),transparent 42%),
                linear-gradient(145deg,#0b1929,#081522);
            padding:18px 22px;
            margin-top:13px;
        }
        .hero-grid{
            display:grid;grid-template-columns:1.4fr 1fr 1.4fr .9fr;gap:16px;align-items:center
        }
        .hero-team{text-align:center}
        .hero-badge{
            width:61px;height:61px;border-radius:50%;margin:0 auto 7px;
            display:flex;align-items:center;justify-content:center;
            background:linear-gradient(145deg,#1a4269,#0b2239);
            border:1px solid rgba(128,182,230,.3);
            font-size:18px;font-weight:950;color:white
        }
        .hero-team-name{font-size:20px;font-weight:950;color:#fff}
        .hero-mid{text-align:center}
        .hero-date{font-size:11px;color:#8fa5bb}
        .hero-time{font-size:29px;font-weight:950;color:#fff;margin-top:3px}
        .hero-count{
            border:1px solid var(--line);border-radius:12px;background:#0b1a2b;
            padding:12px;text-align:center
        }
        .hero-count-label{font-size:10px;color:#91a6bb}
        .hero-count-value{font-size:20px;font-weight:950;color:#6be9f0;margin-top:4px}

        .market-nav{
            display:grid;grid-template-columns:1.55fr repeat(5,1fr);gap:8px;margin:10px 0 14px
        }
        .market-chip{
            border:1px solid var(--line);border-radius:10px;padding:12px 8px;
            background:#0b1929;text-align:center;font-size:12px;font-weight:800;color:#d9e7f5
        }
        .market-chip.primary{
            background:linear-gradient(135deg,#236bf3,#328cff);
            border-color:#5ba4ff;color:white
        }

        .section-band{
            display:flex;align-items:center;justify-content:space-between;
            border:1px solid var(--line);border-radius:12px;background:#0a1827;
            padding:12px 15px;margin:9px 0 10px
        }
        .section-band-title{font-size:17px;font-weight:950;color:#fff}
        .section-band-sub{font-size:11px;color:#94a8bd;margin-top:2px}

        .opp{
            border:1px solid var(--line);border-radius:13px;
            background:linear-gradient(150deg,#0c1b2c,#091725);
            padding:13px 14px;height:100%;
        }
        .opp.best{border:2px solid var(--green)}
        .opp-rank{
            display:inline-flex;width:23px;height:23px;border-radius:50%;
            align-items:center;justify-content:center;background:#21c96b;color:#06140b;
            font-size:11px;font-weight:950;margin-right:6px
        }
        .opp-title{font-size:13px;font-weight:900;color:#fff}
        .opp-ev{
            display:inline-block;float:right;border:1px solid rgba(41,229,120,.5);
            background:rgba(41,229,120,.10);color:#7ff2ad;padding:3px 7px;border-radius:7px;
            font-size:11px;font-weight:950
        }
        .opp-row{display:flex;justify-content:space-between;gap:8px;font-size:11px;color:#9fb3c6;margin-top:7px}
        .opp-row strong{color:#eff7ff}
        .tag{
            display:inline-block;border-radius:7px;padding:4px 8px;font-size:9px;
            font-weight:950;margin-top:9px
        }
        .tag-strong{color:#ff9a76;background:rgba(255,101,61,.12);border:1px solid rgba(255,121,83,.25)}
        .tag-approved{color:#83f1ab;background:rgba(41,229,120,.11);border:1px solid rgba(41,229,120,.25)}
        .tag-secondary{color:#ffd576;background:rgba(255,193,59,.10);border:1px solid rgba(255,193,59,.25)}
        .tag-unvalidated{color:#ffa5aa;background:rgba(255,95,109,.10);border:1px solid rgba(255,95,109,.23)}

        .detail-shell{
            border-top:1px solid var(--line);
            margin-top:15px;padding-top:14px
        }
        .detail-title{font-size:17px;font-weight:950;color:white;margin-bottom:10px}
        .detail-card{
            border:1px solid var(--line);border-radius:12px;background:#091725;
            padding:13px;height:100%;
        }
        .detail-label{font-size:10px;color:#8ea4b9;text-transform:uppercase;font-weight:900}
        .detail-big{font-size:24px;font-weight:950;color:#fff;margin-top:4px}
        .detail-row{display:flex;justify-content:space-between;font-size:11px;color:#9db1c5;margin-top:7px}
        .detail-row strong{color:#fff}

        /* native widgets */
        div[data-testid="stVerticalBlockBorderWrapper"]{
            border:1px solid var(--line)!important;border-radius:14px!important;
            background:linear-gradient(145deg,#0b1929,#081522)!important;
        }
        div[data-testid="stMetric"]{
            border:1px solid var(--line);border-radius:12px;background:#091725;padding:9px 11px
        }
        .stButton>button{
            border-radius:10px;min-height:39px;font-weight:850;
            border:1px solid var(--line2)
        }
        .stButton>button[kind="primary"]{
            background:linear-gradient(135deg,#236bf3,#328cff);
            border-color:#5da5ff;color:white
        }
        div[data-testid="stExpander"]{
            border:1px solid var(--line)!important;border-radius:12px!important;background:#091725
        }
        div[data-baseweb="select"]>div,
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input{
            background:#091725!important;border-color:var(--line)!important
        }

        @media(max-width:900px){
            .block-container{padding-top:4rem!important;padding-left:.75rem;padding-right:.75rem}
            .title{font-size:25px}
            .hero-grid{grid-template-columns:1fr 1fr;gap:12px}
            .market-nav{grid-template-columns:1fr 1fr}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# ENGINE / HELPERS
# =========================================================

@st.cache_resource
def engine(league):
    if league == "eliteserien":
        return EliteserienEngine()
    if PredictionEngineLaLiga is None:
        raise RuntimeError("No se pudo cargar el motor de LaLiga.")
    return PredictionEngineLaLiga()


@st.cache_data(ttl=300)
def predictions_for(league):
    return engine(league).predict_next_round()


def now():
    return pd.Timestamp.now(tz=TZ)


def local_time(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(TZ)


def pre(match):
    return now() < local_time(match["utc_time"])


def initials(team):
    ws = [x for x in str(team).replace("-"," ").split() if x]
    if len(ws) == 1:
        return ws[0][:2].upper()
    return (ws[0][0] + ws[-1][0]).upper()


def fair(p):
    return 1 / p if p and p > 0 else None


def min_odds(p, ev):
    return (1 + ev) / p if p and p > 0 else None


def ev(p, odd):
    return p * odd - 1 if odd and odd > 1 else None


def level_html(level):
    x = str(level).upper()
    if "FUERTE" in x:
        return '<span class="tag tag-strong">🔥 FUERTE</span>'
    if "APROBADO" in x:
        return '<span class="tag tag-approved">✅ APROBADO</span>'
    if "NO VALIDADO" in x:
        return '<span class="tag tag-unvalidated">⚠️ NO VALIDADO</span>'
    return '<span class="tag tag-secondary">⚠️ SECUNDARIO</span>'


def market_rows(match, league):
    out = []

    def add(cat, key, label, level, p, extra=None):
        if p is None:
            return
        out.append({
            "id": f"{match['match_id']}|{cat}|{key}",
            "cat": cat, "key": key, "label": label,
            "level": level, "p": float(p), "extra": extra or {}
        })

    goals = match.get("goals", {}) or {}
    goal_lines = [(1.5,"FUERTE"),(2.5,"APROBADO"),(3.5,"FUERTE")] if league=="laliga" else [(2.5,"FUERTE"),(3.5,"SECUNDARIO")]
    for line, lvl in goal_lines:
        k = f"over_{str(line).replace('.','_')}"
        if k in goals:
            add("goals", k, f"Más de {line} goles", lvl, goals[k])

    corners = match.get("corners", {}) or {}
    corner_lines = [(8.5,"FUERTE"),(10.5,"APROBADO")] if league=="laliga" else [(9.5,"APROBADO"),(10.5,"FUERTE"),(11.5,"FUERTE")]
    for line, lvl in corner_lines:
        k = f"over_{str(line).replace('.','_')}"
        if k in corners:
            add("corners", k, f"Más de {line} córners", lvl, corners[k])

    cards = match.get("cards", {}) or {}
    card_lines = [(2.5,"FUERTE"),(3.5,"FUERTE"),(4.5,"FUERTE"),(5.5,"FUERTE")] if league=="laliga" else [(3.5,"SECUNDARIO"),(4.5,"APROBADO"),(5.5,"FUERTE")]
    refinfo = cards.get("referee_info", {}) or {}
    for line, lvl in card_lines:
        k = f"over_{str(line).replace('.','_')}"
        if k in cards:
            add("cards", k, f"Más de {line} tarjetas", lvl, cards[k], {"referee_info": refinfo})

    if league=="laliga":
        sot = match.get("shots_on_target", {}) or {}
        total_lines = [(7.5,"APROBADO"),(8.5,"FUERTE"),(9.5,"APROBADO"),(10.5,"FUERTE"),(11.5,"SECUNDARIO")]
        for line,lvl in total_lines:
            k = f"total_over_{str(line).replace('.','_')}"
            if k in sot:
                add("sot",k,f"Más de {line} SOT",lvl,sot[k])

        for side, tkey in [("home","home_team"),("away","away_team")]:
            for line in (2.5,3.5,4.5,5.5,6.5):
                k=f"{side}_over_{str(line).replace('.','_')}"
                if k in sot:
                    lvl="SECUNDARIO" if line==6.5 else "APROBADO"
                    add("sot",k,f"{match[tkey]} +{line} SOT",lvl,sot[k])

    if all(k in goals for k in ("home_win_raw","draw_raw","away_win_raw")):
        add("1x2","home",f"{match['home_team']} gana","NO VALIDADO",goals["home_win_raw"])
        add("1x2","draw","Empate","NO VALIDADO",goals["draw_raw"])
        add("1x2","away",f"{match['away_team']} gana","NO VALIDADO",goals["away_win_raw"])

    return out


def odds_data():
    return st.session_state.setdefault("v151_odds", {})


def books():
    return st.session_state.setdefault("v151_books", list(DEFAULT_BOOKMAKERS))



@st.cache_data(ttl=300, show_spinner=False)
def _auto_odds_snapshot(league, match_id, home_team, away_team, utc_time, rows_signature):
    # rows_signature is only for cache invalidation; rows are rebuilt below.
    fake_match = {
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "utc_time": utc_time,
    }
    rows = [
        {
            "id": item[0],
            "cat": item[1],
            "key": item[2],
            "label": item[3],
        }
        for item in rows_signature
    ]
    return populate_match_odds(
        league,
        fake_match,
        rows,
        selected_bookmakers=tuple(books()[:3]),
    )


def sync_auto_odds(match, league):
    if not pre(match):
        return {
            "ok": False,
            "locked": True,
            "message": "Partido iniciado: cuotas live bloqueadas.",
        }

    rows = market_rows(match, league)
    sig = tuple(
        (r["id"], r["cat"], r["key"], r["label"])
        for r in rows
    )

    result = _auto_odds_snapshot(
        league,
        str(match["match_id"]),
        str(match["home_team"]),
        str(match["away_team"]),
        str(match["utc_time"]),
        sig,
    )

    if result.get("prices"):
        store = odds_data()
        for market_id, prices in result["prices"].items():
            target = store.setdefault(market_id, {})
            for bookmaker, odd in prices.items():
                target[bookmaker] = float(odd)

    st.session_state["v151_auto_odds_status"] = result
    return result


def best_odds(row):
    d = odds_data().get(row["id"], {})
    vals = [(b,float(o)) for b,o in d.items() if o and float(o)>1]
    return max(vals,key=lambda x:x[1]) if vals else (None,None)


def analyzed(row, min_ev):
    b,o = best_odds(row)
    return {
        **row,
        "fair": fair(row["p"]),
        "min": min_odds(row["p"],min_ev),
        "book": b,
        "odds": o,
        "ev": ev(row["p"],o) if o else None,
    }


def interest_count(match, league, min_ev):
    rows=[analyzed(r,min_ev) for r in market_rows(match,league) if r["cat"]!="1x2"]
    quoted=[r for r in rows if r["ev"] is not None]
    if quoted:
        return sum(r["ev"]>=min_ev for r in quoted)
    return sum(r["p"]>=.55 and "SECUNDARIO" not in r["level"] for r in rows)


def countdown(match):
    d=local_time(match["utc_time"])-now()
    if d.total_seconds()<=0: return "Iniciado"
    mins=int(d.total_seconds()//60)
    days, mins=divmod(mins,1440)
    hrs, mins=divmod(mins,60)
    if days:return f"{days}d {hrs}h {mins}m"
    if hrs:return f"{hrs}h {mins}m"
    return f"{mins}m"


def get_match(predictions, mid):
    for m in predictions:
        if str(m["match_id"])==str(mid):
            return m
    return predictions[0]


def data_time(league):
    db=BASE_DIR/"data"/("laliga.db" if league=="laliga" else "eliteserien.db")
    if not db.exists(): return "sin dato"
    try:
        return pd.Timestamp(db.stat().st_mtime,unit="s",tz="UTC").tz_convert(TZ).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return "sin dato"


# =========================================================
# SIDEBAR
# =========================================================

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="brand">
          <div class="brand-flex">
            <div class="brand-mark"><i></i><i></i><i></i></div>
            <div>
              <div class="brand-title">Football Edge Pro</div>
              <div class="brand-sub">Datos. Modelos. Valor.</div>
            </div>
          </div>
        </div>
        """,unsafe_allow_html=True)

        pages=["🏠 Jornada","🔥 Mejores oportunidades","🕘 Histórico","⚙️ Ajustes"]
        pg=st.radio("Menú",pages,label_visibility="collapsed",key="v151_nav")

        st.markdown("#### LIGA ACTUAL")
        league_label=st.selectbox("Liga",list(LEAGUES.keys()),label_visibility="collapsed",key="v151_league_label")
        league=LEAGUES[league_label]

        st.markdown("---")
        mev=st.slider("EV mínimo",0,20,5,1,key="v151_ev")/100

        st.markdown("<br><div style='font-size:10px;color:#74899e'>Football Edge Pro v15.1<br>Analistas, no apostadores.</div>",unsafe_allow_html=True)
    return pg,league,mev


# =========================================================
# HOME: MATCH STRIP + SELECTED MATCH
# =========================================================

def top_header(predictions,league):
    rnd=predictions[0].get("round","")
    times=[local_time(m["utc_time"]) for m in predictions]
    c1,c2=st.columns([4.5,1.5])
    with c1:
        st.markdown(f'<div class="title">{"LaLiga" if league=="laliga" else "Eliteserien"} – Jornada {rnd}</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="subtitle">📅 {min(times).strftime("%d")} – {max(times).strftime("%d %B %Y")} · {len(predictions)} partidos</div>',unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="statusbox"><span class="green-dot"></span><b>Solo cuotas prepartido</b><br>'
            f'<span style="color:#7e94a9">Última actualización: {data_time(league)}</span></div>',
            unsafe_allow_html=True
        )


def fixture_strip(predictions,league,min_ev):
    selected=st.session_state.get("v151_selected")
    if selected is None or all(str(m["match_id"])!=str(selected) for m in predictions):
        selected=str(predictions[0]["match_id"])
        st.session_state["v151_selected"]=selected

    # 5 cards per row, just like the approved mockup.
    for start in range(0,len(predictions),5):
        row=predictions[start:start+5]
        cols=st.columns(5,gap="small")
        for col,m in zip(cols,row):
            with col:
                sel=str(m["match_id"])==str(selected)
                kick=local_time(m["utc_time"])
                status='<span class="pill-pre">PREPARTIDO</span>' if pre(m) else '<span class="pill-started">INICIADO</span>'
                n=interest_count(m,league,min_ev)
                st.markdown(f"""
                <div class="fixture-card {'selected' if sel else ''}">
                  <div class="mini-teams">
                    <div class="mini-badge">{initials(m['home_team'])}</div>
                    <div class="mini-vs">+</div>
                    <div class="mini-badge">{initials(m['away_team'])}</div>
                  </div>
                  <div class="fixture-name">{m['home_team']} vs {m['away_team']}</div>
                  <div class="fixture-time">{kick.strftime('%a, %d %b · %H:%M')}</div>
                  {status}
                  <div class="fixture-interest">{n} mercados interesantes ›</div>
                </div>
                """,unsafe_allow_html=True)
                if st.button("Ver partido",key=f"pick_{m['match_id']}",use_container_width=True):
                    st.session_state["v151_selected"]=str(m["match_id"])
                    st.session_state["v151_market"]=None
                    st.rerun()

    return get_match(predictions,st.session_state["v151_selected"])


def hero(match):
    kick=local_time(match["utc_time"])
    st.markdown(f"""
    <div class="hero">
      <div class="hero-grid">
        <div class="hero-team">
          <div class="hero-badge">{initials(match['home_team'])}</div>
          <div class="hero-team-name">{match['home_team']}</div>
        </div>
        <div class="hero-mid">
          <div class="hero-date">{kick.strftime('%A, %d %B %Y')}</div>
          <div class="hero-time">{kick.strftime('%H:%M')}</div>
          <div style="margin-top:5px">{'<span class="pill-pre">PREPARTIDO</span>' if pre(match) else '<span class="pill-started">INICIADO</span>'}</div>
        </div>
        <div class="hero-team">
          <div class="hero-badge">{initials(match['away_team'])}</div>
          <div class="hero-team-name">{match['away_team']}</div>
        </div>
        <div class="hero-count">
          <div class="hero-count-label">Tiempo para el inicio</div>
          <div class="hero-count-value">{countdown(match)}</div>
        </div>
      </div>
    </div>
    """,unsafe_allow_html=True)


def quick_market_buttons(match,league):
    st.markdown('<div class="market-nav">',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    labels=[("all","🔎 Buscar errores de cuota"),("goals","⚽ Goles"),("cards","🟨 Tarjetas"),("corners","🚩 Córners")]
    if league=="laliga":
        labels.append(("sot","🎯 Remates a puerta"))
    labels.append(("1x2","🏆 1X2"))

    cols=st.columns(len(labels),gap="small")
    for col,(key,label) in zip(cols,labels):
        with col:
            if st.button(label,key=f"nav_{match['match_id']}_{key}",use_container_width=True,type="primary" if key=="all" else "secondary",disabled=(key=="all" and not pre(match))):
                st.session_state["v151_market"]=key
                st.rerun()


def top_opportunities(match,league,min_ev):
    sync_auto_odds(match, league)
    rows=[analyzed(r,min_ev) for r in market_rows(match,league) if r["cat"]!="1x2"]
    quoted=[r for r in rows if r["ev"] is not None]
    if quoted:
        ranked=sorted(quoted,key=lambda r:r["ev"],reverse=True)[:4]
        title=f"{sum(r['ev']>=min_ev for r in quoted)} oportunidades con EV ≥ {min_ev*100:.0f}%"
        sub="Ordenadas por valor esperado (EV)"
    else:
        ranked=sorted(rows,key=lambda r:(("FUERTE" in r["level"]),r["p"]),reverse=True)[:4]
        title=f"{len(ranked)} mercados interesantes detectados"
        sub="Sin cuotas aún: priorizados por modelo para revisión"

    st.markdown(f"""
    <div class="section-band">
      <div>
        <div class="section-band-title">🔥 {title}</div>
        <div class="section-band-sub">{sub}</div>
      </div>
      <div style="font-size:11px;color:#8fa5bb">Casas: {' · '.join(books()[:3])}</div>
    </div>
    """,unsafe_allow_html=True)

    cols=st.columns(4,gap="small")
    for i,(col,r) in enumerate(zip(cols,ranked),1):
        with col:
            evtxt="—" if r["ev"] is None else f"{r['ev']*100:+.1f}%"
            best="Esperando cuota" if r["odds"] is None else f"{r['odds']:.2f} ({r['book']})"
            st.markdown(f"""
            <div class="opp {'best' if i==1 else ''}">
              <div><span class="opp-rank">{i}</span><span class="opp-title">{r['label']}</span>
                   <span class="opp-ev">{'EV '+evtxt if r['ev'] is not None else f"{r['p']*100:.0f}%"} </span></div>
              <div class="opp-row"><span>Probabilidad modelo</span><strong>{r['p']*100:.1f}%</strong></div>
              <div class="opp-row"><span>Cuota justa</span><strong>{r['fair']:.2f}</strong></div>
              <div class="opp-row"><span>Mejor cuota</span><strong>{best}</strong></div>
              <div class="opp-row"><span>Valor esperado</span><strong>{evtxt}</strong></div>
              {level_html(r['level'])}
            </div>
            """,unsafe_allow_html=True)


def detailed_home(match,league,min_ev):
    rows=[analyzed(r,min_ev) for r in market_rows(match,league) if r["cat"]!="1x2"]
    rows=sorted(rows,key=lambda r:(r["ev"] is not None, r["ev"] if r["ev"] is not None else r["p"]),reverse=True)
    if not rows:return
    r=rows[0]
    st.markdown('<div class="detail-shell">',unsafe_allow_html=True)
    st.markdown(f'<div class="detail-title">📊 Análisis detallado — {r["label"]}</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns([1,1.4,.9],gap="small")
    with c1:
        st.markdown(f"""
        <div class="detail-card">
          <div class="detail-label">Resumen</div>
          <div class="detail-row"><span>Probabilidad modelo</span><strong>{r['p']*100:.1f}%</strong></div>
          <div class="detail-row"><span>Cuota justa</span><strong>{r['fair']:.2f}</strong></div>
          <div class="detail-row"><span>Cuota mínima</span><strong>{r['min']:.2f}</strong></div>
          <div class="detail-row"><span>Mejor cuota</span><strong>{'—' if r['odds'] is None else f"{r['odds']:.2f}"}</strong></div>
          <div style="margin-top:7px">{level_html(r['level'])}</div>
        </div>
        """,unsafe_allow_html=True)
    with c2:
        if r["cat"]=="cards":
            ref=r.get("extra",{}).get("referee_info",{}) or {}
            name=ref.get("name") or match.get("referee") or "Pendiente"
            prof=ref.get("profile","PENDIENTE")
            av=ref.get("avg_yellow")
            avtxt="sin dato" if av is None or pd.isna(av) else f"{float(av):.2f}"
            body=f"<b>Árbitro:</b> {name}<br><b>Perfil:</b> {prof}<br><b>Media histórica:</b> {avtxt} tarjetas"
        else:
            body="Contexto prepartido del modelo disponible dentro de cada mercado. La V15 prioriza el precio, pero mantiene el análisis técnico."
        st.markdown(f"""
        <div class="detail-card">
          <div class="detail-label">Contexto del partido</div>
          <div style="font-size:12px;color:#c6d5e4;line-height:1.8;margin-top:6px">{body}</div>
        </div>
        """,unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="detail-card">
          <div class="detail-label">Comparativa de cuotas</div>
          <div style="font-size:11px;color:#9db1c5;margin-top:7px">Cuota automática / respaldo:</div>
          <div class="detail-big">{'—' if r['odds'] is None else f"{r['odds']:.2f}"}</div>
          <div style="font-size:11px;color:#9db1c5">{r['book'] or 'sin cuota cargada'}</div>
        </div>
        """,unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)


# =========================================================
# MARKET PAGE
# =========================================================

def odds_inputs(row,locked):
    d=odds_data().setdefault(row["id"],{})
    cols=st.columns(max(1,len(books()[:3])),gap="small")
    for col,name in zip(cols,books()[:3]):
        with col:
            val=st.number_input(
                name,0.0,100.0,float(d.get(name,0.0) or 0.0),.01,
                format="%.2f",key=f"odd_{row['id']}_{name}",disabled=locked
            )
            if val>1:d[name]=float(val)
            else:d.pop(name,None)


def market_page(match,league,cat,min_ev):
    if st.button("← Volver al partido",key="v151_back"):
        st.session_state["v151_market"]=None
        st.rerun()

    names={"goals":"⚽ Goles","cards":"🟨 Tarjetas","corners":"🚩 Córners","sot":"🎯 Remates a puerta","1x2":"🏆 1X2"}
    st.markdown(f'<div class="title">{names.get(cat,cat)}</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{match["home_team"]} vs {match["away_team"]} · análisis prepartido</div>',unsafe_allow_html=True)

    if cat=="1x2":
        st.warning("1X2 es contexto analítico. No lo tratamos como mercado aprobado para apostar.")
    if cat=="sot":
        st.info("🎯 SOT: mantenemos el modelo, pero no mezclamos SOT total/equipo con props individuales. Aquí queda respaldo manual.")

    if cat=="cards":
        cards=match.get("cards",{}) or {}
        ref=cards.get("referee_info",{}) or {}
        nm=ref.get("name") or match.get("referee") or "Pendiente"
        prof=ref.get("profile","PENDIENTE")
        av=ref.get("avg_yellow")
        st.markdown(f"""
        <div class="detail-card" style="margin-bottom:10px">
          <div class="detail-label">Árbitro</div>
          <div style="font-weight:900;font-size:15px;margin-top:5px">{nm}</div>
          <div style="font-size:11px;color:#9fb3c6;margin-top:4px">{prof} · media {'—' if av is None or pd.isna(av) else f"{float(av):.2f}"} tarjetas</div>
        </div>
        """,unsafe_allow_html=True)
        st.caption("🟨 Cuotas: Bet365 / Winamax / Pinnacle vía OddsPapi. El mercado automático es Bookings - Over Under Full Time; revisa las reglas de liquidación de cada casa.")

    auto_status = sync_auto_odds(match, league)
    if get_api_key():
        if auto_status.get("ok"):
            rem = auto_status.get("remaining")
            st.caption("📡 Cuotas PREPARTIDO · OddsPapi · " + " / ".join(books()[:3]))
        elif not auto_status.get("locked"):
            st.caption("📡 " + str(auto_status.get("message", "Sin cuotas automáticas para este mercado.")))
    else:
        st.warning("Falta ODDSPAPI_API_KEY. La app no puede descargar cuotas automáticas todavía.")
    rows=[r for r in market_rows(match,league) if r["cat"]==cat]
    for r in rows:
        a=analyzed(r,min_ev)
        with st.container(border=True):
            c1,c2,c3,c4,c5=st.columns([2.2,1,1,1,1],gap="small")
            c1.markdown(f"**{r['label']}**<br>{level_html(r['level'])}",unsafe_allow_html=True)
            c2.metric("Modelo",f"{r['p']*100:.1f}%")
            c3.metric("Justa",f"{a['fair']:.2f}")
            c4.metric("Mín. EV",f"{a['min']:.2f}")
            c5.metric("EV","—" if a["ev"] is None else f"{a['ev']*100:+.1f}%")
            if pre(match):
                prices = odds_data().get(r["id"], {})
                automatic = [(b, prices.get(b)) for b in books()[:3] if prices.get(b)]
                if automatic:
                    cols_auto = st.columns(len(books()[:3]), gap="small")
                    for col_auto, book_name in zip(cols_auto, books()[:3]):
                        with col_auto:
                            price = prices.get(book_name)
                            if price:
                                st.metric(book_name, f"{float(price):.2f}")
                            else:
                                st.metric(book_name, "—")
                else:
                    st.caption("Sin cuota automática para esta línea en nuestras 3 casas.")
                with st.expander("✏️ Cuota manual de respaldo"):
                    odds_inputs(r,False)
            else:
                st.caption("🔒 Partido iniciado: no se aceptan nuevas cuotas.")

            with st.expander("Ver modelo y detalles"):
                if cat=="goals":
                    g=match.get("goals",{}) or {}
                    st.write(f"λ local: **{float(g.get('lambda_home',0)):.2f}** · λ visitante: **{float(g.get('lambda_away',0)):.2f}**")
                elif cat=="corners":
                    c=match.get("corners",{}) or {}
                    st.write(f"λ córners local: **{float(c.get('lambda_home',0)):.2f}** · visitante: **{float(c.get('lambda_away',0)):.2f}**")
                elif cat=="sot":
                    s=match.get("shots_on_target",{}) or {}
                    st.write(f"λ SOT local: **{float(s.get('lambda_home',0)):.2f}** · visitante: **{float(s.get('lambda_away',0)):.2f}**")
                elif cat=="cards":
                    st.write("Modelo de tarjetas con comportamiento arbitral prepartido cuando hay árbitro asignado.")
                st.write(f"Cuota justa: **{a['fair']:.2f}** · Cuota mínima con EV objetivo: **{a['min']:.2f}**")


def scan_page(match,league,min_ev):
    if st.button("← Volver al partido",key="v151_back_scan"):
        st.session_state["v151_market"]=None
        st.rerun()

    st.markdown('<div class="title">🔍 Buscar errores de cuota</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{match["home_team"]} vs {match["away_team"]} · escaneo de todos los mercados</div>',unsafe_allow_html=True)

    if not pre(match):
        st.error("🔴 Partido iniciado. El escaneo de precio queda cerrado.")
        return

    auto_status = sync_auto_odds(match, league)
    if get_api_key():
        if auto_status.get("ok"):
            st.caption("📡 Cuotas automáticas PREPARTIDO · OddsPapi · " + " / ".join(books()[:3]))
        elif not auto_status.get("locked"):
            st.caption("📡 " + str(auto_status.get("message", "Sin cuotas automáticas para este partido.")))
    rows=[analyzed(r,min_ev) for r in market_rows(match,league) if r["cat"]!="1x2"]
    quoted=[r for r in rows if r["ev"] is not None]
    if not quoted:
        st.info("OddsPapi no ha devuelto cuota para esas líneas en Bet365 / Winamax / Pinnacle todavía. En tarjetas y córners puede ser simplemente que la casa aún no haya abierto el mercado; SOT total/equipo sigue manual.")
        rows=sorted(rows,key=lambda r:(("FUERTE" in r["level"]),r["p"]),reverse=True)
    else:
        rows=sorted(quoted,key=lambda r:r["ev"],reverse=True)

    for i,r in enumerate(rows[:12],1):
        val="—" if r["ev"] is None else f"{r['ev']*100:+.1f}%"
        st.markdown(f"""
        <div class="opp {'best' if i==1 else ''}" style="margin-bottom:8px">
          <div><span class="opp-rank">{i}</span><span class="opp-title">{r['label']}</span>
          <span class="opp-ev">{'EV '+val if r['ev'] is not None else f"{r['p']*100:.0f}%"} </span></div>
          <div class="opp-row"><span>Modelo</span><strong>{r['p']*100:.1f}%</strong></div>
          <div class="opp-row"><span>Cuota justa</span><strong>{r['fair']:.2f}</strong></div>
          <div class="opp-row"><span>Mejor cuota</span><strong>{'—' if r['odds'] is None else f"{r['odds']:.2f} ({r['book']})"}</strong></div>
          {level_html(r['level'])}
        </div>
        """,unsafe_allow_html=True)


# =========================================================
# OTHER PAGES
# =========================================================

def global_opps(predictions,league,min_ev):
    st.markdown('<div class="title">🔥 Mejores oportunidades</div>',unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Todos los partidos · solo prepartido · ordenados por EV</div>',unsafe_allow_html=True)
    rows=[]
    for m in predictions:
        if not pre(m):continue
        for r in market_rows(m,league):
            if r["cat"]=="1x2":continue
            a=analyzed(r,min_ev)
            if a["ev"] is not None:
                a["match"]=f"{m['home_team']} vs {m['away_team']}"
                rows.append(a)
    rows=sorted([r for r in rows if r["ev"]>=min_ev],key=lambda r:r["ev"],reverse=True)
    if not rows:
        st.info("No hay oportunidades con cuota cargada que superen el EV mínimo.")
        return
    for i,r in enumerate(rows,1):
        st.markdown(f"""
        <div class="opp" style="margin-bottom:8px">
          <div><span class="opp-rank">{i}</span><span class="opp-title">{r['label']} · {r['match']}</span>
          <span class="opp-ev">EV {r['ev']*100:+.1f}%</span></div>
          <div class="opp-row"><span>Modelo</span><strong>{r['p']*100:.1f}%</strong></div>
          <div class="opp-row"><span>Cuota</span><strong>{r['odds']:.2f} ({r['book']})</strong></div>
        </div>
        """,unsafe_allow_html=True)


def settings(league):
    st.markdown('<div class="title">⚙️ Ajustes</div>',unsafe_allow_html=True)
    if get_api_key():
        st.success("📡 OddsPapi conectada · Bet365 / Winamax / Pinnacle · PREPARTIDO")
    else:
        st.error("📡 OddsPapi sin configurar · añade ODDSPAPI_API_KEY")

    st.markdown('<div class="subtitle">Configuración de trabajo del analista</div>',unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### Casas de apuestas")
        current=books()
        vals=[]
        for i in range(3):
            vals.append(st.text_input(f"Casa {i+1}",current[i] if i<len(current) else "",key=f"book_{i}").strip())
        if st.button("Guardar casas",type="primary"):
            st.session_state["v151_books"]=[x for x in vals if x][:3]
            st.success("Casas guardadas.")
        st.caption("Máximo 3 casas. Solo cuotas prepartido.")

    if league=="laliga":
        with st.container(border=True):
            st.markdown("### Actualización")
            st.caption("Usa preferentemente el BAT de actualización local + GitHub que ya tienes.")


def history():
    st.markdown('<div class="title">🕘 Histórico</div>',unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Reservado para Pick Tracker, ROI, yield y rendimiento por mercado.</div>',unsafe_allow_html=True)
    st.info("No inventamos resultados. Esta pantalla se activará cuando guardemos picks reales prepartido.")


# =========================================================
# APP
# =========================================================

css()
page,league,min_ev=sidebar()

try:
    predictions=predictions_for(league)
except Exception as e:
    st.error("No se pudo cargar el motor.")
    st.exception(e)
    st.stop()

if not predictions:
    st.warning("No hay próxima jornada disponible.")
    st.stop()

if page=="🏠 Jornada":
    top_header(predictions,league)
    match=fixture_strip(predictions,league,min_ev)
    hero(match)
    quick_market_buttons(match,league)

    chosen=st.session_state.get("v151_market")
    if chosen=="all":
        scan_page(match,league,min_ev)
    elif chosen:
        market_page(match,league,chosen,min_ev)
    else:
        top_opportunities(match,league,min_ev)
        detailed_home(match,league,min_ev)

elif page=="🔥 Mejores oportunidades":
    global_opps(predictions,league,min_ev)

elif page=="🕘 Histórico":
    history()

elif page=="⚙️ Ajustes":
    settings(league)
