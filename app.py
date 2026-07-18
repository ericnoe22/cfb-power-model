"""
CFB Power Rankings & Betting Edge Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.append(os.path.dirname(__file__))
from config import CURRENT_SEASON, EDGE_THRESHOLD_SPREAD, EDGE_THRESHOLD_TOTAL, RATING_WEIGHTS
from model.power_rankings import build_composite_ratings, load_prebuilt_ratings
from model.game_predictor import predict_all_games
from model.edge_finder import find_edges, summarize_edges, track_ats_performance
from data.owls_fetcher import fetch_ncaaf_lines, fetch_ncaaf_lines_consensus
from data.cfbd_fetcher import fetch_lines
from model.situational_factors import (
    load_historical_data, build_matchup_ou_tendencies,
    build_team_ou_tendencies, build_team_ats_tendencies,
    apply_adjustments_to_predictions, get_ats_situational_note,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CFB Power Model",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject fonts ────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ── JS icon fix via components.html (actually executes — markdown strips scripts) ──
import streamlit.components.v1 as _components
_components.html("""
<script>
const ICON_MAP = {
    'keyboard_double_arrow_right': '›',
    'keyboard_double_arrow_left':  '‹',
    'chevron_right': '›',
    'chevron_left':  '‹',
};

function fixIcons() {
    try {
        var doc = window.parent.document;
        doc.querySelectorAll('span').forEach(function(span) {
            var txt = span.textContent.trim();
            if (ICON_MAP[txt]) {
                span.textContent = ICON_MAP[txt];
                span.style.fontFamily = 'sans-serif';
                span.style.fontSize = txt.startsWith('keyboard') ? '20px' : '16px';
                span.style.fontWeight = '700';
                span.style.color = txt.startsWith('keyboard') ? '#00b074' : '#7a95b5';
                span.style.lineHeight = '1';
            }
        });
    } catch(e) {}
}

fixIcons();
var observer = new MutationObserver(fixIcons);
observer.observe(window.parent.document.body, { childList: true, subtree: true });
</script>
""", height=0)

# ── Global styles ───────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Sidebar toggle button — hide raw icon text, show CSS arrow ── */
[data-testid="collapsedControl"] {
    background: #162030 !important;
    border: 1px solid #1e3050 !important;
    border-radius: 0 8px 8px 0 !important;
    color: #00b074 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 24px !important;
    min-height: 48px !important;
}
[data-testid="collapsedControl"]:hover {
    background: #1e3050 !important;
    border-color: #00b074 !important;
}
/* Hide the raw material icon text */
[data-testid="collapsedControl"] span {
    font-family: 'Material Symbols Rounded', sans-serif !important;
    font-size: 20px !important;
    color: #00b074 !important;
    /* Fallback: if font doesn't load, hide text and show CSS arrow */
    overflow: hidden !important;
    white-space: nowrap !important;
}
/* CSS arrow fallback when font fails */
[data-testid="collapsedControl"] span::before {
    content: "›" !important;
    font-family: sans-serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #00b074 !important;
    display: block !important;
}
/* Hide the actual icon text (shows as fallback char above instead) */
[data-testid="collapsedControl"] span {
    font-size: 0 !important;
    width: 22px !important;
    height: 22px !important;
}


/* ── Expander icon — show clean + / − instead of raw material icon text ── */
[data-testid="stExpander"] summary span {
    font-size: 0 !important;
    width: 16px !important;
    display: inline-block !important;
}
[data-testid="stExpander"] summary span::before {
    content: "＋" !important;
    font-size: 14px !important;
    font-family: sans-serif !important;
    font-weight: 700 !important;
    color: #00b074 !important;
}
[data-testid="stExpander"][open] summary span::before,
details[open] summary span::before {
    content: "−" !important;
    color: #00b074 !important;
}

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Background ── */
[data-testid="stAppViewContainer"] { background-color: #0f1923 !important; }
[data-testid="stHeader"] { background-color: #0f1923 !important; border-bottom: 1px solid #1e3050; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0a1628 !important;
    border-right: 1px solid #1e3050 !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    display: block;
    height: 3px;
    background: linear-gradient(90deg, #00b074, #f5a623);
}
[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.3px;
}
[data-testid="stSidebar"] .stCaption p { color: #4d6a8a !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #6b8aad !important; font-size: 0.7rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 1px; }

/* Nav radio — hide circles, style as buttons */
[data-testid="stRadio"] > label { display: none !important; }
[data-testid="stRadio"] div[role="radiogroup"] { gap: 2px !important; }
[data-testid="stRadio"] div[role="radiogroup"] label {
    display: flex !important;
    align-items: center;
    width: 100%;
    padding: 0.6rem 0.9rem !important;
    border-radius: 8px !important;
    color: #7a95b5 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: all 0.15s;
    cursor: pointer;
    border-left: 3px solid transparent !important;
    background: transparent !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(0,176,116,0.08) !important;
    color: #ffffff !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(0,176,116,0.12) !important;
    border-left: 3px solid #00b074 !important;
    color: #00b074 !important;
    font-weight: 600 !important;
}
[data-testid="stRadio"] div[role="radiogroup"] input[type="radio"] {
    display: none !important;
}

/* ── Typography ── */
h1 { color: #ffffff !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2 { color: #e2e8f0 !important; font-weight: 700 !important; }
h3 { color: #cdd9e8 !important; font-weight: 600 !important; }
p, [data-testid="stMarkdownContainer"] p { color: #8ba3be !important; }
hr { border-color: #1e3050 !important; margin: 0.8rem 0 !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #162030 !important;
    border: 1px solid #1e3050 !important;
    border-radius: 10px !important;
    padding: 0.9rem 1rem !important;
}
[data-testid="metric-container"] label {
    color: #4d6a8a !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700 !important; font-size: 1rem !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Buttons ── */
.stButton > button {
    background: #162030 !important;
    border: 1px solid #243a5e !important;
    color: #cdd9e8 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #1e3050 !important;
    border-color: #00b074 !important;
    color: #ffffff !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #00b074, #008f5c) !important;
    border: none !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ── Inputs / Selects ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #162030 !important;
    border: 1px solid #243a5e !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Pills ── */
[data-testid="stPills"] button {
    background: #162030 !important;
    border: 1px solid #243a5e !important;
    color: #7a95b5 !important;
    border-radius: 20px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
}
[data-testid="stPills"] button[aria-pressed="true"] {
    background: #00b074 !important;
    border-color: #00b074 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e3050 !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: #162030 !important;
    color: #4d6a8a !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stDataFrame"] td { color: #cdd9e8 !important; }

/* ── Alerts ── */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #162030 !important;
    border: 1px solid #1e3050 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #7a95b5 !important; font-weight: 500 !important; }

/* ── Caption ── */
.stCaption p { color: #4d6a8a !important; font-size: 0.8rem !important; }

/* ── Dialog title ── */
[data-baseweb="dialog-header"],
[data-baseweb="dialog-header"] * {
    color: #ffffff !important;
    background-color: #162030 !important;
}

/* ── Dialog modal — dark background to match site theme ── */
/* Target every possible container Streamlit/BaseWeb might render */
[role="dialog"],
[aria-modal="true"],
div[data-baseweb="dialog"],
[data-testid="stModal"] > div,
[data-testid="stModal"] > div > div {
    background-color: #162030 !important;
    border: 1px solid #243a5e !important;
}
[data-testid="stModal"] [data-testid="stVerticalBlock"],
[data-testid="stModal"] [data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #162030 !important;
}
/* Keep text white/light — consistent with the rest of the site */
[role="dialog"] p,
[role="dialog"] span,
[role="dialog"] label,
[role="dialog"] h1,
[role="dialog"] h2,
[role="dialog"] h3,
[data-testid="stModal"] p,
[data-testid="stModal"] span,
[data-testid="stModal"] label,
[data-testid="stModal"] h1,
[data-testid="stModal"] h2,
[data-testid="stModal"] h3 {
    color: #e2e8f0 !important;
}
[role="dialog"] [data-testid="stMetricValue"],
[data-testid="stModal"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}
[role="dialog"] [data-testid="metric-container"] label,
[data-testid="stModal"] [data-testid="metric-container"] label {
    color: #6b8aad !important;
}
[role="dialog"] [data-testid="metric-container"],
[data-testid="stModal"] [data-testid="metric-container"] {
    background: #0f1923 !important;
    border: 1px solid #1e3050 !important;
}
[role="dialog"] hr,
[data-testid="stModal"] hr {
    border-color: #1e3050 !important;
}

/* ── Matchup card styles ── */
.matchup-card {
    background: #162030;
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.matchup-card:hover { border-color: #00b074; }
.mc-header {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.8rem;
}
.mc-badge {
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
.mc-badge-week { background: #0d2518; color: #00b074; }
.mc-badge-neutral { background: #2a1f08; color: #f5a623; }
.mc-badge-conf { background: #1a2742; color: #6b8aad; }
.mc-teams {
    display: grid;
    grid-template-columns: 1fr 36px 1fr;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.85rem;
}
.mc-team { display: flex; flex-direction: column; }
.mc-team.home { align-items: flex-end; text-align: right; }
.mc-team-name { font-size: 1.05rem; font-weight: 700; color: #ffffff; line-height: 1.2; }
.mc-team-name.fcs { color: #7a95b5; font-size: 0.9rem; }
.mc-team-rtg { font-size: 0.75rem; color: #4d6a8a; font-weight: 500; margin-top: 2px; }
.mc-vs {
    background: #1e3050;
    border-radius: 50%;
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #6b8aad;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
    align-self: center;
    justify-self: center;
}
.mc-lines {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    background: #0f1923;
    border-radius: 8px;
    padding: 0.65rem 0.75rem;
    margin-bottom: 0.6rem;
}
.mc-line { display: flex; flex-direction: column; gap: 2px; }
.mc-line-label { font-size: 0.6rem; font-weight: 700; color: #4d6a8a; text-transform: uppercase; letter-spacing: 0.5px; }
.mc-line-main { font-size: 0.92rem; font-weight: 700; color: #ffffff; }
.mc-line-sub { font-size: 0.74rem; color: #7a95b5; }
.mc-footer { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.mc-bet {
    display: inline-block;
    background: rgba(0,176,116,0.12);
    border: 1px solid rgba(0,176,116,0.35);
    color: #00b074;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
}
.mc-pts {
    font-size: 0.75rem;
    color: #4d6a8a;
    margin-left: auto;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    /* Reduce page padding so tables use full width */
    .main .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 1rem !important;
    }

    /* Tables: horizontally scrollable, don't shrink text to nothing */
    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th {
        font-size: 0.75rem !important;
        white-space: nowrap !important;
    }

    /* Stack metric cards vertically */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        min-width: 45% !important;
        flex: 1 1 45% !important;
    }

    /* Bigger tap targets for buttons */
    .stButton > button {
        min-height: 44px !important;
        font-size: 0.95rem !important;
        padding: 0.5rem 1rem !important;
    }

    /* Pills: wrap onto multiple lines */
    [data-testid="stPills"] {
        flex-wrap: wrap !important;
        gap: 4px !important;
    }
    [data-testid="stPills"] button {
        font-size: 0.78rem !important;
        padding: 4px 10px !important;
    }

    /* Matchup card lines — stack to 2 columns on small screens */
    .mc-lines {
        grid-template-columns: repeat(2, 1fr) !important;
    }

    /* Headings slightly smaller */
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }

    /* Hide sidebar by default hint — already handled by Streamlit */
    [data-testid="collapsedControl"] {
        top: 0.5rem !important;
    }
}

@media (max-width: 480px) {
    /* Very small phones — single column metrics */
    [data-testid="stHorizontalBlock"] > div {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Matchup card teams — slightly smaller */
    .mc-team-name { font-size: 0.9rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────

PREBUILT_RATINGS_PATH = os.path.join(os.path.dirname(__file__), f"{CURRENT_SEASON}_power_rating_cleaned.csv")
SCHEDULE_PATH         = os.path.join(os.path.dirname(__file__), f"{CURRENT_SEASON}_schedule_with_power.csv")

@st.cache_data(ttl=3600)
def load_ratings():
    """Load composite ratings — cached 1 hour. Ratings only change on manual CSV updates."""
    if os.path.exists(PREBUILT_RATINGS_PATH):
        df = load_prebuilt_ratings(PREBUILT_RATINGS_PATH)
        df = _compute_composite(df)

        # Merge SP+ offense/defense ratings for accurate total predictions.
        # predict_total uses these directly when available; otherwise falls back
        # to the composite-based estimate which over-predicts for elite matchups.
        sp_path = os.path.join(os.path.dirname(__file__), f"cache/sp_plus_{CURRENT_SEASON}.csv")
        if os.path.exists(sp_path) and os.path.getsize(sp_path) > 0:
            try:
                sp = pd.read_csv(sp_path)
            except Exception:
                sp = pd.DataFrame()
            if {"offense.rating", "defense.rating", "team"}.issubset(sp.columns):
                from data.team_names import normalize as _norm
                sp = sp[["team", "offense.rating", "defense.rating"]].copy()
                sp["team"] = sp["team"].map(_norm)
                df["team"] = df["team"].map(_norm)
                df = df.merge(sp, on="team", how="left")

        return df
    return pd.DataFrame()


def _compute_composite(df):
    """Add composite rating column to a ratings DataFrame."""
    from model.power_rankings import build_composite_ratings, z_score, _normalize_elo, \
        _normalize_returning, _normalize_talent

    # The prebuilt CSV already has sp_plus, fpi, elo, returning_prod, talent
    weights = RATING_WEIGHTS

    if "sp_plus" not in df.columns:
        return df

    df = df.copy()
    df["sp_plus_norm"]   = df["sp_plus"].fillna(df["sp_plus"].mean())
    df["fpi_norm"]       = df["fpi"].fillna(df["fpi"].mean()) if "fpi" in df.columns else df["sp_plus_norm"]
    df["elo_norm"]       = _normalize_elo(df["elo"].fillna(1500)) if "elo" in df.columns else 0
    df["returning_norm"] = _normalize_returning(df["returning_prod"].fillna(0.55)) if "returning_prod" in df.columns else 0
    df["talent_norm"]    = _normalize_talent(df["talent"].fillna(df["talent"].mean())) if "talent" in df.columns else 0

    df["composite"] = (
        weights["sp_plus"]        * df["sp_plus_norm"]   +
        weights["fpi"]            * df["fpi_norm"]        +
        weights["elo"]            * df["elo_norm"]        +
        weights["returning_prod"] * df["returning_norm"]  +
        weights["talent"]         * df["talent_norm"]
    )

    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df = df.drop(columns=["rank"], errors="ignore")  # remove existing rank before re-adding
    df.index += 1
    df.index.name = "rank"
    df = df.reset_index()
    return df


def _normalize_elo(s):
    return (s - 1500) / 30

def _normalize_returning(s):
    return (s - s.mean()) * 20

def _normalize_talent(s):
    mean, std = s.mean(), s.std()
    return ((s - mean) / std) * 5 if std > 0 else s * 0


@st.cache_data(ttl=3600)
def load_schedule():
    if os.path.exists(SCHEDULE_PATH):
        df = pd.read_csv(SCHEDULE_PATH)
        df = df.drop_duplicates(subset=["homeTeam", "awayTeam", "week"])
        return df
    return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_multibook_lines():
    """Pull multi-book lines from Owls Insight. Returns (multibook_df, source)."""
    try:
        _, multibook_df = fetch_ncaaf_lines()
        if not multibook_df.empty:
            multibook_df.to_csv("cache/lines_multibook.csv", index=False)
            return multibook_df, "Owls Insight (live)"
    except Exception:
        pass
    cache_path = "cache/lines_multibook.csv"
    if os.path.exists(cache_path):
        import time
        age_mins = (time.time() - os.path.getmtime(cache_path)) / 60
        return pd.read_csv(cache_path), f"cached ({int(age_mins)}m ago)"
    return pd.DataFrame(), "unavailable"


@st.cache_data(ttl=1800)  # cache 30 mins — refresh picks up new lines
def load_live_lines():
    """Pull current NCAAF lines from Owls Insight, tagged with CFB week number."""
    try:
        df = fetch_ncaaf_lines_consensus()
        if not df.empty:
            df.to_csv("cache/lines_live.csv", index=False)
            source = "Owls Insight (live)"
        else:
            raise ValueError("empty")
    except Exception:
        cache_path = "cache/lines_live.csv"
        cfbd_path  = os.path.join(os.path.dirname(__file__), f"cache/lines_{CURRENT_SEASON}.csv")
        if os.path.exists(cache_path):
            import time
            age_mins = (time.time() - os.path.getmtime(cache_path)) / 60
            df = pd.read_csv(cache_path)
            source = f"cached ({int(age_mins)}m ago)"
        else:
            # Try The Odds API as secondary live source
            try:
                from data.odds_api_fetcher import fetch_ncaaf_game_lines
                odds_df = fetch_ncaaf_game_lines()
                if not odds_df.empty:
                    df = odds_df
                    source = f"The Odds API / DraftKings ({len(df)} games)"
                else:
                    raise ValueError("empty")
            except Exception:
                # Final fallback: CFBD lines cache
                if os.path.exists(cfbd_path):
                    raw = pd.read_csv(cfbd_path)
                    raw = raw[raw["spread"].notna()].copy()
                    dk = raw[raw["provider"].str.lower() == "draftkings"]
                    raw = dk if not dk.empty else raw
                    raw = raw.drop_duplicates(subset=["homeTeam", "awayTeam"]).reset_index(drop=True)
                    df = raw
                    source = f"CFBD / DraftKings ({len(df)} games)"
                else:
                    return pd.DataFrame(), "unavailable"

    # ── Tag each game with a CFB week number ──────────────────────────────
    # Match to the season schedule by team pair (normalize both sides).
    # Owls and CFBD sometimes disagree on home/away, so we try both orderings.
    sched_path = os.path.join(os.path.dirname(__file__), f"{CURRENT_SEASON}_schedule_with_power.csv")
    if os.path.exists(sched_path):
        from data.team_names import normalize
        sched = pd.read_csv(sched_path)[["homeTeam", "awayTeam", "week", "neutralSite"]].drop_duplicates()
        sched["homeTeam"] = sched["homeTeam"].map(normalize)
        sched["awayTeam"] = sched["awayTeam"].map(normalize)
        # Build a lookup keyed by frozenset of the two teams → week
        sched_week_map = {
            frozenset([h, a]): (w, n)
            for h, a, w, n in zip(sched["homeTeam"], sched["awayTeam"],
                                   sched["week"], sched["neutralSite"])
        }
        df["homeTeam"] = df["homeTeam"].map(normalize)
        df["awayTeam"] = df["awayTeam"].map(normalize)
        df["week"] = df.apply(
            lambda r: sched_week_map.get(frozenset([r["homeTeam"], r["awayTeam"]]), (None, False))[0],
            axis=1
        )
        df["neutralSite"] = df.apply(
            lambda r: sched_week_map.get(frozenset([r["homeTeam"], r["awayTeam"]]), (None, False))[1],
            axis=1
        )
    else:
        df["week"] = None
        df["neutralSite"] = False

    # Fallback: derive week from game date (week 1 = Aug 29, 2026)
    if "commence_dt" in df.columns and df["week"].isna().any():
        import datetime, pytz
        season_start = pd.Timestamp(f"{CURRENT_SEASON}-08-28", tz="UTC")
        df.loc[df["week"].isna(), "week"] = (
            (df.loc[df["week"].isna(), "commence_dt"] - season_start).dt.days // 7 + 1
        ).clip(lower=1)

    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(1).astype(int)
    return df, source


@st.cache_data(ttl=3600)
def load_historical_lines(year):
    """Load CFBD historical lines (with opening lines) for line movement analysis."""
    cache_path = f"cache/lines_{year}.csv"
    try:
        df = fetch_lines(year=year, force_refresh=False)
        if not df.empty:
            df.to_csv(cache_path, index=False)
            return df
    except Exception:
        pass
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    return pd.DataFrame()


def load_backtest_results(year):
    """Load saved backtest results CSV if available."""
    path = os.path.join(os.path.dirname(__file__), "outputs", f"backtest_{year}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_season_projections(schedule_df, ratings_df):
    """Run model against full schedule and return projected win totals."""
    from model.game_predictor import predict_all_games
    from model.season_projector import project_season_wins
    if schedule_df.empty or ratings_df.empty:
        return pd.DataFrame()
    predictions = predict_all_games(schedule_df, ratings_df)
    return project_season_wins(schedule_df, predictions)


@st.cache_data(ttl=3600)
def load_win_totals():
    """
    Load Vegas season win totals.
    Priority: manual CSV → Owls Insight API → empty.

    Manual import: drop cache/win_totals_{year}_manual.csv with columns:
        team, wins_line, over_odds, under_odds
    """
    manual_path = os.path.join(os.path.dirname(__file__), f"cache/win_totals_{CURRENT_SEASON}_manual.csv")
    if os.path.exists(manual_path):
        try:
            df = pd.read_csv(manual_path)
            if "team" in df.columns and "wins_line" in df.columns:
                return df
        except Exception:
            pass
    try:
        from data.owls_fetcher import fetch_ncaaf_win_totals
        return fetch_ncaaf_win_totals()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_championship_odds():
    """Fetch NCAAF national championship winner odds from The Odds API."""
    try:
        from data.odds_api_fetcher import fetch_ncaaf_championship_odds
        return fetch_ncaaf_championship_odds()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400)  # refresh once per day
def load_situational_tendencies():
    """Build matchup O/U, team O/U, and team ATS tendency tables from historical data."""
    hist = load_historical_data()
    if hist.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    matchup_ou = build_matchup_ou_tendencies(hist)
    team_ou    = build_team_ou_tendencies(hist)
    team_ats   = build_team_ats_tendencies(hist)
    return matchup_ou, team_ou, team_ats


@st.cache_data(ttl=86400)
def load_fpi_projections():
    """Load ESPN FPI preseason projections (proj wins, conf %, playoff %, NC %)."""
    path = os.path.join(os.path.dirname(__file__), f"cache/fpi_projections_{CURRENT_SEASON}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _roi(wins, losses):
    total = wins + losses
    if total == 0:
        return 0.0
    return round((wins / total - 110 / 210) * 100, 1)


def fmt_spread(df, spread_cols):
    """
    For each spread column in the DataFrame, format values so that when the
    away team is the favorite (spread > 0 from home perspective), the value
    is displayed in parentheses: e.g. +3.5 → (3.5), -7.0 stays -7.0.
    Returns a copy with those columns converted to strings.
    """
    df = df.copy()
    for col in spread_cols:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        df[col] = numeric.apply(
            lambda v: f"({abs(v):.1f})" if pd.notna(v) and v > 0
            else (f"{v:.1f}" if pd.notna(v) else "—")
        )
    return df


CFB_SCORE_STD = 14.0  # empirical std dev of CFB margins (~14 pts)


def _implied_prob_no_vig(home_ml, away_ml):
    """
    Convert American moneylines to no-vig win probabilities.
    Returns (home_prob, away_prob).
    """
    def raw(ml):
        ml = float(ml)
        return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)

    h, a = raw(home_ml), raw(away_ml)
    total = h + a
    return h / total, a / total


def _spread_to_win_prob(spread):
    """
    Home win probability from predicted spread (home perspective, negative = home favored).
    Uses a normal distribution with CFB_SCORE_STD.
    """
    from scipy.stats import norm
    return float(norm.cdf(-spread / CFB_SCORE_STD))


def compute_bet_recommendations(df, spread_thresh, total_thresh,
                                min_confidence=0.55, min_ml_edge=0.08):
    """
    Add a 'bet' column to a predictions DataFrame.
    Evaluates spread edges, total edges, and underdog ML value.
    Only flags Grade B or better bets.
    """
    bets = []

    for _, row in df.iterrows():
        pred_spread  = pd.to_numeric(row.get("predicted_spread"), errors="coerce")
        vegas_spread = pd.to_numeric(row.get("vegas_spread"),     errors="coerce")
        pred_total   = pd.to_numeric(row.get("predicted_total"),  errors="coerce")
        vegas_total  = pd.to_numeric(row.get("vegas_total"),      errors="coerce")
        confidence   = float(row.get("confidence", 0) or 0)
        home_ml      = row.get("home_ml")
        away_ml      = row.get("away_ml")

        if confidence < min_confidence:
            bets.append("—")
            continue

        parts = []   # (label, score)

        # ── Spread edge ───────────────────────────────────────────────
        if pd.notna(pred_spread) and pd.notna(vegas_spread):
            edge = pred_spread - vegas_spread
            if edge < -spread_thresh:
                parts.append(("Home ATS", abs(edge) * confidence))
            elif edge > spread_thresh:
                parts.append(("Away ATS", abs(edge) * confidence))

        # ── Total edge ────────────────────────────────────────────────
        if pd.notna(pred_total) and pd.notna(vegas_total):
            edge_t = pred_total - vegas_total
            if edge_t > total_thresh:
                parts.append(("Over", abs(edge_t) * confidence))
            elif edge_t < -total_thresh:
                parts.append(("Under", abs(edge_t) * confidence))

        # ── Underdog ML value ──────────────────────────────────────────
        if pd.notna(pred_spread):
            try:
                hml = float(home_ml) if pd.notna(home_ml) else None
                aml = float(away_ml) if pd.notna(away_ml) else None

                if hml is not None and aml is not None:
                    h_prob, a_prob = _implied_prob_no_vig(hml, aml)
                else:
                    h_prob = _spread_to_win_prob(pred_spread)
                    a_prob = 1 - h_prob
                    h_prob = (hml and (abs(hml) / (abs(hml) + 100) if hml < 0 else 100 / (hml + 100))) or h_prob
                    a_prob = (aml and (abs(aml) / (abs(aml) + 100) if aml < 0 else 100 / (aml + 100))) or a_prob

                model_h = _spread_to_win_prob(pred_spread)
                model_a = 1 - model_h

                # Home ML underdog
                if hml is not None and hml > 0 and (model_h - h_prob) >= min_ml_edge:
                    score = (model_h - h_prob) * 10 * confidence
                    parts.append((f"Home ML +{int(hml)}", score))

                # Away ML underdog
                if aml is not None and aml > 0 and (model_a - a_prob) >= min_ml_edge:
                    score = (model_a - a_prob) * 10 * confidence
                    parts.append((f"Away ML +{int(aml)}", score))

            except (TypeError, ValueError, ZeroDivisionError):
                pass

        if not parts:
            bets.append("—")
            continue

        # Grade based on best single signal score
        max_score = max(s for _, s in parts)
        if max_score >= 8:    grade = "A+"
        elif max_score >= 6:  grade = "A"
        elif max_score >= 4.5: grade = "B"
        else:
            bets.append("—")   # below B — skip for schedule view
            continue

        labels = " + ".join(p[0] for p in parts)
        bets.append(f"{labels}  ({grade})")

    df = df.copy()
    df["bet"] = bets
    return df


def grade_color(grade):
    colors = {"A+": "#00b300", "A": "#33cc33", "B": "#ffcc00", "C": "#ff9933"}
    return colors.get(grade, "#cccccc")


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏈 CFB Power Model")
    st.caption(f"Season: {CURRENT_SEASON}")

    # ── Admin unlock ──────────────────────────────────────────────────────
    import hashlib
    _ADMIN_HASH = "a42979990988b809ae42b234cfe7805bc57fd5c8eefc2d03fb0293ebc0c789cb"
    if "admin_unlocked" not in st.session_state:
        st.session_state["admin_unlocked"] = False

    _public_pages = [
        "📊 Power Rankings",
        "📅 Schedule & Predictions",
        "🏆 Season Projections",
        "🎰 Title Odds",
        "⚔️ Head-to-Head",
        "🎯 Betting Edges",
    ]
    _admin_pages = [
        "📈 Model Performance",
        "🔧 Update Data",
    ]

    if st.session_state["admin_unlocked"]:
        nav_pages = _public_pages + _admin_pages
    else:
        nav_pages = _public_pages

    page = st.radio("Navigate", nav_pages)

    # Lock icon at the bottom of sidebar
    st.sidebar.divider()
    if st.session_state["admin_unlocked"]:
        if st.sidebar.button("🔓 Lock Admin"):
            st.session_state["admin_unlocked"] = False
            st.rerun()
    else:
        with st.sidebar.expander("🔐 Admin"):
            _pw = st.text_input("Password", type="password", key="admin_pw")
            if st.button("Unlock", key="admin_unlock_btn"):
                if hashlib.sha256(_pw.encode()).hexdigest() == _ADMIN_HASH:
                    st.session_state["admin_unlocked"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password")

    ratings_df = load_ratings()
    schedule_df = load_schedule()


# Default thresholds — overridden inline on the Betting Edges page
spread_thresh = EDGE_THRESHOLD_SPREAD
total_thresh  = EDGE_THRESHOLD_TOTAL


# ── Board view renderer ────────────────────────────────────────────────────

def render_board_view(sched_df):
    """Render a compact sportsbook-style board sorted by kickoff time."""
    import pytz
    from datetime import datetime

    def _parse_time(raw):
        try:
            dt = pd.to_datetime(raw, utc=True).tz_convert("US/Eastern")
            tbd = False
        except Exception:
            return None, True
        return dt, False

    def _fmt_time(dt):
        if dt is None:
            return "TBD"
        day = dt.strftime("%a")
        t = dt.strftime("%-I:%M %p").replace(" AM", "am").replace(" PM", "pm")
        return f"{day} {t} ET"

    def _fmt_spread(v, home, away):
        try:
            v = float(v)
            if v == 0: return "PK"
            fav = home if v < 0 else away
            return f"{fav} -{abs(v):.1f}"
        except (TypeError, ValueError):
            return "—"

    def _fmt_total(v):
        try: return f"{float(v):.1f}"
        except (TypeError, ValueError): return "—"

    def _fmt_ml(v):
        try:
            v = float(v)
            return f"+{int(v)}" if v > 0 else str(int(v))
        except (TypeError, ValueError): return "—"

    # Sort by kickoff time
    df = sched_df.copy()
    if "startDate" in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["startDate"], utc=True, errors="coerce")
        df = df.sort_values("_sort_dt", na_position="last")

    rows_html = ""
    prev_time_str = None

    for _, row in df.iterrows():
        home = str(row.get("homeTeam", "")).replace(" (FCS)", "")
        away = str(row.get("awayTeam", "")).replace(" (FCS)", "")
        neutral = str(row.get("neutralSite", "")).lower() in ("true", "1", "yes")
        home_pts = row.get("homePoints")
        away_pts = row.get("awayPoints")
        completed = str(row.get("completed", "")).lower() in ("true", "1", "yes")

        dt, tbd = _parse_time(row.get("startDate"))
        time_str = _fmt_time(dt)

        # Time group divider
        if time_str != prev_time_str:
            rows_html += f'<div style="padding:6px 12px;background:#0d1626;color:#7a95b5;font-size:0.75rem;font-weight:700;letter-spacing:0.05em;border-top:1px solid #1a2744;margin-top:4px">{time_str}</div>'
            prev_time_str = time_str

        m_spread = _fmt_spread(row.get("predicted_spread"), home, away)
        v_spread = _fmt_spread(row.get("vegas_spread"),     home, away)
        m_total  = _fmt_total(row.get("predicted_total"))
        v_total  = _fmt_total(row.get("vegas_total"))
        home_ml  = _fmt_ml(row.get("home_ml"))
        away_ml  = _fmt_ml(row.get("away_ml"))
        bet      = str(row.get("bet", "") or "")

        # Edge badge
        edge_badge = ""
        if bet:
            edge_badge = f'<span style="background:#00b074;color:#000;font-size:0.65rem;font-weight:700;padding:1px 5px;border-radius:3px;margin-left:6px">EDGE</span>'

        # Score or spread display
        if completed and home_pts is not None and away_pts is not None:
            try:
                score_away = f'<span style="font-weight:700;color:#fff">{int(away_pts)}</span>'
                score_home = f'<span style="font-weight:700;color:#fff">{int(home_pts)}</span>'
            except Exception:
                score_away = score_home = ""
        else:
            score_away = score_home = ""

        neutral_tag = ' <span style="color:#7a95b5;font-size:0.7rem">N</span>' if neutral else ""

        # Two-line matchup row: away on top, home on bottom
        rows_html += f'''
<div style="display:grid;grid-template-columns:1fr auto auto auto;align-items:center;
            padding:8px 12px;border-bottom:1px solid #1a2744;gap:8px;">
  <div>
    <div style="color:#ccc;font-size:0.9rem;padding-bottom:4px">{away} {score_away}{edge_badge}</div>
    <div style="color:#ccc;font-size:0.9rem">{home}{neutral_tag} {score_home}</div>
  </div>
  <div style="text-align:center;min-width:90px">
    <div style="color:#7a95b5;font-size:0.65rem;font-weight:600;margin-bottom:2px">MODEL</div>
    <div style="color:#fff;font-size:0.8rem;white-space:nowrap">{m_spread}</div>
    <div style="color:#7a95b5;font-size:0.75rem">O/U {m_total}</div>
  </div>
  <div style="text-align:center;min-width:90px">
    <div style="color:#7a95b5;font-size:0.65rem;font-weight:600;margin-bottom:2px">VEGAS</div>
    <div style="color:#fff;font-size:0.8rem;white-space:nowrap">{v_spread}</div>
    <div style="color:#7a95b5;font-size:0.75rem">O/U {v_total}</div>
  </div>
  <div style="text-align:center;min-width:70px">
    <div style="color:#7a95b5;font-size:0.65rem;font-weight:600;margin-bottom:2px">ML</div>
    <div style="color:#aaa;font-size:0.75rem">{away_ml}</div>
    <div style="color:#aaa;font-size:0.75rem">{home_ml}</div>
  </div>
</div>'''

    if not rows_html:
        st.info("No games to display.")
        return

    board_html = f'<div style="background:#0a1628;border-radius:8px;border:1px solid #1a2744;overflow:hidden">{rows_html}</div>'
    st.html(board_html)


# ── Matchup card renderer ──────────────────────────────────────────────────

def render_matchup_card(row, idx, ratings_df, synopsis=None):
    """Render a sportsbook-style matchup card using native Streamlit components."""
    home = str(row.get("homeTeam", ""))
    away = str(row.get("awayTeam", ""))
    week = row.get("week")
    neutral = str(row.get("neutralSite", "")).lower() in ("true", "1", "yes")
    conf_game = str(row.get("conferenceGame", "")).lower() in ("true", "1", "yes")

    home_is_fcs = "(FCS)" in home
    away_is_fcs = "(FCS)" in away
    home_display = home.replace(" (FCS)", " *(FCS)*") if home_is_fcs else home
    away_display = away.replace(" (FCS)", " *(FCS)*") if away_is_fcs else away

    h_comp = row.get("home_composite")
    a_comp = row.get("away_composite")
    model_spread = row.get("predicted_spread")
    vegas_spread = row.get("vegas_spread")
    model_total  = row.get("predicted_total")
    vegas_total  = row.get("vegas_total")
    home_ml      = row.get("home_ml")
    away_ml      = row.get("away_ml")
    bet          = row.get("bet", "") or ""
    home_pts     = row.get("homePoints")
    away_pts     = row.get("awayPoints")

    def fmt_spread(v):
        try:
            v = float(v)
            fav = home.replace(" (FCS)", "") if v < 0 else away.replace(" (FCS)", "")
            if v == 0:
                return "PK"
            return f"{fav} -{abs(v):.1f}"
        except (TypeError, ValueError):
            return "—"

    def fmt_num(v):
        try:
            return f"{float(v):.1f}"
        except (TypeError, ValueError):
            return "—"

    def fmt_ml(v):
        try:
            v = float(v)
            return f"+{int(v)}" if v > 0 else str(int(v))
        except (TypeError, ValueError):
            return "—"

    with st.container(border=True):
        # Header row: badges + detail button
        hcol1, hcol2 = st.columns([8, 1])
        with hcol1:
            badges = []
            if week and pd.notna(week):
                badges.append(f"**Wk {int(week)}**")
            if neutral:
                badges.append("🏟 Neutral")
            if conf_game:
                badges.append("🏆 Conference")
            st.caption("  ·  ".join(badges) if badges else " ")
        with hcol2:
            if st.button("Detail", key=f"card_detail_{idx}", use_container_width=True):
                show_matchup_detail(row, ratings_df)

        # Teams row
        t_away, t_vs, t_home = st.columns([5, 1, 5])
        with t_away:
            st.markdown(f"### {away_display}")
            if a_comp and pd.notna(a_comp):
                st.caption(f"Rtg: {float(a_comp):+.1f}")
        with t_vs:
            st.markdown("<div style='text-align:center;padding-top:8px;color:#4d6a8a;font-weight:700;'>@</div>", unsafe_allow_html=True)
        with t_home:
            st.markdown(f"### {home_display}")
            if h_comp and pd.notna(h_comp):
                st.caption(f"Rtg: {float(h_comp):+.1f}")

        # Lines row
        l1, l2, l3 = st.columns(3)
        with l1:
            st.caption("SPREAD")
            st.markdown(f"**Model:** {fmt_spread(model_spread)}")
            st.markdown(f"**Vegas:** {fmt_spread(vegas_spread)}")
        with l2:
            st.caption("TOTAL")
            st.markdown(f"**Model:** {fmt_num(model_total)}")
            st.markdown(f"**Vegas:** {fmt_num(vegas_total)}")
        with l3:
            st.caption("MONEYLINE")
            if home_ml and pd.notna(home_ml):
                st.markdown(f"**{home.replace(' (FCS)','')}:** {fmt_ml(home_ml)}")
                st.markdown(f"**{away.replace(' (FCS)','')}:** {fmt_ml(away_ml)}")
            else:
                st.markdown("—")

        # Footer
        if bet:
            st.success(f"**Bet:** {bet}", icon="🎯")
        if home_pts and pd.notna(home_pts) and away_pts and pd.notna(away_pts):
            st.caption(f"Final: {away} {int(away_pts)} – {int(home_pts)} {home}")

        # AI synopsis
        if synopsis and not synopsis.startswith("Preview unavailable"):
            sentences = synopsis.split(". ")
            teaser = sentences[0] + ("." if not sentences[0].endswith(".") else "")
            st.markdown(f"*{teaser}*")
            if len(sentences) > 1:
                with st.expander("Full preview"):
                    st.markdown(synopsis)


# ── Matchup detail dialog ──────────────────────────────────────────────────

@st.dialog("Matchup Detail", width="large")
def show_matchup_detail(row, ratings_df):
    """Modal popup with full matchup breakdown."""
    from model.synopsis_generator import generate_synopsis

    home = str(row.get("homeTeam", "")).replace(" (FCS)", "")
    away = str(row.get("awayTeam", "")).replace(" (FCS)", "")
    neutral = str(row.get("neutralSite", "")).lower() in ("true", "1", "yes")
    week = row.get("week")

    def _get_rating(team, col):
        if ratings_df.empty:
            return None
        from data.team_names import normalize as _norm
        team_norm = _norm(team)
        r = ratings_df[ratings_df["team"] == team_norm]
        if r.empty or col not in r.columns:
            return None
        v = r[col].values[0]
        return None if pd.isna(v) else round(float(v), 1)

    h_comp = _get_rating(home, "composite") or row.get("home_composite")
    a_comp = _get_rating(away, "composite") or row.get("away_composite")
    h_fpi  = _get_rating(home, "fpi")
    a_fpi  = _get_rating(away, "fpi")
    h_off  = _get_rating(home, "offense.rating")
    a_off  = _get_rating(away, "offense.rating")
    h_def  = _get_rating(home, "defense.rating")
    a_def  = _get_rating(away, "defense.rating")

    model_spread = row.get("predicted_spread")
    vegas_spread = row.get("vegas_spread")
    model_total  = row.get("predicted_total")
    vegas_total  = row.get("vegas_total")
    home_ml      = row.get("home_ml")
    away_ml      = row.get("away_ml")
    bet_rec      = row.get("bet", "") or ""
    fcs_note     = row.get("fcs_note")

    def fmt_spread(v, h, a):
        try:
            v = float(v)
            if v < 0: return f"{h} {v:.1f}"
            if v > 0: return f"{a} -{abs(v):.1f}"
            return "Pick'em"
        except (TypeError, ValueError):
            return "—"

    def fmt_num(v):
        try: return f"{float(v):.1f}"
        except (TypeError, ValueError): return "—"

    def fmt_ml(v):
        try:
            v = float(v)
            return f"+{int(v)}" if v > 0 else str(int(v))
        except (TypeError, ValueError): return "—"

    site = "Neutral Site" if neutral else home
    wk_label = f"Week {int(week)}" if week and pd.notna(week) else ""
    st.markdown(f"### {away} @ {site}")
    if wk_label:
        st.caption(wk_label)
    st.divider()

    # Ratings
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{away}**")
        if a_comp is not None: st.metric("Composite", f"{float(a_comp):+.1f}")
        if a_fpi  is not None: st.metric("FPI", f"{a_fpi:+.1f}")
        if a_off  is not None: st.metric("SP+ Offense", f"{a_off:.1f}")
        if a_def  is not None: st.metric("SP+ Defense", f"{a_def:.1f}")
    with c2:
        st.markdown(f"**{home}**")
        if h_comp is not None: st.metric("Composite", f"{float(h_comp):+.1f}")
        if h_fpi  is not None: st.metric("FPI", f"{h_fpi:+.1f}")
        if h_off  is not None: st.metric("SP+ Offense", f"{h_off:.1f}")
        if h_def  is not None: st.metric("SP+ Defense", f"{h_def:.1f}")

    st.divider()

    # Lines
    l1, l2, l3 = st.columns(3)
    with l1:
        st.markdown("**Spread**")
        st.write(f"Model: {fmt_spread(model_spread, home, away)}")
        st.write(f"Vegas: {fmt_spread(vegas_spread, home, away)}")
    with l2:
        st.markdown("**Total**")
        st.write(f"Model: {fmt_num(model_total)}")
        st.write(f"Vegas: {fmt_num(vegas_total)}")
    with l3:
        st.markdown("**Moneyline**")
        if home_ml and pd.notna(home_ml):
            st.write(f"{home}: {fmt_ml(home_ml)}")
            st.write(f"{away}: {fmt_ml(away_ml)}")
        else:
            st.write("—")

    if bet_rec:
        st.success(f"**Model Recommendation:** {bet_rec}", icon="🎯")
    if fcs_note and pd.notna(fcs_note):
        st.info(fcs_note)

    st.divider()

    # AI Preview
    st.markdown("**AI Game Preview**")
    game_data = {
        "homeTeam": home, "awayTeam": away, "neutral": neutral,
        "home_composite": h_comp, "away_composite": a_comp,
        "home_off_rating": h_off, "away_off_rating": a_off,
        "home_def_rating": h_def, "away_def_rating": a_def,
        "predicted_spread": model_spread, "vegas_spread": vegas_spread,
        "predicted_total": model_total, "vegas_total": vegas_total,
        "week": week,
    }
    synopsis_week = int(week) if week and pd.notna(week) else None
    with st.spinner("Loading preview..."):
        synopsis = generate_synopsis(game_data, week=synopsis_week)
    if synopsis and not synopsis.startswith("Preview unavailable"):
        st.markdown(synopsis)
    else:
        st.caption("Preview not available.")


# ── Page: Power Rankings ───────────────────────────────────────────────────

if page == "📊 Power Rankings":
    st.title("Power Rankings")
    st.caption("Composite rating = SP+, FPI, Elo, Opp-Adj EPA (weighted)")

    if ratings_df.empty:
        st.warning("No ratings data found. Run the weekly update or check your data files.")
        st.stop()

    # ── Build team → conference lookup from schedule ───────────────────────
    ranked_df = ratings_df.copy()
    if not schedule_df.empty and "homeTeam" in schedule_df.columns:
        conf_map = {}
        for _, row in schedule_df[["homeTeam", "homeConference"]].dropna().iterrows():
            conf_map[row["homeTeam"]] = row["homeConference"]
        for _, row in schedule_df[["awayTeam", "awayConference"]].dropna().iterrows():
            conf_map[row["awayTeam"]] = row["awayConference"]
        ranked_df["conference"] = ranked_df["team"].map(conf_map).fillna("Independent")
    else:
        ranked_df["conference"] = "Independent"

    # ── Filters ───────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=False):
        conf_options = ["All Conferences"] + sorted(
            ranked_df["conference"].dropna().unique().tolist()
        )
        filter_conf = st.selectbox("Conference", conf_options, key="rankings_conf_filter")

    if filter_conf != "All Conferences":
        ranked_df = ranked_df[ranked_df["conference"] == filter_conf].copy()
        # Re-rank within the filtered set
        ranked_df = ranked_df.drop(columns=["rank"], errors="ignore")
        ranked_df = ranked_df.sort_values("composite", ascending=False).reset_index(drop=True)
        ranked_df.index += 1
        ranked_df.index.name = "rank"
        ranked_df = ranked_df.reset_index()

    # ── Default columns (public-facing) ──────────────────────────────────
    default_cols = {
        "rank": "Rank",
        "team": "Team",
        "composite": "⭐ Composite",
        "conference": "Conference",
        "sp_plus": "SP+",
        "fpi": "FPI",
        "elo": "Elo",
    }
    # ── Advanced columns (hidden by default) ─────────────────────────────
    advanced_cols = {
        "offense_overall": "Off EPA",
        "defense_overall": "Def EPA",
        "epa_net": "Net EPA",
        "returning_prod": "Ret. Prod.",
        "talent": "Talent",
    }

    def _make_table(col_map):
        avail = {k: v for k, v in col_map.items() if k in ranked_df.columns}
        df = ranked_df[list(avail.keys())].rename(columns=avail).copy()
        for col in df.columns:
            if df[col].dtype == float:
                df[col] = df[col].round(1)
        return df

    show_df = _make_table(default_cols)

    composite_config = {}
    if "⭐ Composite" in show_df.columns:
        composite_config["⭐ Composite"] = st.column_config.ProgressColumn(
            "⭐ Composite",
            min_value=float(show_df["⭐ Composite"].min()),
            max_value=float(show_df["⭐ Composite"].max()),
            format="%.1f",
        )

    st.dataframe(
        show_df,
        use_container_width=True,
        height=700,
        hide_index=True,
        column_config=composite_config,
    )

    # Advanced metrics expander
    adv_available = {k: v for k, v in advanced_cols.items() if k in ranked_df.columns}
    if adv_available:
        with st.expander("Advanced Metrics", expanded=False):
            adv_cols = {**{"rank": "Rank", "team": "Team"}, **advanced_cols}
            adv_df = _make_table({k: v for k, v in adv_cols.items() if k in ranked_df.columns})
            st.dataframe(adv_df, use_container_width=True, height=500, hide_index=True)

    st.divider()
    # Top 25 bar chart
    if "⭐ Composite" in show_df.columns:
        chart_n  = min(25, len(show_df))
        chart_df = show_df.head(chart_n)
        chart_title = (
            f"Top {chart_n} — {filter_conf}" if filter_conf != "All Conferences"
            else f"Top {chart_n} — Composite Power Rating"
        )
        fig = px.bar(
            chart_df,
            x="⭐ Composite", y="Team",
            orientation="h",
            color="⭐ Composite",
            color_continuous_scale="RdYlGn",
            title=chart_title,
            labels={"⭐ Composite": "Composite Rating"},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

    # Export
    st.download_button(
        "📥 Export to Excel",
        data=ratings_df.to_csv(index=False).encode(),
        file_name=f"cfb_power_rankings_{CURRENT_SEASON}.csv",
        mime="text/csv",
    )


# ── Page: Season Projections ──────────────────────────────────────────────

elif page == "🏆 Season Projections":
    st.title("Season Projections")
    st.caption(f"Projected win totals for every FBS team — {CURRENT_SEASON} season")

    if schedule_df.empty or ratings_df.empty:
        st.warning("Schedule or ratings data missing. Run the weekly update first.")
        st.stop()

    with st.spinner("Running model against full schedule..."):
        proj_df = load_season_projections(schedule_df, ratings_df)

    if proj_df.empty:
        st.warning("Could not generate projections.")
        st.stop()

    # Remove FCS teams — they appear as opponents in the schedule but aren't
    # FBS programs we want to project. Filter to teams in ratings_df (FBS only).
    fbs_team_set = set(ratings_df["team"].tolist())
    proj_df = proj_df[proj_df["team"].isin(fbs_team_set)].copy()

    # Try to load Vegas win totals
    vegas_totals = load_win_totals()
    has_vegas = not vegas_totals.empty and "wins_line" in vegas_totals.columns

    # Merge Vegas win totals if available
    if has_vegas:
        proj_df = proj_df.merge(
            vegas_totals[["team", "wins_line", "over_odds", "under_odds"]],
            on="team", how="left"
        )
        proj_df["model_vs_vegas"] = (proj_df["projected_wins"] - proj_df["wins_line"]).round(1)
    else:
        st.info(
            f"**Win totals not yet available from sportsbooks.** "
            f"Once posted, drop a CSV at `cache/win_totals_{CURRENT_SEASON}_manual.csv` "
            f"with columns: `team, wins_line, over_odds, under_odds` and reload.",
            icon="ℹ️",
        )

    # Merge ESPN FPI projected wins
    fpi_proj = load_fpi_projections()
    has_fpi_proj = not fpi_proj.empty and "fpi_proj_wins" in fpi_proj.columns
    if has_fpi_proj:
        proj_df = proj_df.merge(fpi_proj[["team", "fpi_proj_wins"]], on="team", how="left")

    # Load championship odds
    champ_df = load_championship_odds()
    has_champ = not champ_df.empty and "best_odds" in champ_df.columns
    if has_champ:
        proj_df = proj_df.merge(
            champ_df[["team", "best_odds", "dk_odds", "fd_odds", "betmgm_odds", "caesars_odds"]],
            on="team", how="left"
        )

    # Build conference lookup for filter
    all_confs = sorted(proj_df["conference"].dropna().unique().tolist())
    all_confs = [c for c in all_confs if c and c not in ("Independent",)]

    # ── Filters ──────────────────────────────────────────────────────────
    fcol1, fcol2 = st.columns([2, 2])
    conf_filter = fcol1.selectbox(
        "Conference", ["All Conferences"] + all_confs, key="proj_conf"
    )
    sort_options = {
        "Projected Wins": "projected_wins",
        "Model vs Vegas": "model_vs_vegas" if has_vegas else "projected_wins",
        "Title Odds (favorite first)": "best_odds" if has_champ else "projected_wins",
        "Win %": "win_pct",
        "Team (A–Z)": "team",
    }
    sort_by_label = fcol2.selectbox("Sort by", list(sort_options.keys()), key="proj_sort")
    sort_by = sort_options[sort_by_label]

    view_df = proj_df.copy()
    if conf_filter != "All Conferences":
        view_df = view_df[view_df["conference"] == conf_filter]

    # Sort and re-rank
    ascending = sort_by == "team"
    view_df = (
        view_df
        .drop(columns=["rank"], errors="ignore")
        .sort_values(sort_by, ascending=ascending)
        .reset_index(drop=True)
    )
    view_df.index += 1
    view_df.index.name = "rank"
    view_df = view_df.reset_index()

    st.divider()

    # ── Table ─────────────────────────────────────────────────────────────
    col_rename = {
        "rank":             "Rank",
        "team":             "Team",
        "projected_wins":   "Proj W",
        "fpi_proj_wins":    "ESPN FPI",
        "conference":       "Conference",
        "projected_losses": "Proj L",
        "win_pct":          "Win %",
        "floor_wins":       "Floor",
        "ceiling_wins":     "Ceiling",
    }
    if has_vegas:
        col_rename["wins_line"]      = "Vegas O/U"
        col_rename["model_vs_vegas"] = "Model Edge"
    if has_champ:
        col_rename["best_odds"] = "Title Odds"

    from data.odds_api_fetcher import fmt_american_odds
    show_cols = [c for c in col_rename if c in view_df.columns]
    table_df = view_df[show_cols].rename(columns=col_rename).copy()
    table_df["Win %"] = (table_df["Win %"] * 100).round(1).astype(str) + "%"
    if "Title Odds" in table_df.columns:
        table_df["Title Odds"] = table_df["Title Odds"].apply(fmt_american_odds)

    col_config = {
        "Proj W": st.column_config.ProgressColumn(
            "Proj W",
            min_value=0,
            max_value=float(proj_df["projected_wins"].max()),
            format="%.1f",
        ),
    }
    if has_vegas and "Model Edge" in table_df.columns:
        col_config["Model Edge"] = st.column_config.NumberColumn(
            "Model Edge", format="%+.1f"
        )

    st.dataframe(
        table_df,
        use_container_width=True,
        height=700,
        hide_index=True,
        column_config=col_config,
    )

    # ── Conference standings ───────────────────────────────────────────────
    st.divider()
    st.subheader("Conference Standings")

    if conf_filter != "All Conferences":
        conf_list = [conf_filter]
    else:
        conf_list = all_confs

    # Show 2 conferences per row
    for i in range(0, len(conf_list), 2):
        cols = st.columns(2)
        for j, conf in enumerate(conf_list[i:i+2]):
            conf_teams = proj_df[proj_df["conference"] == conf].sort_values(
                "projected_wins", ascending=False
            ).reset_index(drop=True)
            if conf_teams.empty:
                continue
            with cols[j]:
                st.markdown(f"**{conf}**")
                for _, row in conf_teams.iterrows():
                    wins  = row["projected_wins"]
                    floor = row["floor_wins"]
                    ceil  = row["ceiling_wins"]
                    edge_str = ""
                    if has_vegas and pd.notna(row.get("model_vs_vegas")):
                        mv = row["model_vs_vegas"]
                        color = "#00b074" if mv > 0.5 else ("#ff6633" if mv < -0.5 else "#7a95b5")
                        edge_str = f" <span style='color:{color};font-size:0.8rem'>({mv:+.1f})</span>"
                    st.markdown(
                        f"{row['team']} — **{wins}w** <span style='color:#4d6a8a;font-size:0.8rem'>"
                        f"({floor}–{ceil})</span>{edge_str}",
                        unsafe_allow_html=True,
                    )
                st.write("")

    # ── Championship odds ──────────────────────────────────────────────────
    if has_champ:
        st.divider()
        st.subheader("🏆 National Championship Odds")
        st.caption("Best available odds across DraftKings, FanDuel, BetMGM, Caesars")

        # Merge in projected wins so we can compare model ranking vs market odds
        proj_slim = proj_df[["team", "projected_wins", "floor_wins", "ceiling_wins"]].copy()
        champ_display = champ_df.merge(proj_slim, on="team", how="left")

        for col in ["best_odds", "dk_odds", "fd_odds", "betmgm_odds", "caesars_odds"]:
            if col in champ_display.columns:
                champ_display[col] = champ_display[col].apply(fmt_american_odds)

        champ_display = champ_display.rename(columns={
            "team":            "Team",
            "projected_wins":  "Proj W",
            "floor_wins":      "Floor",
            "ceiling_wins":    "Ceiling",
            "best_odds":       "Best Odds",
            "best_book":       "Best Book",
            "dk_odds":         "DraftKings",
            "fd_odds":         "FanDuel",
            "betmgm_odds":     "BetMGM",
            "caesars_odds":    "Caesars",
        })

        # Filter to conference selection if active
        if conf_filter != "All Conferences":
            conf_teams = set(proj_df[proj_df["conference"] == conf_filter]["team"])
            champ_display = champ_display[champ_display["Team"].isin(conf_teams)]

        st.dataframe(champ_display, use_container_width=True, hide_index=True)

    # ── Export ────────────────────────────────────────────────────────────
    st.divider()
    st.download_button(
        "📥 Export Projections",
        data=proj_df.to_csv(index=False).encode(),
        file_name=f"cfb_season_projections_{CURRENT_SEASON}.csv",
        mime="text/csv",
    )


# ── Page: Title Odds ──────────────────────────────────────────────────────

elif page == "🎰 Title Odds":
    st.title("National Championship Odds")
    st.caption("Odds to win the CFB National Championship — sorted by favorite. Updates live as books move lines.")

    col_refresh, col_status = st.columns([1, 4])
    if col_refresh.button("🔄 Refresh Odds"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Fetching latest odds..."):
        champ_df = load_championship_odds()

    fpi_proj = load_fpi_projections()
    has_fpi_proj = not fpi_proj.empty

    if champ_df.empty and not has_fpi_proj:
        col_status.warning("Championship odds unavailable — check back closer to the season.")
        st.stop()

    # If no market odds yet, show ESPN FPI projections as a stand-in
    if champ_df.empty and has_fpi_proj:
        col_status.info("Market odds not yet posted — showing ESPN FPI preseason projections.")
        st.subheader("ESPN FPI Preseason Projections")
        fpi_display = fpi_proj.copy().sort_values("fpi_playoff_pct", ascending=False).reset_index(drop=True)
        fpi_display.insert(0, "Rank", fpi_display.index + 1)
        fpi_display = fpi_display.rename(columns={
            "team":             "Team",
            "fpi_proj_wins":    "Proj W",
            "fpi_win_conf_pct": "Win Conf %",
            "fpi_playoff_pct":  "Playoff %",
            "fpi_make_nc_pct":  "Make NC %",
            "fpi_win_nc_pct":   "Win NC %",
        })
        st.dataframe(fpi_display, use_container_width=True, hide_index=True)
        st.stop()

    from data.odds_api_fetcher import fmt_american_odds

    col_status.success(f"✅ {len(champ_df)} teams listed")

    # ── Add implied probability (no-vig) ─────────────────────────────────
    def _american_to_prob(odds):
        try:
            v = float(odds)
            if v < 0:
                return abs(v) / (abs(v) + 100)
            else:
                return 100 / (v + 100)
        except (TypeError, ValueError):
            return None

    champ_df = champ_df.copy()
    champ_df["impl_prob"] = champ_df["best_odds"].apply(_american_to_prob)

    # ── Summary metrics ───────────────────────────────────────────────────
    st.divider()
    top3 = champ_df.head(3)
    m_cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        prob = row["impl_prob"]
        prob_str = f"{prob*100:.1f}%" if prob else "—"
        m_cols[i].metric(
            f"#{i+1} Favorite",
            row["team"],
            f"{fmt_american_odds(row['best_odds'])}  ·  {prob_str} implied"
        )

    st.divider()

    # ── Full odds table ───────────────────────────────────────────────────
    display_df = champ_df.copy()
    if has_fpi_proj:
        display_df = display_df.merge(
            fpi_proj[["team", "fpi_proj_wins", "fpi_playoff_pct", "fpi_win_nc_pct"]],
            on="team", how="left"
        )
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))
    display_df["Implied %"] = display_df["impl_prob"].apply(
        lambda v: f"{v*100:.1f}%" if v else "—"
    )
    for col in ["best_odds", "dk_odds", "fd_odds", "betmgm_odds", "caesars_odds"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_american_odds)

    display_df = display_df.rename(columns={
        "team":              "Team",
        "best_odds":         "Best Odds",
        "best_book":         "Best Book",
        "dk_odds":           "DraftKings",
        "fd_odds":           "FanDuel",
        "betmgm_odds":       "BetMGM",
        "caesars_odds":      "Caesars",
        "fpi_proj_wins":     "ESPN Proj W",
        "fpi_playoff_pct":   "ESPN PO%",
        "fpi_win_nc_pct":    "ESPN NC%",
    })

    show_cols = ["Rank", "Team", "Best Odds", "Implied %", "ESPN Proj W", "ESPN PO%", "ESPN NC%", "DraftKings", "FanDuel", "BetMGM", "Caesars", "Best Book"]
    show_cols = [c for c in show_cols if c in display_df.columns]

    st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

    # ── Bar chart — implied probability top 20 ────────────────────────────
    st.divider()
    chart_df = champ_df.head(20).copy()
    chart_df["Team"] = chart_df["team"]
    chart_df["Implied Win %"] = (chart_df["impl_prob"] * 100).round(1)
    chart_df = chart_df.sort_values("Implied Win %", ascending=True)

    fig = px.bar(
        chart_df,
        x="Implied Win %",
        y="Team",
        orientation="h",
        color="Implied Win %",
        color_continuous_scale="RdYlGn",
        title="Top 20 — Market-Implied Championship Win Probability",
        labels={"Implied Win %": "Implied Win %"},
        text="Implied Win %",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=600,
        plot_bgcolor="#0f1923",
        paper_bgcolor="#0f1923",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e3050"),
        yaxis=dict(gridcolor="#1e3050"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page: Betting Edges ────────────────────────────────────────────────────

elif page == "🎯 Betting Edges":
    st.title("Betting Edges")
    st.caption("Model vs. Vegas — games auto-pulled from Owls Insight")

    if ratings_df.empty:
        st.warning("No ratings data found. Run setup_season.py first.")
        st.stop()

    # ── Pull live lines ───────────────────────────────────────────────────
    col_refresh, col_status = st.columns([1, 4])
    if col_refresh.button("🔄 Refresh Lines"):
        st.cache_data.clear()
        st.rerun()

    live_lines, lines_source = load_live_lines()

    if live_lines.empty:
        col_status.warning("No live lines yet — it's the offseason.")

        # ── Offseason landing ─────────────────────────────────────────────
        import datetime
        season_start = datetime.date(CURRENT_SEASON, 8, 29)
        days_until   = (season_start - datetime.date.today()).days

        st.divider()
        if days_until > 0:
            st.info(f"🏈 **{days_until} days until Week 1** ({season_start.strftime('%B %d, %Y')}). "
                    f"Live betting lines will appear here once books open for the season.")
        else:
            st.info("🏈 Season is underway — lines should appear once books post odds for upcoming games.")

        # Teaser: top 10 power rankings
        if not ratings_df.empty and "composite" in ratings_df.columns:
            st.subheader("Current Power Rankings — Top 10")
            top10 = ratings_df.nlargest(10, "composite")[["team", "composite"]].copy()
            top10.columns = ["Team", "Composite Rating"]
            top10["Composite Rating"] = top10["Composite Rating"].round(1)
            top10.index = range(1, 11)
            top10.index.name = "Rank"
            st.dataframe(top10, use_container_width=True, hide_index=False)

        # Teaser: season projections
        st.subheader("Season Win Projections")
        st.caption("See how every team is projected to finish — sorted by projected wins.")
        with st.spinner("Loading projections..."):
            _proj = load_season_projections(schedule_df, ratings_df)
        if not _proj.empty:
            _fbs = set(ratings_df["team"].tolist())
            _proj = _proj[_proj["team"].isin(_fbs)]
            _top = _proj.head(15)[["team", "conference", "projected_wins",
                                   "floor_wins", "ceiling_wins"]].copy()
            _top.columns = ["Team", "Conference", "Proj W", "Floor", "Ceiling"]
            st.dataframe(_top, use_container_width=True, hide_index=True)
            st.caption("Full projections available on the 🏆 Season Projections page.")

        # Championship odds teaser
        _champ = load_championship_odds()
        if not _champ.empty:
            st.subheader("National Championship Odds — Top 10 Favorites")
            from data.odds_api_fetcher import fmt_american_odds
            _champ_top = _champ.head(10)[["team", "best_odds", "best_book"]].copy()
            _champ_top["best_odds"] = _champ_top["best_odds"].apply(fmt_american_odds)
            _champ_top.columns = ["Team", "Best Odds", "Best Book"]
            st.dataframe(_champ_top, use_container_width=True, hide_index=True)

        st.stop()
    else:
        col_status.success(f"✅ {len(live_lines)} games loaded — {lines_source}")

    # ── Threshold controls ────────────────────────────────────────────────
    with st.expander("⚙️ Edge Thresholds", expanded=False):
        _t1, _t2 = st.columns(2)
        spread_thresh = _t1.number_input("Spread edge (pts)", value=EDGE_THRESHOLD_SPREAD, step=0.5, min_value=0.5)
        total_thresh  = _t2.number_input("Total edge (pts)",  value=EDGE_THRESHOLD_TOTAL,  step=0.5, min_value=0.5)

    # ── Week filter ───────────────────────────────────────────────────────
    if not live_lines.empty and "week" in live_lines.columns:
        available_weeks = sorted(live_lines["week"].dropna().unique().astype(int))
        if available_weeks:
            selected_edge_week = st.selectbox(
                "Filter by Week",
                options=available_weeks,
                index=0,
                format_func=lambda w: f"Week {w}",
            )
            live_lines = live_lines[live_lines["week"] == selected_edge_week].copy()
            st.caption(f"Showing **{len(live_lines)} games** with lines for Week {selected_edge_week}")

    # ── Run predictions against live lines ───────────────────────────────
    if not live_lines.empty:
        neutral_col = "neutralSite_y" if "neutralSite_y" in live_lines.columns else \
                      "neutralSite" if "neutralSite" in live_lines.columns else None
        live_sched = live_lines[["homeTeam", "awayTeam"]].copy()
        live_sched["neutralSite"] = live_lines[neutral_col] if neutral_col else False

        with st.spinner("Running predictions..."):
            predicted = predict_all_games(live_sched, ratings_df)

        # Apply historical situational adjustments to predicted totals
        matchup_ou, team_ou, team_ats = load_situational_tendencies()
        predicted = apply_adjustments_to_predictions(predicted, matchup_ou, team_ou, team_ats)

        # Use adjusted total for edge finding (falls back to predicted_total if no adj)
        if "predicted_total_adj" in predicted.columns:
            predicted["predicted_total"] = predicted["predicted_total_adj"]

        # Pass Vegas lines separately to edge finder
        lines_for_edge = live_lines[["homeTeam", "awayTeam", "spread", "overUnder"]].rename(
            columns={"spread": "vegas_spread", "overUnder": "vegas_total"}
        )
        # Carry week into predictions — normalize keys so names match.
        # Primary: use week from live_lines if available.
        # Fallback: look up week from the season schedule CSV.
        from data.team_names import normalize as _norm
        if "week" in live_lines.columns and live_lines["week"].notna().any():
            week_map = {
                (_norm(str(h)), _norm(str(a))): w
                for h, a, w in zip(live_lines["homeTeam"], live_lines["awayTeam"], live_lines["week"])
            }
        elif not schedule_df.empty and "week" in schedule_df.columns:
            week_map = {
                (_norm(str(h)), _norm(str(a))): w
                for h, a, w in zip(schedule_df["homeTeam"], schedule_df["awayTeam"], schedule_df["week"])
            }
        else:
            week_map = {}
        if week_map:
            predicted["week"] = predicted.apply(
                lambda r: week_map.get((_norm(str(r.get("homeTeam",""))), _norm(str(r.get("awayTeam",""))))), axis=1
            )

        edges = find_edges(predicted, lines_for_edge)
        # Add situational ATS flags per edge
        if not team_ats.empty and "vegas_spread" in edges.columns:
            edges["situational_note"] = edges.apply(
                lambda r: get_ats_situational_note(
                    r.get("homeTeam",""), r.get("awayTeam",""),
                    r.get("vegas_spread"), r.get("bet_spread"), team_ats
                ), axis=1
            )
        edges_summary = summarize_edges(edges)

        # ── Pre-generate synopses for all edge games ───────────────────────
        from model.synopsis_generator import generate_synopses_batch, _game_key
        synopsis_week = selected_edge_week if "selected_edge_week" in dir() else None
        if not edges_summary.empty:
            edge_games = edges_summary.to_dict("records")
            # Enrich with ratings for better synopses
            for g in edge_games:
                home_row = ratings_df[ratings_df["team"] == g.get("homeTeam", "")]
                away_row = ratings_df[ratings_df["team"] == g.get("awayTeam", "")]
                if not home_row.empty:
                    g["home_composite"] = home_row["composite"].values[0]
                    if "offense.rating" in home_row.columns:
                        g["home_off_rating"] = home_row["offense.rating"].values[0]
                    if "defense.rating" in home_row.columns:
                        g["home_def_rating"] = home_row["defense.rating"].values[0]
                if not away_row.empty:
                    g["away_composite"] = away_row["composite"].values[0]
                    if "offense.rating" in away_row.columns:
                        g["away_off_rating"] = away_row["offense.rating"].values[0]
                    if "defense.rating" in away_row.columns:
                        g["away_def_rating"] = away_row["defense.rating"].values[0]

            with st.spinner("Generating AI game previews..."):
                synopses = generate_synopses_batch(edge_games, week=synopsis_week)
        else:
            synopses = {}

        # ── Actionable edges ──────────────────────────────────────────────
        st.subheader("🔥 Actionable Edges")

        if edges_summary.empty:
            st.info("No edges above threshold right now. Lines may not have settled yet, or try lowering the threshold in the sidebar.")
        else:
            for _, row in edges_summary.iterrows():
                grade = row.get("edge_grade", "")
                color = grade_color(grade)
                with st.container():
                    wk = row.get("week")
                    wk_label = f"Wk {int(wk)}" if pd.notna(wk) else ""
                    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
                    c1.metric("Away", row.get("awayTeam", ""))
                    c2.metric("Home", f"{row.get('homeTeam', '')}  {wk_label}")
                    if row.get("bet_spread"):
                        edge_val = row.get("edge_spread", 0)
                        c3.metric("Spread Bet", row["bet_spread"],
                                  delta=f"{edge_val:+.1f} pts" if not pd.isna(edge_val) else "")
                    if row.get("bet_total"):
                        edge_val = row.get("edge_total", 0)
                        c4.metric("Total Bet", row["bet_total"],
                                  delta=f"{edge_val:+.1f} pts" if not pd.isna(edge_val) else "")
                    c5.markdown(f"<h3 style='color:{color};text-align:center'>{grade}</h3>",
                                unsafe_allow_html=True)
                    st.caption(
                        f"Model spread: **{row.get('predicted_spread')}** | Vegas: **{row.get('vegas_spread')}** | "
                        f"Model total: **{row.get('predicted_total')}** | Vegas: **{row.get('vegas_total')}** | "
                        f"Confidence: {row.get('confidence', 0):.0%}"
                    )
                    if row.get("hist_total_note"):
                        adj = row.get("total_hist_adj", 0)
                        direction = "▼" if adj > 0 else "▲"
                        st.caption(f"📊 Historical: {row['hist_total_note']} ({direction}{abs(adj):.1f} pt total adj)")
                    if row.get("situational_note"):
                        st.caption(f"📋 Situational: {row['situational_note']}")
                    if row.get("coaching_flag"):
                        st.caption(f"⚠️ {row['coaching_flag']}")

                    # AI game preview
                    syn_key = _game_key(row.get("homeTeam", ""), row.get("awayTeam", ""))
                    synopsis = synopses.get(syn_key)
                    if synopsis and not synopsis.startswith("Preview unavailable"):
                        # First sentence always visible as a teaser
                        sentences = synopsis.split(". ")
                        teaser = sentences[0] + ("." if not sentences[0].endswith(".") else "")
                        st.markdown(f"*{teaser}*")
                        if len(sentences) > 1:
                            with st.expander("Full game preview"):
                                st.markdown(synopsis)

                    st.divider()

        # ── All predictions table ─────────────────────────────────────────
        st.subheader("All Games with Lines")
        show_cols = [c for c in [
            "week", "homeTeam", "awayTeam", "predicted_spread", "vegas_spread", "edge_spread",
            "predicted_total", "vegas_total", "edge_total",
            "total_hist_adj", "bet_spread", "bet_total", "edge_grade", "confidence",
            "hist_total_note", "situational_note",
        ] if c in edges.columns]
        games_with_lines = edges[edges["vegas_spread"].notna()][show_cols].copy()
        if not games_with_lines.empty:
            games_with_lines = games_with_lines.sort_values("edge_grade", na_position="last")
            games_with_lines = fmt_spread(games_with_lines,
                                          ["predicted_spread", "vegas_spread"])
            st.caption("( ) denotes away team is the favorite")
            st.dataframe(games_with_lines, use_container_width=True)

        # ── Multi-book line shopping ──────────────────────────────────────
        st.subheader("📚 Line Shopping — All Sportsbooks")
        multibook_df, mb_source = load_multibook_lines()
        if multibook_df.empty:
            st.info("Multi-book data unavailable.")
        else:
            # Filter to the same week as the main view
            if "week" in live_lines.columns and not live_lines.empty:
                # Build a set of (home, away) for the current week filter
                week_games = set(zip(live_lines["homeTeam"], live_lines["awayTeam"]))
                if week_games:
                    mask = multibook_df.apply(
                        lambda r: (r["homeTeam"], r["awayTeam"]) in week_games, axis=1
                    )
                    mb_view = multibook_df[mask].copy()
                else:
                    mb_view = multibook_df.copy()
            else:
                mb_view = multibook_df.copy()

            if mb_view.empty:
                st.info("No multi-book data for the selected week.")
            else:
                # Let user pick a game to compare
                game_options = mb_view[["homeTeam", "awayTeam"]].drop_duplicates()
                game_labels = [f"{r.awayTeam} @ {r.homeTeam}" for _, r in game_options.iterrows()]
                selected_game_label = st.selectbox(
                    "Select game to compare books",
                    options=game_labels,
                    key="mb_game_select"
                )
                sel_idx = game_labels.index(selected_game_label)
                sel_home = game_options.iloc[sel_idx]["homeTeam"]
                sel_away = game_options.iloc[sel_idx]["awayTeam"]

                game_mb = mb_view[
                    (mb_view["homeTeam"] == sel_home) & (mb_view["awayTeam"] == sel_away)
                ][["book", "spread", "home_ml", "away_ml", "overUnder"]].copy()
                game_mb.columns = ["Book", "Spread (Home)", "Home ML", "Away ML", "Total"]
                game_mb = game_mb.reset_index(drop=True)

                # Format: spreads/totals to 1 decimal, moneylines to no decimal
                for col in ["Spread (Home)", "Total"]:
                    if col in game_mb.columns:
                        game_mb[col] = pd.to_numeric(game_mb[col], errors="coerce").round(1)
                for col in ["Home ML", "Away ML"]:
                    if col in game_mb.columns:
                        game_mb[col] = pd.to_numeric(game_mb[col], errors="coerce").apply(
                            lambda v: int(v) if pd.notna(v) else None
                        )

                # Highlight best values
                def highlight_best(df):
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    for col, best_fn in [("Spread (Home)", min), ("Home ML", max), ("Away ML", max), ("Total", max)]:
                        if col in df.columns:
                            valid = df[col].dropna()
                            if not valid.empty:
                                best_val = best_fn(valid)
                                styles.loc[df[col] == best_val, col] = "background-color: #1a472a; color: white; font-weight: bold"
                    return styles

                fmt = {}
                for col in ["Spread (Home)", "Total"]:
                    if col in game_mb.columns:
                        fmt[col] = "{:.1f}"
                for col in ["Home ML", "Away ML"]:
                    if col in game_mb.columns:
                        fmt[col] = "{:.0f}"

                st.caption(f"Source: {mb_source} | Green = best available line")
                st.dataframe(
                    game_mb.style.apply(highlight_best, axis=None).format(fmt, na_rep="—"),
                    use_container_width=True,
                    hide_index=True,
                )

        # ── Line Movement ─────────────────────────────────────────────────
        st.subheader("📈 Line Movement")
        hist_lines = load_historical_lines(CURRENT_SEASON)
        if hist_lines.empty:
            st.info("No historical line movement data yet — available once the season starts.")
        else:
            # Get best closing line per game (prefer DraftKings)
            hl = hist_lines.copy()
            if "provider" in hl.columns:
                dk = hl[hl["provider"] == "DraftKings"]
                hl = dk if not dk.empty else hl
            hl = hl.drop_duplicates(subset=["homeTeam", "awayTeam", "week"])

            # Only keep games with both opening and closing lines
            hl = hl[hl["spreadOpen"].notna() & hl["spread"].notna()].copy()
            if not hl.empty:
                hl["movement"] = hl["spread"] - hl["spreadOpen"]
                hl["move_label"] = hl["movement"].apply(
                    lambda m: f"← Home {abs(m):.1f}" if m < -0.5
                    else (f"Away {abs(m):.1f} →" if m > 0.5 else "No move")
                )
                hl["significant"] = hl["movement"].abs() >= 2.0

                lm_filter_week = st.selectbox(
                    "Week", options=["All"] + sorted(hl["week"].dropna().unique().astype(int).tolist()),
                    key="lm_week", format_func=lambda w: f"Week {w}" if w != "All" else "All Weeks"
                )
                lm_sig_only = st.checkbox("Significant moves only (≥2 pts)", value=False, key="lm_sig")

                lm_view = hl.copy()
                if lm_filter_week != "All":
                    lm_view = lm_view[lm_view["week"] == int(lm_filter_week)]
                if lm_sig_only:
                    lm_view = lm_view[lm_view["significant"]]

                lm_view = lm_view.sort_values("movement").reset_index(drop=True)

                lm_cols = [c for c in [
                    "week", "homeTeam", "awayTeam",
                    "spreadOpen", "spread", "movement", "move_label",
                    "overUnderOpen", "overUnder",
                    "homeScore", "awayScore",
                ] if c in lm_view.columns]
                lm_display = lm_view[lm_cols].rename(columns={
                    "spreadOpen": "Open Spread", "spread": "Close Spread",
                    "movement": "Move", "move_label": "Direction",
                    "overUnderOpen": "Open O/U", "overUnder": "Close O/U",
                    "homeScore": "Home Pts", "awayScore": "Away Pts",
                })

                def color_movement(val):
                    if isinstance(val, float):
                        if val <= -2:
                            return "color: #33cc33; font-weight: bold"
                        elif val >= 2:
                            return "color: #ff6633; font-weight: bold"
                    return ""

                st.caption(f"{len(lm_display)} games | Green = line moved toward home, Orange = toward away")
                st.dataframe(
                    lm_display.style.applymap(color_movement, subset=["Move"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Opening line data not yet available for this season.")



elif page == "📅 Schedule & Predictions":
    _sched_title_col, _sched_refresh_col = st.columns([6, 1])
    _sched_title_col.title("Schedule & Predictions")
    if _sched_refresh_col.button("🔄 Refresh Lines", key="sched_refresh"):
        st.cache_data.clear()
        st.rerun()

    # Show lines source / age
    _lines_cache = "cache/lines_live.csv"
    if os.path.exists(_lines_cache):
        import time as _time
        _age_mins = int((_time.time() - os.path.getmtime(_lines_cache)) / 60)
        st.caption(f"Lines last updated: {_age_mins}m ago — click Refresh to pull latest")
    else:
        st.caption("Lines via The Odds API / DraftKings — click Refresh to update")

    if schedule_df.empty:
        st.warning("No schedule data available.")
        st.stop()

    # ── Week selector — centered pill tabs on the page ─────────────────────
    max_week = int(schedule_df["week"].max()) if "week" in schedule_df.columns else 15
    week_options = ["All"] + list(range(1, max_week + 1))

    # Center the pills using columns
    _, center_col, _ = st.columns([1, 6, 1])
    with center_col:
        page_week = st.pills(
            "Select Week",
            options=week_options,
            format_func=lambda w: "All Weeks" if w == "All" else f"Wk {w}",
            default=week_options[1],
            key="sched_week_pills",
        )
    if page_week is None:
        page_week = week_options[1]

    st.divider()

    # ── Team / Conference filters ───────────────────────────────────────────
    filter_col1, filter_col2 = st.columns(2)

    # Build conference list from both home and away conferences
    all_conferences = sorted(set(
        schedule_df["homeConference"].dropna().tolist() +
        schedule_df["awayConference"].dropna().tolist()
    ))
    with filter_col1:
        filter_conf = st.selectbox("Filter by Conference", ["All Conferences"] + all_conferences, key="sched_conf_filter")

    # Build team list — optionally scoped to selected conference
    if filter_conf != "All Conferences":
        conf_teams = sorted(set(
            schedule_df.loc[schedule_df["homeConference"] == filter_conf, "homeTeam"].tolist() +
            schedule_df.loc[schedule_df["awayConference"] == filter_conf, "awayTeam"].tolist()
        ))
    else:
        conf_teams = sorted(set(
            schedule_df["homeTeam"].dropna().tolist() +
            schedule_df["awayTeam"].dropna().tolist()
        ))
    with filter_col2:
        filter_team = st.selectbox("Filter by Team", ["All Teams"] + conf_teams, key="sched_team_filter")

    # Apply week filter
    if page_week == "All":
        week_sched = schedule_df.copy()
    else:
        week_sched = schedule_df[schedule_df["week"] == page_week].copy()

    # Apply conference filter: keep games where either team is in the conference
    if filter_conf != "All Conferences":
        week_sched = week_sched[
            (week_sched["homeConference"] == filter_conf) |
            (week_sched["awayConference"] == filter_conf)
        ]

    # Apply team filter: keep games where either team matches
    if filter_team != "All Teams":
        week_sched = week_sched[
            (week_sched["homeTeam"] == filter_team) |
            (week_sched["awayTeam"] == filter_team)
        ]

    week_label = "All Weeks" if page_week == "All" else f"Week {page_week}"
    st.caption(f"Showing **{week_label}** — {len(week_sched)} game(s)"
               + (f" · {filter_conf}" if filter_conf != "All Conferences" else "")
               + (f" · {filter_team}" if filter_team != "All Teams" else ""))

    # Bye week: team filter active but no games found for this week
    if week_sched.empty and filter_team != "All Teams":
        st.info(f"**{filter_team}** has a bye in {week_label}.")
        st.stop()

    if week_sched.empty:
        st.info(f"No games scheduled for {week_label}.")
        st.stop()

    if not ratings_df.empty:
        with st.spinner("Generating predictions..."):
            week_sched = predict_all_games(week_sched, ratings_df)

    # Merge Vegas lines — try Owls cache, then Odds API, then CFBD cache
    _lines_loaded = False
    _lines_cache = "cache/lines_live.csv"
    _cfbd_cache  = os.path.join(os.path.dirname(__file__), f"cache/lines_{CURRENT_SEASON}.csv")

    if os.path.exists(_lines_cache):
        try:
            _raw = pd.read_csv(_lines_cache)
            _want = {"homeTeam", "awayTeam", "spread", "overUnder", "home_ml", "away_ml"}
            _cols = [c for c in _want if c in _raw.columns]
            vegas_df = _raw[_cols].rename(columns={"spread": "vegas_spread", "overUnder": "vegas_total"})
            week_sched = week_sched.merge(vegas_df, on=["homeTeam", "awayTeam"], how="left")
            _lines_loaded = True
        except Exception:
            pass

    if not _lines_loaded:
        try:
            from data.odds_api_fetcher import fetch_ncaaf_game_lines
            from data.team_names import normalize as _norm
            _odds = fetch_ncaaf_game_lines()
            if not _odds.empty:
                # Normalize both sides so team names align with schedule
                _odds = _odds.copy()
                _odds["homeTeam"] = _odds["homeTeam"].map(_norm)
                _odds["awayTeam"] = _odds["awayTeam"].map(_norm)
                _want = {"homeTeam", "awayTeam", "spread", "overUnder", "home_ml", "away_ml"}
                _cols = [c for c in _want if c in _odds.columns]
                vegas_df = _odds[_cols].rename(columns={"spread": "vegas_spread", "overUnder": "vegas_total"})
                # Normalize schedule team names for merge, then restore
                _sched_norm = week_sched[["homeTeam", "awayTeam"]].copy()
                _sched_norm["homeTeam_n"] = _sched_norm["homeTeam"].map(_norm)
                _sched_norm["awayTeam_n"] = _sched_norm["awayTeam"].map(_norm)
                vegas_df = vegas_df.rename(columns={"homeTeam": "homeTeam_n", "awayTeam": "awayTeam_n"})
                week_sched = week_sched.assign(
                    homeTeam_n=week_sched["homeTeam"].map(_norm),
                    awayTeam_n=week_sched["awayTeam"].map(_norm),
                ).merge(vegas_df, on=["homeTeam_n", "awayTeam_n"], how="left").drop(
                    columns=["homeTeam_n", "awayTeam_n"], errors="ignore"
                )
                _lines_loaded = True
        except Exception:
            pass

    if not _lines_loaded and os.path.exists(_cfbd_cache):
        try:
            _raw = pd.read_csv(_cfbd_cache)
            _raw = _raw[_raw["spread"].notna()].copy()
            _dk = _raw[_raw["provider"].str.lower() == "draftkings"]
            _raw = _dk if not _dk.empty else _raw
            _raw = _raw.drop_duplicates(subset=["homeTeam", "awayTeam"])
            _want = {"homeTeam", "awayTeam", "spread", "overUnder"}
            _cols = [c for c in _want if c in _raw.columns]
            vegas_df = _raw[_cols].rename(columns={"spread": "vegas_spread", "overUnder": "vegas_total"})
            week_sched = week_sched.merge(vegas_df, on=["homeTeam", "awayTeam"], how="left")
        except Exception:
            pass

    # Compute bet recommendations on raw numerics before formatting
    week_sched = compute_bet_recommendations(
        week_sched, spread_thresh, total_thresh
    )

    # Tag FCS teams by name before building display
    fbs_teams = set(ratings_df["team"].tolist()) if not ratings_df.empty else set()
    if "homeClassification" in week_sched.columns:
        week_sched["homeTeam"] = week_sched.apply(
            lambda r: f"{r['homeTeam']} (FCS)" if str(r.get("homeClassification", "")).lower() == "fcs" else r["homeTeam"], axis=1
        )
        week_sched["awayTeam"] = week_sched.apply(
            lambda r: f"{r['awayTeam']} (FCS)" if str(r.get("awayClassification", "")).lower() == "fcs" else r["awayTeam"], axis=1
        )
    else:
        # Fallback: check against FBS ratings
        week_sched["homeTeam"] = week_sched["homeTeam"].apply(
            lambda t: f"{t} (FCS)" if t not in fbs_teams else t
        )
        week_sched["awayTeam"] = week_sched["awayTeam"].apply(
            lambda t: f"{t} (FCS)" if t not in fbs_teams else t
        )

    # Build clean display DataFrame
    col_map = {
        "homeTeam":        "Home",
        "awayTeam":        "Away",
        "neutralSite":     "Neutral",
        "bet":             "Bet",
        "predicted_spread":"Model Spread",
        "vegas_spread":    "Vegas Spread",
        "home_ml":         "Home ML",
        "away_ml":         "Away ML",
        "predicted_total": "Model Total",
        "vegas_total":     "Vegas Total",
        "home_composite":  "Home Rtg",
        "away_composite":  "Away Rtg",
        "homePoints":      "Home Pts",
        "awayPoints":      "Away Pts",
    }
    display_cols = [c for c in col_map if c in week_sched.columns]
    display_df = week_sched[display_cols].rename(columns=col_map).copy()

    # Format spreads: always show favorite's name with minus sign
    for col, home_col, away_col in [
        ("Model Spread", "Home", "Away"),
        ("Vegas Spread", "Home", "Away"),
    ]:
        if col in display_df.columns:
            def _fmt_spread_row(row, col=col, home_col=home_col, away_col=away_col):
                v = pd.to_numeric(row[col], errors="coerce")
                if pd.isna(v): return "—"
                if v == 0: return "PK"
                fav = row[home_col] if v < 0 else row[away_col]
                return f"{fav} -{abs(v):.1f}"
            display_df[col] = display_df.apply(_fmt_spread_row, axis=1)

    # Format moneylines as American odds
    for col in ["Home ML", "Away ML"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").apply(
                lambda v: f"+{int(v)}" if pd.notna(v) and v > 0
                else (f"{int(v)}" if pd.notna(v) else "—")
            )

    # Format ratings to 1 decimal
    for col in ["Home Rtg", "Away Rtg"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(1)

    # Neutral site: show checkmark or blank instead of True/False
    if "Neutral" in display_df.columns:
        display_df["Neutral"] = display_df["Neutral"].apply(
            lambda v: "✓" if str(v).lower() in ("true", "1", "yes") else ""
        )

    week_sched = week_sched.reset_index(drop=True)

    # ── View toggle ────────────────────────────────────────────────────────
    _v_col, _ = st.columns([2, 5])
    with _v_col:
        view_mode = st.segmented_control(
            "View", ["Board", "Cards"], default="Board", key="sched_view_mode"
        )

    if view_mode == "Board":
        render_board_view(week_sched)
    else:
        from model.synopsis_generator import generate_synopses_batch, _game_key
        sched_synopses = {}
        if page_week != "All" and os.getenv("ANTHROPIC_API_KEY"):
            games_for_syn = [
                {
                    "homeTeam": str(r.get("homeTeam", "")).replace(" (FCS)", ""),
                    "awayTeam": str(r.get("awayTeam", "")).replace(" (FCS)", ""),
                    "neutral": str(r.get("neutralSite", "")).lower() in ("true", "1", "yes"),
                    "predicted_spread": r.get("predicted_spread"),
                    "vegas_spread": r.get("vegas_spread"),
                    "predicted_total": r.get("predicted_total"),
                    "vegas_total": r.get("vegas_total"),
                    "week": int(page_week),
                }
                for _, r in week_sched.iterrows()
            ]
            with st.spinner("Generating AI game previews..."):
                sched_synopses = generate_synopses_batch(games_for_syn, week=int(page_week))

        for idx, row in week_sched.iterrows():
            home_clean = str(row.get("homeTeam", "")).replace(" (FCS)", "")
            away_clean = str(row.get("awayTeam", "")).replace(" (FCS)", "")
            syn_key = _game_key(home_clean, away_clean)
            render_matchup_card(row.to_dict(), idx, ratings_df, synopsis=sched_synopses.get(syn_key))

    # Download
    st.download_button(
        "📥 Export Predictions",
        data=week_sched.to_csv(index=False).encode(),
        file_name=f"{'all_weeks' if page_week == 'All' else f'week{page_week}'}_predictions.csv",
        mime="text/csv",
    )


# ── Page: Model Performance ────────────────────────────────────────────────

elif page == "📈 Model Performance":
    st.title("Model Performance Tracker")

    bt_year_options = list(range(2022, CURRENT_SEASON))
    bt_year = st.selectbox("Season", options=bt_year_options,
                           index=len(bt_year_options)-1,
                           format_func=lambda y: str(y))

    bt_df = load_backtest_results(bt_year)

    col_run, col_status = st.columns([1, 4])
    if col_run.button("▶ Run Backtest"):
        with st.spinner(f"Running {bt_year} backtest — this takes ~30 seconds..."):
            try:
                import importlib.util, sys as _sys
                _bt_path = os.path.join(os.path.dirname(__file__), "backtest.py")
                _spec = importlib.util.spec_from_file_location("backtest", _bt_path)
                _bt = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_bt)
                _bt.run_backtest(year=bt_year, save=True)
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                col_status.error(f"Backtest failed: {e}")
                import traceback
                st.text(traceback.format_exc())

    if bt_df.empty:
        st.info(f"No backtest results for {bt_year} yet. Click **▶ Run Backtest** above.")
        st.stop()

    # ── Filter controls ───────────────────────────────────────────────────
    with st.expander("Filters", expanded=False):
        grade_opts = ["All Grades"] + [g for g in ["A+", "A", "B", "C"] if g in bt_df.get("grade", pd.Series()).values]
        f_grade = st.selectbox("Min Grade", grade_opts, key="perf_grade")
        week_opts = ["All Weeks"] + sorted(bt_df["week"].dropna().unique().astype(int).tolist())
        f_week = st.selectbox("Week", week_opts, key="perf_week",
                              format_func=lambda w: f"Week {w}" if w != "All Weeks" else w)

    filtered = bt_df.copy()
    grade_order = {"A+": 0, "A": 1, "B": 2, "C": 3, None: 4}
    if f_grade != "All Grades":
        keep = [g for g, v in grade_order.items() if g is not None and v <= grade_order[f_grade]]
        filtered = filtered[filtered["grade"].isin(keep)]
    if f_week != "All Weeks":
        filtered = filtered[filtered["week"] == int(f_week)]

    # ── Top-line metrics ──────────────────────────────────────────────────
    st.divider()
    sp_df_m = filtered[filtered["bet_spread"].notna() & filtered["won_spread"].notna()]
    ou_df_m = filtered[filtered["bet_total"].notna()  & filtered["won_total"].notna()]

    sp_w = int((sp_df_m["won_spread"] == True).sum())
    sp_l = int((sp_df_m["won_spread"] == False).sum())
    sp_p = int(len(sp_df_m) - sp_w - sp_l)
    ou_w = int((ou_df_m["won_total"] == True).sum())
    ou_l = int((ou_df_m["won_total"] == False).sum())
    sp_roi = _roi(sp_w, sp_l)
    ou_roi = _roi(ou_w, ou_l)
    all_w  = sp_w + ou_w
    all_l  = sp_l + ou_l
    all_roi = _roi(all_w, all_l)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ATS Record", f"{sp_w}-{sp_l}-{sp_p}")
    c2.metric("ATS Win %",  f"{round(sp_w/(sp_w+sp_l)*100,1)}%" if sp_w+sp_l else "—")
    c3.metric("ATS ROI",    f"{sp_roi:+.1f}%", delta="break-even: -4.5%", delta_color="off")
    c4.metric("O/U Record", f"{ou_w}-{ou_l}")
    c5.metric("Combined ROI", f"{all_roi:+.1f}%",
              delta_color="normal" if all_roi > 0 else "inverse")

    st.divider()

    # ── Cumulative profit chart (week-by-week) ────────────────────────────
    if not sp_df_m.empty:
        # Aggregate by week: net units won/lost per week (bet = 1 unit, win = +100/110, loss = -1)
        sp_df_m["bet_result"] = sp_df_m["won_spread"].map({True: 100/110, False: -1.0})
        ou_df_m["bet_result"] = ou_df_m["won_total"].map({True: 100/110, False: -1.0})

        sp_weekly  = sp_df_m.groupby("week")["bet_result"].sum().reset_index().rename(columns={"bet_result": "spread_profit"})
        ou_weekly  = ou_df_m.groupby("week")["bet_result"].sum().reset_index().rename(columns={"bet_result": "total_profit"})
        wk_chart   = sp_weekly.merge(ou_weekly, on="week", how="outer").sort_values("week").fillna(0)
        wk_chart["spread_cumulative"] = wk_chart["spread_profit"].cumsum()
        wk_chart["total_cumulative"]  = wk_chart["total_profit"].cumsum()
        wk_chart["combined_cumulative"] = wk_chart["spread_profit"].add(wk_chart["total_profit"]).cumsum()

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=wk_chart["week"], y=wk_chart["spread_cumulative"],
                                     mode="lines+markers", name="Spread", line=dict(color="#4a90d9")))
        fig_cum.add_trace(go.Scatter(x=wk_chart["week"], y=wk_chart["total_cumulative"],
                                     mode="lines+markers", name="Totals", line=dict(color="#f5a623")))
        fig_cum.add_trace(go.Scatter(x=wk_chart["week"], y=wk_chart["combined_cumulative"],
                                     mode="lines+markers", name="Combined",
                                     line=dict(color="#7ed321", width=3)))
        fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_cum.update_layout(
            title=f"{bt_year} Cumulative Profit by Week (units at -110)",
            xaxis_title="Week", yaxis_title="Cumulative Units",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    # ── By-grade breakdown ────────────────────────────────────────────────
    st.subheader("Performance by Edge Grade")
    grade_rows = []
    for grade in ["A+", "A", "B", "C", "ALL"]:
        sub = sp_df_m if grade == "ALL" else sp_df_m[sp_df_m["grade"] == grade]
        if sub.empty:
            continue
        w = int((sub["won_spread"] == True).sum())
        l = int((sub["won_spread"] == False).sum())
        if w + l == 0:
            continue
        grade_rows.append({
            "Grade": grade, "Picks": w+l, "Wins": w, "Losses": l,
            "Win %": f"{w/(w+l)*100:.1f}%", "ROI": f"{_roi(w,l):+.1f}%"
        })
    if grade_rows:
        st.dataframe(pd.DataFrame(grade_rows), use_container_width=True, hide_index=True)

    # ── Week-by-week breakdown ─────────────────────────────────────────────
    st.subheader("Week-by-Week (Spread)")
    wk_rows = []
    for wk in sorted(sp_df_m["week"].dropna().unique()):
        sub = sp_df_m[sp_df_m["week"] == wk]
        w = int((sub["won_spread"] == True).sum())
        l = int((sub["won_spread"] == False).sum())
        if w + l == 0:
            continue
        wk_rows.append({
            "Week": int(wk), "Picks": w+l, "Wins": w, "Losses": l,
            "Win %": f"{w/(w+l)*100:.1f}%", "ROI": f"{_roi(w,l):+.1f}%"
        })
    if wk_rows:
        st.dataframe(pd.DataFrame(wk_rows), use_container_width=True, hide_index=True)

    # ── Week-by-week totals breakdown ──────────────────────────────────────
    st.subheader("Week-by-Week (Totals)")
    ou_wk_rows = []
    for wk in sorted(ou_df_m["week"].dropna().unique()):
        sub = ou_df_m[ou_df_m["week"] == wk]
        w = int((sub["won_total"] == True).sum())
        l = int((sub["won_total"] == False).sum())
        if w + l == 0:
            continue
        ou_wk_rows.append({
            "Week": int(wk), "Picks": w+l, "Wins": w, "Losses": l,
            "Win %": f"{w/(w+l)*100:.1f}%", "ROI": f"{_roi(w,l):+.1f}%"
        })
    if ou_wk_rows:
        st.dataframe(pd.DataFrame(ou_wk_rows), use_container_width=True, hide_index=True)

    # ── Full game log ─────────────────────────────────────────────────────
    st.subheader("Full Game Log")
    show_cols = [c for c in [
        "week", "homeTeam", "awayTeam", "home_score", "away_score",
        "model_spread", "vegas_spread", "edge_spread", "actual_margin",
        "bet_spread", "won_spread",
        "model_total", "vegas_total", "edge_total", "actual_total",
        "bet_total", "won_total", "grade", "confidence"
    ] if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True)

    st.download_button(
        "📥 Export Game Log",
        data=filtered[show_cols].to_csv(index=False).encode(),
        file_name=f"backtest_{bt_year}.csv",
        mime="text/csv",
    )


# ── Page: Update Data ──────────────────────────────────────────────────────

elif page == "🔧 Update Data":
    st.title("Update Data")
    st.markdown("""
### Weekly Workflow

**Run this each week (Sunday or Monday after games complete):**
```bash
cd ~/cfb_power_model
python update_weekly.py
```

This script will:
1. Pull the latest game results from CFBD
2. Update Elo ratings based on results
3. Pull any new SP+ ratings (if CFBD has published them)
4. Cache everything locally
5. Refresh the dashboard automatically

---
### Manual Refresh

Use the button below to force-reload data from your local CSV files.
    """)

    if st.button("🔄 Reload Data From CSV Files"):
        st.cache_data.clear()
        st.success("Cache cleared — data reloaded from CSV files.")
        st.rerun()

    st.divider()
    st.subheader("Data Status")
    files = {
        f"Power Ratings ({CURRENT_SEASON})": PREBUILT_RATINGS_PATH,
        f"Schedule ({CURRENT_SEASON})":      SCHEDULE_PATH,
        "Elo CSV":                           os.path.join(os.path.dirname(__file__), "cache", "elo_current.csv"),
        f"SP+ ({CURRENT_SEASON})":           os.path.join(os.path.dirname(__file__), "cache", f"sp_plus_{CURRENT_SEASON}.csv"),
        f"FPI ({CURRENT_SEASON})":           os.path.join(os.path.dirname(__file__), "cache", f"fpi_{CURRENT_SEASON}.csv"),
        "Performance Log":                   os.path.join(os.path.dirname(__file__), "outputs", "performance_log.csv"),
    }
    for label, path in files.items():
        exists = os.path.exists(path)
        icon = "✅" if exists else "❌"
        mtime = ""
        if exists:
            import time
            mtime = f"(last updated: {time.ctime(os.path.getmtime(path))})"
        st.write(f"{icon} **{label}** — `{os.path.basename(path)}` {mtime}")

    st.divider()
    st.subheader("Upgrade to $1/month CFBD Patreon")
    st.markdown("""
The [$1/month CFBD Patreon tier](https://www.patreon.com/collegefootballdata) adds:
- **Opponent-adjusted EPA** (much better for betting edges)
- **Predicted points** metrics
- **Success rate** by down/distance

Once you upgrade:
1. Set `CFBD_PATREON = True` in `config.py`
2. Run `python update_weekly.py --force`

The model will automatically use opponent-adjusted metrics when available.
    """)

# ── Page: Head-to-Head ────────────────────────────────────────────────────
elif page == "⚔️ Head-to-Head":
    from model.game_predictor import predict_spread, predict_total

    st.title("Head-to-Head Comparison")
    st.caption("Pick any two teams to see a full side-by-side breakdown and model prediction.")

    ratings_df = load_ratings()

    if ratings_df.empty:
        st.warning("Ratings data not available. Please run an update first.")
        st.stop()

    # Sorted team list for dropdowns
    all_teams = sorted(ratings_df["team"].dropna().unique().tolist())

    # Use separate (non-widget) state keys to track selections so swap works
    if "h2h_home" not in st.session_state:
        st.session_state["h2h_home"] = "Ohio State" if "Ohio State" in all_teams else all_teams[0]
    if "h2h_away" not in st.session_state:
        st.session_state["h2h_away"] = "Georgia" if "Georgia" in all_teams else (all_teams[1] if len(all_teams) > 1 else all_teams[0])

    col_a, col_vs, col_b = st.columns([5, 1, 5])
    with col_a:
        home_idx = all_teams.index(st.session_state["h2h_home"]) if st.session_state["h2h_home"] in all_teams else 0
        team_a = st.selectbox("🏠 Home Team", all_teams, index=home_idx)
    with col_vs:
        st.markdown("<div style='text-align:center;padding-top:2rem;font-size:1.4rem;font-weight:800;color:#00b074'>@</div>", unsafe_allow_html=True)
    with col_b:
        away_idx = all_teams.index(st.session_state["h2h_away"]) if st.session_state["h2h_away"] in all_teams else (1 if len(all_teams) > 1 else 0)
        team_b = st.selectbox("✈️ Away Team", all_teams, index=away_idx)

    # Keep state vars in sync with current widget values
    st.session_state["h2h_home"] = team_a
    st.session_state["h2h_away"] = team_b

    col_neutral, col_swap = st.columns([3, 1])
    with col_neutral:
        neutral = st.checkbox("Neutral site (no home field advantage)", value=False)
    with col_swap:
        if st.button("⇄ Swap Home/Away", use_container_width=True):
            st.session_state["h2h_home"] = team_b
            st.session_state["h2h_away"] = team_a
            st.rerun()

    if team_a == team_b:
        st.warning("Please select two different teams.")
        st.stop()

    def _get(team, col):
        r = ratings_df[ratings_df["team"] == team]
        if r.empty or col not in r.columns:
            return None
        v = r[col].values[0]
        return None if pd.isna(v) else float(v)

    # Pull all metrics for both teams
    metrics = {
        "composite":       ("⭐ Composite",    True),
        "sp_plus":         ("SP+",             True),
        "fpi":             ("FPI",             True),
        "elo":             ("Elo",             True),
        "offense.rating":  ("SP+ Offense",     True),
        "defense.rating":  ("SP+ Defense",     False),   # lower = better defense
        "returning_prod":  ("Returning Prod.", True),
        "talent":          ("Talent",          True),
    }

    a_vals = {k: _get(team_a, k) for k in metrics}
    b_vals = {k: _get(team_b, k) for k in metrics}

    # ── Prediction banner ─────────────────────────────────────────────────
    h_comp = a_vals["composite"]
    b_comp = b_vals["composite"]
    h_off  = a_vals["offense.rating"]
    b_off  = b_vals["offense.rating"]
    h_def  = a_vals["defense.rating"]
    b_def  = b_vals["defense.rating"]

    if h_comp is not None and b_comp is not None:
        # Team A treated as home unless neutral
        spread = predict_spread(h_comp, b_comp, neutral=neutral)
        total  = predict_total(h_comp, b_comp,
                               home_off_rating=h_off, away_off_rating=b_off,
                               home_def_rating=h_def, away_def_rating=b_def)

        # Win probability from spread using logistic approximation
        import math
        win_prob_a = 1 / (1 + math.exp(spread * 0.15))   # spread < 0 means A favored
        win_prob_b = 1 - win_prob_a

        if spread < 0:
            fav, dog = team_a, team_b
            fav_spread = spread
        elif spread > 0:
            fav, dog = team_b, team_a
            fav_spread = -spread
        else:
            fav, dog = None, None
            fav_spread = 0

        st.markdown("---")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Model Spread", f"{fav} {fav_spread:.1f}" if fav else "Pick'em")
        with p2:
            st.metric("Model Total", f"{total:.1f}")
        with p3:
            st.metric(f"{team_a} Win Prob", f"{win_prob_a*100:.0f}%")
        with p4:
            st.metric(f"{team_b} Win Prob", f"{win_prob_b*100:.0f}%")

        st.markdown("---")

    # ── Side-by-side metric comparison ───────────────────────────────────
    st.subheader("Metric Breakdown")

    # Build comparison rows
    rows_html = ""
    for col_key, (label, higher_better) in metrics.items():
        a_v = a_vals[col_key]
        b_v = b_vals[col_key]

        if a_v is None and b_v is None:
            continue

        a_str = f"{a_v:.1f}" if a_v is not None else "—"
        b_str = f"{b_v:.1f}" if b_v is not None else "—"

        # Determine advantage
        if a_v is not None and b_v is not None:
            a_wins = (a_v > b_v) if higher_better else (a_v < b_v)
            a_color = "#00b074" if a_wins else ("#e05" if a_v != b_v else "#888")
            b_color = "#00b074" if not a_wins else ("#e05" if a_v != b_v else "#888")
            a_bold  = "font-weight:700" if a_wins else ""
            b_bold  = "font-weight:700" if not a_wins else ""

            # Bar widths — normalize within this row
            span = abs(a_v - b_v)
            base = max(abs(a_v), abs(b_v), 0.1)
            pct  = min(span / base * 40, 40)   # max 40% extra bar for winner

            if higher_better:
                a_bar_w = 50 + (pct if a_wins else 0)
                b_bar_w = 50 + (pct if not a_wins else 0)
            else:
                a_bar_w = 50 + (pct if a_wins else 0)
                b_bar_w = 50 + (pct if not a_wins else 0)
        else:
            a_color = b_color = "#888"
            a_bold = b_bold = ""
            a_bar_w = b_bar_w = 50

        rows_html += f"""
        <div style="display:grid;grid-template-columns:1fr 120px 1fr;align-items:center;
                    margin-bottom:10px;gap:8px;">
          <div style="text-align:right;">
            <div style="background:#1a2744;border-radius:4px;height:28px;
                        width:{a_bar_w:.0f}%;margin-left:auto;display:flex;
                        align-items:center;justify-content:flex-end;padding-right:8px;">
              <span style="color:{a_color};{a_bold};font-size:0.95rem">{a_str}</span>
            </div>
          </div>
          <div style="text-align:center;font-size:0.8rem;color:#7a95b5;
                      font-weight:600;white-space:nowrap">{label}</div>
          <div>
            <div style="background:#1a2744;border-radius:4px;height:28px;
                        width:{b_bar_w:.0f}%;display:flex;
                        align-items:center;padding-left:8px;">
              <span style="color:{b_color};{b_bold};font-size:0.95rem">{b_str}</span>
            </div>
          </div>
        </div>
        """

    team_header = f"""
    <div style="display:grid;grid-template-columns:1fr 120px 1fr;gap:8px;margin-bottom:16px;">
      <div style="text-align:right;font-weight:800;font-size:1.1rem;color:#fff">{team_a}</div>
      <div></div>
      <div style="font-weight:800;font-size:1.1rem;color:#fff">{team_b}</div>
    </div>
    """
    st.html(team_header + rows_html)

    # ── Radar chart ───────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go

        radar_metrics = {
            "composite":      ("Composite",   True),
            "sp_plus":        ("SP+",         True),
            "fpi":            ("FPI",         True),
            "elo":            ("Elo",         True),
            "offense.rating": ("Offense",     True),
            "defense.rating": ("Defense",     False),
        }

        # Normalize each metric 0-100 across all teams for radar
        categories = []
        a_radar = []
        b_radar = []

        for col_key, (label, higher_better) in radar_metrics.items():
            if col_key not in ratings_df.columns:
                continue
            series = ratings_df[col_key].dropna()
            if series.empty:
                continue
            lo, hi = series.min(), series.max()
            span = hi - lo if hi != lo else 1

            def _norm_val(v, lo=lo, hi=hi, span=span, higher_better=higher_better):
                if v is None:
                    return 50
                pct = (v - lo) / span * 100
                return pct if higher_better else (100 - pct)

            a_n = _norm_val(a_vals.get(col_key))
            b_n = _norm_val(b_vals.get(col_key))
            categories.append(label)
            a_radar.append(a_n)
            b_radar.append(b_n)

        if len(categories) >= 3:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=a_radar + [a_radar[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=team_a,
                line_color="#00b074",
                fillcolor="rgba(0,176,116,0.15)",
            ))
            fig.add_trace(go.Scatterpolar(
                r=b_radar + [b_radar[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=team_b,
                line_color="#4e9af1",
                fillcolor="rgba(78,154,241,0.15)",
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="#0f1b35",
                    radialaxis=dict(visible=True, range=[0, 100],
                                   gridcolor="#1a2744", tickfont=dict(color="#7a95b5")),
                    angularaxis=dict(gridcolor="#1a2744", tickfont=dict(color="#ccc")),
                ),
                showlegend=True,
                legend=dict(font=dict(color="#ccc")),
                paper_bgcolor="#0a1628",
                plot_bgcolor="#0a1628",
                margin=dict(t=40, b=20, l=40, r=40),
                height=420,
            )
            st.subheader("Radar Chart")
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass  # radar is optional eye candy
