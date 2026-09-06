# config.py — API keys and global settings
#
# Keys are read from environment variables first (for Streamlit Cloud / CI).
# Local development: set them in a .env file or export them in your shell.
# Streamlit Cloud: add them under App Settings → Secrets as:
#   CFBD_API_KEY = "..."
#   OWLS_INSIGHT_API_KEY = "..."
#   ODDS_API_KEY = "..."
#   ANTHROPIC_API_KEY = "..."

import os

def _get(key, fallback=""):
    """Read from st.secrets if running on Streamlit Cloud, else os.environ / fallback."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, fallback)

# ── College Football Data API ──────────────────────────────────────────────
# Free tier: most endpoints. $1/month Patreon tier adds opponent-adjusted metrics.
# Get yours at: https://collegefootballdata.com/key
CFBD_API_KEY = _get("CFBD_API_KEY", "PcO2OgTyV1A4TTwl6mveP6wV2B8TYdkjcMT5r8Tm5M5B6UPns120RXdlkK9INZ3q")

# ── Owls Insight API (live game lines) ────────────────────────────────────
OWLS_INSIGHT_API_KEY = _get("OWLS_INSIGHT_API_KEY", "owlsinsight_9396243586a24df91d958db0556bc858c659c3eb5b54e8c11b5c755ec588209d")

# ── The Odds API (season win totals / futures) ─────────────────────────────
# Free tier: 500 requests/month. https://the-odds-api.com
ODDS_API_KEY = _get("ODDS_API_KEY", "456077611cb59a254b06042437814040")

# ── Model settings ─────────────────────────────────────────────────────────
CURRENT_SEASON = 2026

# Home field advantage in points (league average ~2.5)
HOME_FIELD_ADVANTAGE = 2.5

# Composite rating weights — must sum to 1.0
# Increase elo_weight as the season progresses (more games = more reliable)
RATING_WEIGHTS = {
    "sp_plus":            0.28,
    "fpi":                0.25,
    "sagarin":            0.30,
    "elo":                0.10,
    "returning_prod":     0.05,
    "talent":             0.02,
    # Opponent-adjusted EPA/PPA (CFBD Patreon). 0.00 here = preseason default
    # (no games played yet). Once build_composite_ratings() is called with a
    # week argument, it auto-ramps this weight 2.5 pts/week (carved out of
    # elo) up to a 10% cap by week 4+ — see model/power_rankings.py. Don't
    # hand-edit this value for in-season weeks; pass week= instead.
    "epa_adj":            0.00,
}

# Minimum edge (in points) to flag a game as a betting opportunity
EDGE_THRESHOLD_SPREAD  = 3.0
EDGE_THRESHOLD_TOTAL   = 3.0

# ── Patreon tier flag ──────────────────────────────────────────────────────
# $1/month CFBD Patreon tier — unlocks opponent-adjusted EPA/PPA endpoints.
CFBD_PATREON = True
