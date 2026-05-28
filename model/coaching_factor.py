"""
coaching_factor.py — applies coaching change adjustments to composite ratings.

Why this matters for betting:
  SP+ is built on last year's scheme and personnel under the OLD coach.
  When a new HC arrives, that rating becomes less predictive — the market
  often misprices teams both directions (over-hyping a big hire, or not
  fully discounting a program losing a good coach).

Two adjustments applied per team:
  1. composite_adj — small point shift based on hire quality vs prior coach SP+
  2. confidence_mult — reduces confidence score so model won't over-grade these games

Both adjustments fade linearly after week 4 as actual results replace SP+.

Data sources (in priority order):
  1. cache/coaching_changes_{year}_manual.csv  — hand-entered current-year data
  2. Auto-detected from CFBD coaches API (year vs year-1)

Coach types:
  first_time_hc      — never been a college HC before (coordinator promoted)
  experienced_hc     — proven HC moving to a new school
  interim            — interim coach, high uncertainty
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CFBD_API_KEY, CURRENT_SEASON
from data.team_names import normalize

HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"}
BASE_URL = "https://api.collegefootballdata.com"

# Confidence multipliers by coach type (applied to model confidence score)
CONFIDENCE_MULT = {
    "first_time_hc":  0.80,   # big unknown — coordinator → HC leap
    "experienced_hc": 0.88,   # scheme/culture change but known commodity
    "interim":        0.75,   # maximum uncertainty
    "unknown":        0.85,
}

# Week after which coaching adjustments fully fade (actual results take over)
FADE_THROUGH_WEEK = 6


def load_coaching_changes(year=CURRENT_SEASON, force_refresh=False):
    """
    Load coaching changes for a given season.

    Checks for manual CSV first, then falls back to CFBD auto-detection.
    Returns DataFrame with columns:
      team, new_coach, old_coach, coach_type, composite_adj, confidence_mult
    """
    manual_path = f"cache/coaching_changes_{year}_manual.csv"
    if os.path.exists(manual_path):
        df = pd.read_csv(manual_path)
        df["team"] = df["team"].map(normalize)
        df = df.dropna(subset=["team"])
        # Fill defaults for optional columns
        if "coach_type" not in df.columns:
            df["coach_type"] = "unknown"
        if "composite_adj" not in df.columns:
            df["composite_adj"] = 0.0
        df["coach_type"] = df["coach_type"].fillna("unknown")
        df["composite_adj"] = pd.to_numeric(df["composite_adj"], errors="coerce").fillna(0.0)
        df["confidence_mult"] = df["coach_type"].map(CONFIDENCE_MULT).fillna(
            CONFIDENCE_MULT["unknown"])
        return df

    # Auto-detect from CFBD by comparing current year vs prior year
    return _detect_from_cfbd(year, force_refresh)


def _detect_from_cfbd(year, force_refresh=False):
    """Pull coaches for year and year-1, detect school-level changes."""
    cache_path = f"cache/coaching_changes_{year}_auto.csv"
    if not force_refresh and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    def _fetch(y):
        try:
            r = requests.get(f"{BASE_URL}/coaches", headers=HEADERS, params={"year": y})
            if r.status_code != 200:
                return pd.DataFrame()
            rows = []
            for c in r.json():
                for s in c.get("seasons", []):
                    rows.append({
                        "coach":      f"{c['firstName']} {c['lastName']}",
                        "school":     normalize(s["school"]),
                        "year":       s["year"],
                        "sp_overall": s.get("spOverall"),
                        "hire_date":  c.get("hireDate"),
                    })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    curr = _fetch(year)
    prev = _fetch(year - 1)

    if curr.empty or prev.empty:
        return pd.DataFrame()

    merged = curr.merge(
        prev[["school", "coach", "sp_overall"]].rename(
            columns={"coach": "old_coach", "sp_overall": "old_sp"}),
        on="school", how="inner"
    )
    changed = merged[merged["coach"] != merged["old_coach"]].copy()
    changed = changed.rename(columns={"coach": "new_coach", "sp_overall": "new_sp"})

    if changed.empty:
        return pd.DataFrame()

    # Estimate composite adjustment: new coach's prior SP+ vs old coach's SP+
    # Use -2 as a default "new coach uncertainty" penalty when we can't compare
    changed["sp_delta"] = (changed["new_sp"] - changed["old_sp"]).fillna(-2.0)
    # Clip: large deltas are noise, not signal
    changed["composite_adj"] = changed["sp_delta"].clip(-4.0, 4.0).round(1)
    changed["coach_type"] = "experienced_hc"   # CFBD can't distinguish type
    changed["confidence_mult"] = CONFIDENCE_MULT["experienced_hc"]
    changed = changed.rename(columns={"school": "team"})

    out = changed[["team", "new_coach", "old_coach", "coach_type",
                   "composite_adj", "confidence_mult"]].drop_duplicates("team")
    out.to_csv(cache_path, index=False)
    return out


def apply_coaching_adjustments(ratings_df, year=CURRENT_SEASON, week=None):
    """
    Apply coaching adjustments to a composite ratings DataFrame in-place.

    ratings_df: output of build_composite_ratings() — needs 'team' and 'composite'
    week: current week number (adjustments fade after FADE_THROUGH_WEEK)

    Returns ratings_df with adjusted 'composite' and 'confidence' columns,
    plus a 'coaching_flag' column marking affected teams.
    """
    changes = load_coaching_changes(year)
    if changes.empty:
        ratings_df["coaching_flag"] = None
        return ratings_df

    # Fade factor: 1.0 in weeks 1–4, linearly fades to 0 by FADE_THROUGH_WEEK
    if week and week > FADE_THROUGH_WEEK:
        fade = 0.0
    elif week and week > 1:
        fade = max(0.0, 1.0 - (week - 1) / (FADE_THROUGH_WEEK - 1))
    else:
        fade = 1.0

    ratings_df = ratings_df.copy()
    ratings_df["coaching_flag"] = None

    for _, chg in changes.iterrows():
        team = chg["team"]
        mask = ratings_df["team"] == team
        if not mask.any():
            continue

        adj  = float(chg.get("composite_adj", 0.0)) * fade
        cmult = float(chg.get("confidence_mult", 1.0))
        # Fade confidence penalty too
        effective_mult = 1.0 - (1.0 - cmult) * fade

        ratings_df.loc[mask, "composite"] = (
            ratings_df.loc[mask, "composite"] + adj
        )
        # Apply to confidence if column exists; otherwise store for predict_game
        if "confidence" in ratings_df.columns:
            ratings_df.loc[mask, "confidence"] = (
                ratings_df.loc[mask, "confidence"] * effective_mult
            )

        flag = f"New HC: {chg.get('new_coach', '?')} ({chg.get('coach_type', '?')})"
        if abs(adj) > 0.05:
            flag += f" | adj={adj:+.1f}pts"
        ratings_df.loc[mask, "coaching_flag"] = flag

    return ratings_df


def coaching_summary(year=CURRENT_SEASON):
    """Print a readable summary of coaching changes affecting the model."""
    changes = load_coaching_changes(year)
    if changes.empty:
        print(f"No coaching changes loaded for {year}.")
        print(f"  Add them to cache/coaching_changes_{year}_manual.csv")
        return

    print(f"\n{'─'*65}")
    print(f"  Coaching Changes — {year} Season ({len(changes)} teams affected)")
    print(f"{'─'*65}")
    print(f"  {'Team':<22} {'New Coach':<22} {'Type':<16} {'Adj':>5}")
    print(f"  {'─'*60}")
    for _, r in changes.sort_values("composite_adj").iterrows():
        adj_str = f"{r['composite_adj']:+.1f}" if pd.notna(r.get("composite_adj")) else "  —"
        print(f"  {r['team']:<22} {str(r.get('new_coach','?')):<22} "
              f"{str(r.get('coach_type','?')):<16} {adj_str:>5}")
    print(f"{'─'*65}\n")
