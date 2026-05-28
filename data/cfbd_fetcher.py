"""
cfbd_fetcher.py — pulls data from the College Football Data API.
All results are cached as CSVs in the /cache folder to avoid hitting
rate limits on repeated runs.
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CFBD_API_KEY, CURRENT_SEASON, CFBD_PATREON

BASE_URL = "https://api.collegefootballdata.com"
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"}


def _get(endpoint, params=None, cache_key=None, force_refresh=False):
    """Make a GET request, using a cached file if available."""
    if cache_key:
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if not force_refresh and os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url, headers=HEADERS, params=params or {})
    if response.status_code != 200:
        raise Exception(f"CFBD API error {response.status_code}: {response.text[:200]}")

    data = response.json()

    if cache_key:
        with open(cache_path, 'w') as f:
            json.dump(data, f)

    return data


# ── Games ──────────────────────────────────────────────────────────────────

def fetch_games(year=CURRENT_SEASON, season_type="regular", week=None, force_refresh=False):
    """Fetch all FBS game results for a season (or a specific week)."""
    params = {"year": year, "seasonType": season_type, "division": "fbs"}
    if week:
        params["week"] = week
    key = f"games_{year}_{season_type}" + (f"_week{week}" if week else "")
    data = _get("/games", params, cache_key=key, force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df


def fetch_completed_games(year=CURRENT_SEASON, force_refresh=False):
    """Return only games that have a final score."""
    df = fetch_games(year=year, force_refresh=force_refresh)
    if df.empty:
        return df
    completed = df[df["homePoints"].notna() & df["awayPoints"].notna()].copy()
    completed["margin"] = completed["homePoints"] - completed["awayPoints"]
    return completed


# ── Team stats & ratings ───────────────────────────────────────────────────

def fetch_sp_plus(year=CURRENT_SEASON, force_refresh=False):
    """Fetch SP+ ratings (offense, defense, overall)."""
    data = _get("/ratings/sp", {"year": year}, cache_key=f"sp_plus_{year}", force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df


def fetch_fpi(year=CURRENT_SEASON, force_refresh=False):
    """Fetch ESPN FPI ratings."""
    data = _get("/ratings/fpi", {"year": year}, cache_key=f"fpi_{year}", force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df


def fetch_elo(year=CURRENT_SEASON, force_refresh=False):
    """Fetch end-of-season Elo ratings."""
    data = _get("/ratings/elo", {"year": year}, cache_key=f"elo_{year}", force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df


def fetch_talent(year=CURRENT_SEASON, force_refresh=False):
    """Fetch 247Sports talent composite ratings."""
    data = _get("/talent", {"year": year}, cache_key=f"talent_{year}", force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df


def fetch_returning_production(year=CURRENT_SEASON, force_refresh=False):
    """Fetch returning production (% of production returning from prior season)."""
    data = _get("/player/returning", {"year": year}, cache_key=f"returning_{year}", force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df


def fetch_advanced_stats(year=CURRENT_SEASON, force_refresh=False):
    """
    Fetch season-level advanced stats: EPA/play, success rate, explosiveness,
    havoc rate, etc. Returns one row per team with offense/defense breakdowns.
    """
    data = _get("/stats/season/advanced",
                {"year": year, "excludeGarbageTime": True},
                cache_key=f"advanced_stats_{year}", force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df


def fetch_ppa_teams(year=CURRENT_SEASON, force_refresh=False):
    """
    Fetch opponent-adjusted PPA (Predicted Points Added) per team.
    Requires $1/month CFBD Patreon tier.

    Returns a DataFrame with columns:
      team, offense_overall, offense_passing, offense_rushing,
      defense_overall, defense_passing, defense_rushing
    where negative defense values are better (points prevented).
    """
    if not CFBD_PATREON:
        print("⚠️  PPA endpoints require CFBD Patreon tier (CFBD_PATREON=True in config).")
        return pd.DataFrame()
    data = _get("/ppa/teams",
                {"year": year, "excludeGarbageTime": True},
                cache_key=f"ppa_teams_{year}", force_refresh=force_refresh)
    df = pd.json_normalize(data)
    if df.empty:
        return df

    # Flatten nested offense/defense dicts into readable columns
    rename = {}
    for side in ("offense", "defense"):
        for metric in ("overall", "passing", "rushing", "firstDown",
                       "secondDown", "thirdDown"):
            raw = f"{side}.{metric}"
            if raw in df.columns:
                rename[raw] = f"{side}_{metric}"
    df = df.rename(columns=rename)
    return df


def fetch_advanced_box_scores(year=CURRENT_SEASON, week=None, force_refresh=False):
    """
    Fetch game-level advanced box scores with opponent-adjusted EPA breakdown.
    Requires $1/month CFBD Patreon tier.

    Returns one row per team per game with EPA, success rate, explosiveness,
    havoc, and line yards.
    """
    if not CFBD_PATREON:
        print("⚠️  Advanced box scores require CFBD Patreon tier.")
        return pd.DataFrame()
    params = {"year": year, "seasonType": "regular"}
    if week:
        params["week"] = week
    key = f"advanced_box_{year}" + (f"_week{week}" if week else "")
    data = _get("/game/box/advanced", params, cache_key=key, force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df


def fetch_epa_per_play(year=CURRENT_SEASON, force_refresh=False):
    """
    Fetch season-level EPA per play broken down by team and situation.
    Opponent-adjusted when CFBD_PATREON=True.

    Returns a cleaned DataFrame with:
      team, epa_per_play_off, epa_per_play_def,
      success_rate_off, success_rate_def,
      explosiveness_off, havoc_total
    """
    df = fetch_advanced_stats(year=year, force_refresh=force_refresh)
    if df.empty:
        return df

    # Map nested columns to flat names
    col_map = {
        "team": "team",
        "offense.epaPerPlay":          "epa_per_play_off",
        "offense.successRate":         "success_rate_off",
        "offense.explosiveness":       "explosiveness_off",
        "offense.pointsPerOpportunity":"points_per_opp_off",
        "offense.lineYards":           "line_yards_off",
        "offense.openFieldYards":      "open_field_yards_off",
        "defense.epaPerPlay":          "epa_per_play_def",
        "defense.successRate":         "success_rate_def",
        "defense.explosiveness":       "explosiveness_def",
        "defense.havoc.total":         "havoc_total",
        "defense.havoc.frontSeven":    "havoc_front7",
        "defense.havoc.db":            "havoc_db",
    }
    keep = {k: v for k, v in col_map.items() if k in df.columns}
    out = df[list(keep.keys())].rename(columns=keep)
    return out


def fetch_opponent_adjusted_stats(year=CURRENT_SEASON, force_refresh=False):
    """
    Convenience wrapper — returns PPA teams data (opponent-adjusted).
    Falls back to unadjusted advanced stats if Patreon flag is off.
    """
    if CFBD_PATREON:
        return fetch_ppa_teams(year=year, force_refresh=force_refresh)
    return fetch_advanced_stats(year=year, force_refresh=force_refresh)


# ── Schedule & venues ─────────────────────────────────────────────────────

def fetch_schedule(year=CURRENT_SEASON, force_refresh=False):
    """Fetch the full schedule including venue and location data."""
    params = {"year": year, "seasonType": "regular", "division": "fbs"}
    data = _get("/games", params, cache_key=f"schedule_{year}", force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df


def fetch_venues(force_refresh=False):
    """Fetch stadium locations (lat/lon) for weather lookups."""
    data = _get("/venues", {}, cache_key="venues", force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df


def fetch_teams(force_refresh=False):
    """Fetch FBS team metadata including conference."""
    data = _get("/teams/fbs", {}, cache_key="teams_fbs", force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df


# ── Betting lines (historical) ─────────────────────────────────────────────

def fetch_lines(year=CURRENT_SEASON, week=None, force_refresh=False):
    """
    Fetch historical betting lines from CFBD.
    Great for back-testing — covers spreads, O/U, and moneyline.
    """
    params = {"year": year, "seasonType": "regular"}
    if week:
        params["week"] = week
    key = f"lines_{year}" + (f"_week{week}" if week else "")
    data = _get("/lines", params, cache_key=key, force_refresh=force_refresh)
    df = pd.json_normalize(data)
    if df.empty:
        return df

    # Explode the nested 'lines' column to get one row per book
    if "lines" in df.columns:
        lines_exploded = df.explode("lines").reset_index(drop=True)
        line_details = pd.json_normalize(lines_exploded["lines"])
        result = pd.concat([
            lines_exploded.drop(columns=["lines"]).reset_index(drop=True),
            line_details.reset_index(drop=True)
        ], axis=1)
        return result
    return df


def fetch_consensus_lines(year=CURRENT_SEASON, week=None, force_refresh=False):
    """Return the consensus (average across books) spread and O/U per game."""
    df = fetch_lines(year=year, week=week, force_refresh=force_refresh)
    if df.empty:
        return df

    # Filter for consensus provider if available, else average numeric lines
    if "provider" in df.columns:
        consensus = df[df["provider"].str.lower().str.contains("consensus", na=False)]
        if not consensus.empty:
            return consensus

    # Fallback: average across all providers
    id_cols = ["id", "season", "week", "homeTeam", "awayTeam", "homeScore", "awayScore",
               "homeConference", "awayConference"]
    id_cols = [c for c in id_cols if c in df.columns]
    numeric_cols = ["spread", "overUnder"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    agg = df.groupby(id_cols)[numeric_cols].mean().reset_index()
    return agg


# ── Coaching ───────────────────────────────────────────────────────────────

def fetch_coaches(year=CURRENT_SEASON, force_refresh=False):
    """Fetch head coaching data — team, seasons, win/loss record."""
    data = _get("/coaches", {"year": year}, cache_key=f"coaches_{year}", force_refresh=force_refresh)
    df = pd.json_normalize(data)
    return df
