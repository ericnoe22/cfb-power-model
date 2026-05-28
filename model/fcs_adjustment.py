"""
fcs_adjustment.py — handles FBS vs FCS game predictions.

FCS teams have no composite rating, so the standard model can't predict these
games. Instead we use 5 years of historical FBS vs FCS results to build a
tier-based lookup:

  Tier 1 Elite   composite > 20  (Alabama, Ohio State, Georgia tier)
  Tier 2 Strong  composite 10-20 (Top 15-25 programs)
  Tier 3 Good    composite 5-10  (Solid bowl teams)
  Tier 4 Average composite 0-5   (Fringe bowl eligible)
  Tier 5 Below   composite < 0   (Struggling FBS programs)

For each tier we store: avg FBS margin, avg total, ATS cover rate,
over/under rate, and sample size — separately for home and away FBS.

Usage:
    from model.fcs_adjustment import build_fcs_lookup, predict_fcs_game, is_fcs_game
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.cfbd_fetcher import _get, fetch_lines
from data.team_names import normalize
from config import HOME_FIELD_ADVANTAGE

DEFAULT_YEARS = [2020, 2021, 2022, 2023, 2024]

TIERS = [
    ("Elite",   20,  999),
    ("Strong",  10,   20),
    ("Good",     5,   10),
    ("Average",  0,    5),
    ("Below", -999,    0),
]

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cache", "fcs_lookup.json")


# ── Detection ───────────────────────────────────────────────────────────────

def is_fcs_game(home_team, away_team, ratings_df):
    """
    Return (is_fcs, fbs_team, fcs_team, fbs_is_home) if exactly one team
    is in the FBS ratings DataFrame. Returns (False, ...) if both are FBS.
    """
    teams = set(ratings_df["team"].tolist()) if not ratings_df.empty else set()
    home_fbs = home_team in teams
    away_fbs = away_team in teams
    if home_fbs and not away_fbs:
        return True, home_team, away_team, True
    if away_fbs and not home_fbs:
        return True, away_team, home_team, False
    return False, None, None, None


# ── Data loading ────────────────────────────────────────────────────────────

def _load_fbs_vs_fcs_games(years=None):
    """Pull FBS vs FCS game results across multiple seasons."""
    if years is None:
        years = DEFAULT_YEARS

    frames = []
    for year in years:
        try:
            data = _get("/games",
                        {"year": year, "seasonType": "regular", "division": "fbs"},
                        cache_key=f"games_{year}_all")
            df = pd.json_normalize(data)
            if df.empty:
                continue
            df["season"] = year
            fbs_vs_fcs = df[
                ((df.get("homeClassification", "fbs") == "fbs") &
                 (df.get("awayClassification", "fbs") == "fcs")) |
                ((df.get("homeClassification", "fbs") == "fcs") &
                 (df.get("awayClassification", "fbs") == "fbs"))
            ].copy()
            frames.append(fbs_vs_fcs)
        except Exception as e:
            print(f"  Could not load {year} FBS vs FCS games: {e}")

    if not frames:
        return pd.DataFrame()

    all_games = pd.concat(frames, ignore_index=True)
    all_games = all_games[
        all_games["homePoints"].notna() & all_games["awayPoints"].notna()
    ].copy()

    # Normalize team names
    all_games["homeTeam"] = all_games["homeTeam"].apply(lambda t: normalize(str(t)))
    all_games["awayTeam"]  = all_games["awayTeam"].apply(lambda t: normalize(str(t)))

    # Add FBS-perspective columns
    all_games["fbs_is_home"] = all_games["homeClassification"] == "fbs"
    all_games["fbs_team"]    = np.where(
        all_games["fbs_is_home"], all_games["homeTeam"], all_games["awayTeam"]
    )
    all_games["fbs_score"]   = np.where(
        all_games["fbs_is_home"],
        all_games["homePoints"].astype(float),
        all_games["awayPoints"].astype(float)
    )
    all_games["fcs_score"]   = np.where(
        all_games["fbs_is_home"],
        all_games["awayPoints"].astype(float),
        all_games["homePoints"].astype(float)
    )
    all_games["fbs_margin"]  = all_games["fbs_score"] - all_games["fcs_score"]
    all_games["actual_total"] = all_games["fbs_score"] + all_games["fcs_score"]
    return all_games


def _load_fbs_vs_fcs_lines(years=None):
    """Pull lines for FBS vs FCS games (subset of all lines)."""
    if years is None:
        years = DEFAULT_YEARS

    frames = []
    for year in years:
        try:
            df = fetch_lines(year=year)
            if df.empty or "homeClassification" not in df.columns:
                continue
            fcs_lines = df[
                ((df["homeClassification"] == "fbs") & (df["awayClassification"] == "fcs")) |
                ((df["homeClassification"] == "fcs") & (df["awayClassification"] == "fbs"))
            ].copy()
            if "provider" in fcs_lines.columns:
                dk = fcs_lines[fcs_lines["provider"] == "DraftKings"]
                fcs_lines = dk if not dk.empty else fcs_lines
            fcs_lines = fcs_lines.drop_duplicates(subset=["homeTeam", "awayTeam", "season", "week"])
            frames.append(fcs_lines)
        except Exception as e:
            print(f"  Could not load {year} FCS lines: {e}")

    if not frames:
        return pd.DataFrame()

    all_lines = pd.concat(frames, ignore_index=True)
    all_lines["homeTeam"] = all_lines["homeTeam"].apply(lambda t: normalize(str(t)))
    all_lines["awayTeam"]  = all_lines["awayTeam"].apply(lambda t: normalize(str(t)))
    return all_lines


# ── Tier assignment ─────────────────────────────────────────────────────────

def _get_tier(composite):
    """Return tier name for a given composite rating."""
    for name, lo, hi in TIERS:
        if lo <= composite < hi:
            return name
    return "Average"


def _assign_tiers(games_df, sp_by_year):
    """
    Add a 'tier' column to games_df based on the FBS team's SP+ rating
    in the season of the game.
    sp_by_year: dict {year: DataFrame with columns team + rating}
    """
    def lookup_tier(row):
        year   = int(row.get("season", 0))
        team   = row["fbs_team"]
        sp_df  = sp_by_year.get(year, pd.DataFrame())
        if sp_df.empty:
            return "Average"
        match = sp_df[sp_df["team"] == team]
        if match.empty:
            return "Average"
        rating = float(match.iloc[0].get("rating", match.iloc[0].get("sp_plus", 0)))
        return _get_tier(rating)

    games_df = games_df.copy()
    games_df["tier"] = games_df.apply(lookup_tier, axis=1)
    return games_df


# ── Lookup builder ──────────────────────────────────────────────────────────

def build_fcs_lookup(years=None, force_refresh=False):
    """
    Build and cache the FBS-vs-FCS historical lookup table.
    Returns dict keyed by (tier, fbs_is_home):
      {
        "Elite_home":  {avg_margin, std_margin, avg_total, cover_rate,
                        over_rate, n_games, n_with_lines},
        "Elite_away":  {...},
        ...
      }
    """
    if years is None:
        years = DEFAULT_YEARS

    if not force_refresh and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)

    print("Building FBS vs FCS historical lookup...")
    games = _load_fbs_vs_fcs_games(years)
    if games.empty:
        return {}

    # Load SP+ for tier assignment
    from data.cfbd_fetcher import fetch_sp_plus
    sp_by_year = {}
    for year in years:
        try:
            sp = fetch_sp_plus(year=year)
            if not sp.empty:
                if "sp_plus" not in sp.columns and "rating" in sp.columns:
                    sp = sp.rename(columns={"rating": "sp_plus"})
                sp_by_year[year] = sp
        except Exception:
            pass

    games = _assign_tiers(games, sp_by_year)

    # Load lines and merge
    lines = _load_fbs_vs_fcs_lines(years)
    if not lines.empty:
        lines["fbs_is_home_ln"] = lines["homeClassification"] == "fbs"
        lines["fbs_team_ln"]    = np.where(
            lines["fbs_is_home_ln"], lines["homeTeam"], lines["awayTeam"]
        )
        # Spread from FBS perspective (FBS favored → negative when home)
        # CFBD spread is from home perspective; positive = home is underdog
        lines["fbs_spread"] = np.where(
            lines["fbs_is_home_ln"],
            lines["spread"].astype(float),      # home FBS: use as-is
            -lines["spread"].astype(float)       # away FBS: flip sign
        )
        lines["covered"] = np.where(
            lines["fbs_is_home_ln"],
            (lines["homeScore"].astype(float) - lines["awayScore"].astype(float)) > -lines["spread"].astype(float),
            (lines["awayScore"].astype(float) - lines["homeScore"].astype(float)) > lines["spread"].astype(float)
        )
        lines["went_over"] = (
            lines["homeScore"].astype(float) + lines["awayScore"].astype(float)
        ) > lines["overUnder"].astype(float)

        merge_key = ["homeTeam", "awayTeam", "season", "week"]
        avail_keys = [k for k in merge_key if k in lines.columns and k in games.columns]
        games = games.merge(
            lines[avail_keys + ["fbs_spread", "covered", "went_over", "overUnder"]],
            on=avail_keys, how="left"
        )

    # Build lookup
    lookup = {}
    for tier, _, _ in TIERS:
        for side, is_home in [("home", True), ("away", False)]:
            key = f"{tier}_{side}"
            subset = games[
                (games["tier"] == tier) & (games["fbs_is_home"] == is_home)
            ]
            if subset.empty:
                continue

            n = len(subset)
            margins    = subset["fbs_margin"].dropna()
            totals     = subset["actual_total"].dropna()
            with_lines = subset["fbs_spread"].notna()
            n_lines    = int(with_lines.sum())

            entry = {
                "n_games":      n,
                "n_with_lines": n_lines,
                "avg_margin":   round(float(margins.mean()), 1) if len(margins) else 25.0,
                "std_margin":   round(float(margins.std()),  1) if len(margins) > 1 else 12.0,
                "avg_total":    round(float(totals.mean()),  1) if len(totals) else 45.0,
                "std_total":    round(float(totals.std()),   1) if len(totals) > 1 else 10.0,
            }

            if n_lines >= 5:
                covers  = subset.loc[with_lines, "covered"].dropna()
                overs   = subset.loc[with_lines, "went_over"].dropna()
                entry["cover_rate"] = round(float(covers.mean()), 3) if len(covers) else 0.5
                entry["over_rate"]  = round(float(overs.mean()),  3) if len(overs) else 0.5
                entry["avg_line"]   = round(float(subset.loc[with_lines, "fbs_spread"].mean()), 1)
            else:
                entry["cover_rate"] = None
                entry["over_rate"]  = None
                entry["avg_line"]   = None

            lookup[key] = entry

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(lookup, f, indent=2)

    print(f"  FCS lookup built: {len(lookup)} tier/side combinations")
    for key, v in sorted(lookup.items()):
        cover_str = f"cover {v['cover_rate']:.0%}" if v["cover_rate"] else "no line data"
        over_str  = f"over {v['over_rate']:.0%}" if v["over_rate"] else ""
        print(f"    {key:<16} n={v['n_games']:3d}  "
              f"avg margin {v['avg_margin']:+.1f}  "
              f"avg total {v['avg_total']:.1f}  "
              f"{cover_str}  {over_str}")

    return lookup


# ── Prediction ──────────────────────────────────────────────────────────────

def predict_fcs_game(fbs_team, fbs_is_home, fbs_composite, lookup):
    """
    Return predicted spread (home perspective) and total for an FBS vs FCS game.

    fbs_composite: the FBS team's current composite rating
    lookup: output of build_fcs_lookup()

    Returns dict with keys: predicted_spread, predicted_total, confidence,
                             fcs_note, tier, cover_rate, over_rate
    """
    tier    = _get_tier(fbs_composite)
    side    = "home" if fbs_is_home else "away"
    key     = f"{tier}_{side}"
    entry   = lookup.get(key)

    if not entry:
        # Fallback: use average tier
        key   = f"Average_{side}"
        entry = lookup.get(key, {})

    avg_margin = float(entry.get("avg_margin", 20.0))
    avg_total  = float(entry.get("avg_total",  45.0))
    cover_rate = entry.get("cover_rate")
    over_rate  = entry.get("over_rate")
    n_games    = entry.get("n_games", 0)
    n_lines    = entry.get("n_with_lines", 0)
    avg_line   = entry.get("avg_line")

    # Convert FBS margin to home-perspective spread
    # FBS home: wins by avg_margin → spread = -avg_margin
    # FBS away: wins by avg_margin → spread = +avg_margin (positive = away fav in CFBD convention)
    if fbs_is_home:
        predicted_spread = round(-avg_margin, 1)
    else:
        predicted_spread = round(avg_margin, 1)

    # Confidence: lower for smaller sample and no-line data
    base_conf = min(0.80, 0.50 + (n_games / 100) * 0.30)
    confidence = round(base_conf, 2)

    # Build informative note
    cover_note = f"cover {cover_rate:.0%} ({n_lines} lines)" if cover_rate else "no line history"
    over_note  = f"O/U {over_rate:.0%} over" if over_rate else ""
    note = (
        f"FCS opponent ({tier} FBS tier) | "
        f"{n_games} historical games | avg margin +{avg_margin:.0f} | "
        f"{cover_note}" + (f" | {over_note}" if over_note else "")
    )

    return {
        "predicted_spread": predicted_spread,
        "predicted_total":  round(avg_total, 1),
        "confidence":       confidence,
        "fcs_note":         note,
        "tier":             tier,
        "cover_rate":       cover_rate,
        "over_rate":        over_rate,
        "avg_line":         avg_line,
    }
