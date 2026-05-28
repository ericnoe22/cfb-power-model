"""
situational_factors.py — historical tendency adjustments for spread and total predictions.

Two primary use cases:
1. Home underdog and big-favorite ATS situational cover rates
2. Team/matchup historical over/under tendencies baked into predicted totals
   (e.g., Iowa-Iowa State consistently goes under → pull predicted total down)

Data source: CFBD historical lines (fetch_lines) over the last N seasons.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.cfbd_fetcher import fetch_lines
from data.team_names import normalize

DEFAULT_YEARS    = [2020, 2021, 2022, 2023, 2024]
MIN_MATCHUP_GAMES = 4    # min games needed for matchup-specific total adj
MIN_TEAM_GAMES    = 8    # min games needed for team-level tendency
PRIOR_WEIGHT      = 4    # Bayesian smoothing games at 50%
MAX_TOTAL_ADJ     = 6.0  # max point adjustment to predicted total
MAX_TEAM_ADJ      = 3.0  # max point adjustment from single team tendency


# ── Data loading ────────────────────────────────────────────────────────────

def load_historical_data(years=None, force_refresh=False):
    """
    Load and combine historical lines + results across multiple seasons.
    Returns one row per FBS game with spread, O/U, scores, and derived columns.
    """
    if years is None:
        years = DEFAULT_YEARS

    cache_path = os.path.join(
        os.path.dirname(__file__), "..", "cache",
        f"historical_lines_{min(years)}_{max(years)}.csv"
    )
    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        # Rebuild matchup_key if missing (older cache)
        if "matchup_key" not in df.columns:
            df["matchup_key"] = df.apply(
                lambda r: "|".join(sorted([str(r["homeTeam"]), str(r["awayTeam"])])), axis=1
            )
        return df

    frames = []
    for year in years:
        try:
            df = fetch_lines(year=year)
            if not df.empty:
                frames.append(df)
                print(f"  Loaded {year}: {len(df)} rows")
        except Exception as e:
            print(f"  Could not fetch {year} lines: {e}")

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)

    # Filter to FBS vs FBS with a spread and both scores
    home_fbs = all_df.get("homeClassification", pd.Series("fbs", index=all_df.index)) == "fbs"
    away_fbs = all_df.get("awayClassification", pd.Series("fbs", index=all_df.index)) == "fbs"
    fbs = all_df[
        home_fbs & away_fbs &
        all_df["spread"].notna() &
        all_df["homeScore"].notna() &
        all_df["awayScore"].notna()
    ].copy()

    # Prefer DraftKings; one row per game
    if "provider" in fbs.columns:
        dk = fbs[fbs["provider"] == "DraftKings"].drop_duplicates(
            subset=["homeTeam", "awayTeam", "season", "week"])
        dk_keys = set(zip(dk["homeTeam"], dk["awayTeam"],
                          dk.get("season", [0]*len(dk)), dk["week"]))
        other = fbs[fbs["provider"] != "DraftKings"].drop_duplicates(
            subset=["homeTeam", "awayTeam", "season", "week"])
        other = other[~other.apply(
            lambda r: (r["homeTeam"], r["awayTeam"],
                       r.get("season", 0), r["week"]) in dk_keys, axis=1)]
        fbs = pd.concat([dk, other], ignore_index=True)
    else:
        fbs = fbs.drop_duplicates(subset=["homeTeam", "awayTeam", "season", "week"])

    # Normalize team names to match model convention
    fbs["homeTeam"] = fbs["homeTeam"].apply(lambda t: normalize(str(t)))
    fbs["awayTeam"] = fbs["awayTeam"].apply(lambda t: normalize(str(t)))

    # Derived columns
    fbs["actual_margin"] = fbs["homeScore"].astype(float) - fbs["awayScore"].astype(float)
    fbs["actual_total"]  = fbs["homeScore"].astype(float) + fbs["awayScore"].astype(float)
    fbs["home_covered"]  = fbs["actual_margin"] > -fbs["spread"].astype(float)
    fbs["push_spread"]   = fbs["actual_margin"] == -fbs["spread"].astype(float)
    fbs["went_over"]     = fbs["actual_total"] > fbs["overUnder"].astype(float)
    fbs["went_under"]    = fbs["actual_total"] < fbs["overUnder"].astype(float)
    fbs["push_total"]    = fbs["actual_total"] == fbs["overUnder"].astype(float)
    fbs["matchup_key"]   = fbs.apply(
        lambda r: "|".join(sorted([str(r["homeTeam"]), str(r["awayTeam"])])), axis=1
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    fbs.to_csv(cache_path, index=False)
    print(f"  Cached {len(fbs)} historical FBS games to {cache_path}")
    return fbs


# ── Tendency builders ───────────────────────────────────────────────────────

def build_matchup_ou_tendencies(hist_df):
    """
    Per matchup-pair O/U stats (e.g., Iowa vs Iowa State always goes under).
    Returns DataFrame: matchup_key, teams, n_games, under_pct, over_pct, total_adj_pts.
    """
    if hist_df.empty:
        return pd.DataFrame()

    valid = hist_df[~hist_df["push_total"]].copy()
    rows = []
    for key, grp in valid.groupby("matchup_key"):
        n = len(grp)
        if n < MIN_MATCHUP_GAMES:
            continue
        under_n = int(grp["went_under"].sum())
        over_n  = int(grp["went_over"].sum())
        under_rate = under_n / n
        adj = _total_adj_from_rate(under_rate, n, min_games=MIN_MATCHUP_GAMES,
                                   max_adj=MAX_TOTAL_ADJ)
        teams = key.split("|")
        rows.append({
            "matchup_key":   key,
            "team_a":        teams[0],
            "team_b":        teams[1],
            "n_games":       n,
            "under_n":       under_n,
            "over_n":        over_n,
            "under_pct":     round(under_rate * 100, 1),
            "over_pct":      round((1 - under_rate) * 100, 1),
            "total_adj_pts": adj,  # positive = adjust total DOWN (toward under)
        })
    return pd.DataFrame(rows)


def build_team_ou_tendencies(hist_df):
    """
    Per-team overall O/U tendency combining home and away games.
    Returns DataFrame: team, n_games, under_pct, over_pct, team_total_adj.
    """
    if hist_df.empty:
        return pd.DataFrame()

    valid = hist_df[~hist_df["push_total"]].copy()
    home = valid[["homeTeam", "went_under", "went_over"]].rename(columns={"homeTeam": "team"})
    away = valid[["awayTeam", "went_under", "went_over"]].rename(columns={"awayTeam": "team"})
    stacked = pd.concat([home, away], ignore_index=True)

    rows = []
    for team, grp in stacked.groupby("team"):
        n = len(grp)
        if n < MIN_TEAM_GAMES:
            continue
        under_n    = int(grp["went_under"].sum())
        under_rate = under_n / n
        adj = _total_adj_from_rate(under_rate, n, min_games=MIN_TEAM_GAMES,
                                   max_adj=MAX_TEAM_ADJ)
        rows.append({
            "team":           team,
            "n_games":        n,
            "under_n":        under_n,
            "under_pct":      round(under_rate * 100, 1),
            "over_pct":       round((1 - under_rate) * 100, 1),
            "team_total_adj": adj,
        })
    return pd.DataFrame(rows)


def build_team_ats_tendencies(hist_df):
    """
    Per-team ATS cover rates by situation.
    Situations tracked:
      - overall home / away cover rate
      - as home underdog (spread > 0)
      - as away underdog (spread < 0 from home = away is giving points... wait no)
      - as big favorite (|spread| >= 14, home or away)
    Returns DataFrame with one row per team+side.
    """
    if hist_df.empty:
        return pd.DataFrame()

    valid = hist_df[~hist_df["push_spread"]].copy()
    rows = []

    # Home-side stats
    for team, grp in valid.groupby("homeTeam"):
        n = len(grp)
        if n < MIN_TEAM_GAMES:
            continue
        covered = grp["home_covered"]

        # Overall home cover
        overall_pct = round(covered.mean() * 100, 1)

        # As home underdog (spread > 0 means home is getting points)
        h_dog = grp[grp["spread"] > 0]
        h_dog_pct = round(h_dog["home_covered"].mean() * 100, 1) if len(h_dog) >= 4 else None

        # As big home favorite (giving 14+)
        h_bigfav = grp[grp["spread"] <= -14]
        h_bigfav_pct = round(h_bigfav["home_covered"].mean() * 100, 1) if len(h_bigfav) >= 4 else None

        rows.append({
            "team": team, "side": "home", "n_games": n,
            "overall_cover_pct": overall_pct,
            "dog_cover_pct":     h_dog_pct,    "dog_n": len(h_dog),
            "bigfav_cover_pct":  h_bigfav_pct, "bigfav_n": len(h_bigfav),
        })

    # Away-side stats
    for team, grp in valid.groupby("awayTeam"):
        n = len(grp)
        if n < MIN_TEAM_GAMES:
            continue
        away_covered = ~grp["home_covered"]

        overall_pct = round(away_covered.mean() * 100, 1)

        # As away underdog (home spread is negative = home favored = away is dog)
        a_dog = grp[grp["spread"] < 0]
        a_dog_covered = ~a_dog["home_covered"]
        a_dog_pct = round(a_dog_covered.mean() * 100, 1) if len(a_dog) >= 4 else None

        # As big away favorite (home spread >= +14 means home is big dog → away is big fav)
        a_bigfav = grp[grp["spread"] >= 14]
        a_bigfav_covered = ~a_bigfav["home_covered"]
        a_bigfav_pct = round(a_bigfav_covered.mean() * 100, 1) if len(a_bigfav) >= 4 else None

        rows.append({
            "team": team, "side": "away", "n_games": n,
            "overall_cover_pct": overall_pct,
            "dog_cover_pct":     a_dog_pct,    "dog_n": len(a_dog),
            "bigfav_cover_pct":  a_bigfav_pct, "bigfav_n": len(a_bigfav),
        })

    return pd.DataFrame(rows)


# ── Application functions ───────────────────────────────────────────────────

def apply_total_adjustment(home_team, away_team, matchup_ou, team_ou):
    """
    Compute how much to adjust the model's predicted total for a specific game.

    Positive return = adjust total DOWN (historical tendency toward under).
    Negative return = adjust total UP (historical tendency toward over).

    Returns (adj_points: float, note: str or None)
    """
    key = "|".join(sorted([str(home_team), str(away_team)]))
    adj = 0.0
    notes = []

    # 1. Matchup-specific tendency (60% weight — most reliable signal)
    if not matchup_ou.empty:
        row = matchup_ou[matchup_ou["matchup_key"] == key]
        if not row.empty:
            r = row.iloc[0]
            matchup_adj = float(r["total_adj_pts"])
            if abs(matchup_adj) > 0:
                adj += matchup_adj * 0.6
                direction = "under" if matchup_adj > 0 else "over"
                notes.append(
                    f"{home_team}/{away_team} go {direction} "
                    f"{r['under_n'] if direction == 'under' else r['over_n']}/"
                    f"{r['n_games']} historically ({r['under_pct'] if direction == 'under' else r['over_pct']}%)"
                )

    # 2. Individual team tendencies (40% weight split between both teams)
    if not team_ou.empty:
        team_adjs = []
        for team in [home_team, away_team]:
            row = team_ou[team_ou["team"] == team]
            if not row.empty:
                r = row.iloc[0]
                t_adj = float(r["team_total_adj"])
                if abs(t_adj) > 0:
                    team_adjs.append(t_adj)
                    direction = "under" if t_adj > 0 else "over"
                    notes.append(
                        f"{team} tends {direction} ({r['under_pct']}% under, {r['n_games']} games)"
                    )
        if team_adjs:
            adj += float(np.mean(team_adjs)) * 0.4

    adj = round(float(np.clip(adj, -MAX_TOTAL_ADJ, MAX_TOTAL_ADJ)), 1)
    return adj, (" | ".join(notes) if notes else None)


def get_ats_situational_note(home_team, away_team, vegas_spread, bet_side, team_ats):
    """
    Return a situational ATS note for a bet.
    bet_side: "Home" or "Away"
    vegas_spread: CFBD convention (negative = home favored)
    """
    if team_ats.empty or pd.isna(vegas_spread):
        return None

    notes = []
    is_home_dog  = float(vegas_spread) > 0     # home getting points
    is_home_bigfav = float(vegas_spread) <= -14
    is_away_bigfav = float(vegas_spread) >= 14

    # Check home team's situational stats
    hr = team_ats[(team_ats["team"] == home_team) & (team_ats["side"] == "home")]
    if not hr.empty:
        r = hr.iloc[0]
        if bet_side == "Home":
            if is_home_dog and r["dog_cover_pct"] is not None:
                pct, n = r["dog_cover_pct"], r["dog_n"]
                if pct >= 60:
                    notes.append(f"{home_team} covers {pct:.0f}% as home dog ({n} games)")
                elif pct <= 40:
                    notes.append(f"⚠ {home_team} covers only {pct:.0f}% as home dog ({n} games)")
            if is_home_bigfav and r["bigfav_cover_pct"] is not None:
                pct, n = r["bigfav_cover_pct"], r["bigfav_n"]
                if pct <= 45:
                    notes.append(f"⚠ {home_team} covers {pct:.0f}% as big home fav ({n} games)")

    # Check away team's situational stats
    ar = team_ats[(team_ats["team"] == away_team) & (team_ats["side"] == "away")]
    if not ar.empty:
        r = ar.iloc[0]
        if bet_side == "Away":
            if not is_home_dog and r["dog_cover_pct"] is not None:
                # away dog = home is favored = spread < 0
                pct, n = r["dog_cover_pct"], r["dog_n"]
                if pct >= 60:
                    notes.append(f"{away_team} covers {pct:.0f}% as away dog ({n} games)")
                elif pct <= 40:
                    notes.append(f"⚠ {away_team} covers only {pct:.0f}% as away dog ({n} games)")
            if is_away_bigfav and r["bigfav_cover_pct"] is not None:
                pct, n = r["bigfav_cover_pct"], r["bigfav_n"]
                if pct <= 45:
                    notes.append(f"⚠ {away_team} covers {pct:.0f}% as big away fav ({n} games)")

    return " | ".join(notes) if notes else None


def apply_adjustments_to_predictions(pred_df, matchup_ou, team_ou, team_ats):
    """
    Apply all situational adjustments to a predictions DataFrame in-place.
    Adds columns:
      - total_hist_adj:    points to add/subtract from predicted_total
      - predicted_total_adj: adjusted total (use this for edge finding)
      - hist_total_note:  human-readable explanation
    """
    if pred_df.empty:
        return pred_df

    adjs, notes = [], []
    for _, row in pred_df.iterrows():
        adj, note = apply_total_adjustment(
            row.get("homeTeam", ""), row.get("awayTeam", ""),
            matchup_ou, team_ou
        )
        adjs.append(adj)
        notes.append(note)

    pred_df = pred_df.copy()
    pred_df["total_hist_adj"] = adjs
    pred_df["hist_total_note"] = notes

    if "predicted_total" in pred_df.columns:
        # Subtract adjustment: positive adj = under tendency = lower the total
        pred_df["predicted_total_adj"] = (
            pred_df["predicted_total"] - pred_df["total_hist_adj"]
        ).round(1)
    return pred_df


# ── Internal helpers ────────────────────────────────────────────────────────

def _total_adj_from_rate(under_rate, n_games, min_games=MIN_MATCHUP_GAMES,
                         max_adj=MAX_TOTAL_ADJ):
    """
    Convert an under rate + sample size into a point adjustment.
    Applies Bayesian smoothing toward 50%.
    Positive = adjust total down; negative = adjust total up.
    """
    if n_games < min_games:
        return 0.0
    smoothed = (under_rate * n_games + PRIOR_WEIGHT * 0.5) / (n_games + PRIOR_WEIGHT)
    # Scale: 0.3 deviation from 50% → max_adj points
    adj = (smoothed - 0.5) * (max_adj / 0.3)
    return round(float(np.clip(adj, -max_adj, max_adj)), 1)
