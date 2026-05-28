"""
season_projector.py — Project win totals for every team by running the model
against the full season schedule.

For each game, the predicted spread is converted to a win probability using
a normal CDF (same standard deviation the game predictor uses).  Win probs
are summed across all regular-season games to produce a projected W-L record.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

CFB_SCORE_STD = 14.0   # empirical std dev of CFB margins


def spread_to_win_prob(spread: float) -> float:
    """Home win probability from predicted spread (negative = home favored)."""
    return float(norm.cdf(-spread / CFB_SCORE_STD))


def project_season_wins(schedule_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the full schedule and model predictions, return a DataFrame with
    projected win totals per team.

    Parameters
    ----------
    schedule_df   : full season schedule (needs homeTeam, awayTeam, week,
                    homeConference, awayConference, neutralSite)
    predictions_df: output of predict_all_games — needs homeTeam, awayTeam,
                    predicted_spread

    Returns
    -------
    DataFrame with one row per team, sorted by projected_wins desc.
    Columns: team, conference, games, projected_wins, projected_losses,
             win_pct, floor_wins, ceiling_wins
    """
    # Build a conference lookup from the schedule
    conf_map = {}
    for _, row in schedule_df[["homeTeam", "homeConference"]].dropna().iterrows():
        conf_map[row["homeTeam"]] = row["homeConference"]
    for _, row in schedule_df[["awayTeam", "awayConference"]].dropna().iterrows():
        if row["awayTeam"] not in conf_map:
            conf_map[row["awayTeam"]] = row["awayConference"]

    team_probs: dict[str, list[float]] = {}

    for _, row in predictions_df.iterrows():
        home = str(row.get("homeTeam", ""))
        away = str(row.get("awayTeam", ""))
        spread = row.get("predicted_spread")

        try:
            spread_val = float(spread)
        except (TypeError, ValueError):
            spread_val = 0.0

        home_wp = spread_to_win_prob(spread_val)
        away_wp = 1.0 - home_wp

        team_probs.setdefault(home, []).append(home_wp)
        team_probs.setdefault(away, []).append(away_wp)

    rows = []
    for team, probs in team_probs.items():
        proj_wins = sum(probs)
        n = len(probs)
        proj_losses = n - proj_wins

        # Simple floor/ceiling: ±1 std dev of a Bernoulli sum
        std = float(np.sqrt(sum(p * (1 - p) for p in probs)))
        floor_wins   = max(0, round(proj_wins - std))
        ceiling_wins = min(n, round(proj_wins + std))

        rows.append({
            "team":            team,
            "conference":      conf_map.get(team, "Independent"),
            "games":           n,
            "projected_wins":  round(proj_wins, 1),
            "projected_losses": round(proj_losses, 1),
            "win_pct":         round(proj_wins / n, 3) if n > 0 else 0.0,
            "floor_wins":      floor_wins,
            "ceiling_wins":    ceiling_wins,
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("projected_wins", ascending=False)
        .reset_index(drop=True)
    )
    df.index += 1
    df.index.name = "rank"
    return df.reset_index()


def project_conference_standings(projections_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split projections into a dict keyed by conference, sorted by projected_wins.
    Excludes FCS / Independent teams from conference views.
    """
    standings = {}
    for conf, group in projections_df.groupby("conference"):
        if not conf or conf in ("Independent", "FCS"):
            continue
        standings[conf] = (
            group.sort_values("projected_wins", ascending=False)
            .reset_index(drop=True)
        )
    return standings
