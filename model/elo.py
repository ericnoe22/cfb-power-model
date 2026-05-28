"""
elo.py — Elo rating system for college football.

Elo tracks form: ratings update after every game based on result vs. expectation.
Key parameters:
  K = how fast ratings change (higher = more reactive to recent games)
  HFA = home field advantage in Elo points (roughly 65–70 for CFB)
  MOV_MULTIPLIER = margin of victory multiplier (prevents running up the score)
"""

import pandas as pd
import numpy as np

# Starting rating for new teams / season initialization
DEFAULT_ELO = 1500

# How much ratings move per game (higher = reacts faster to results)
K_FACTOR = 20

# Home field advantage in Elo points (~65-70 for CFB, equivalent to ~2.5 pts)
ELO_HFA = 65

# Carry-over from prior season (regression to mean: 0.7 = 70% of prior rating carries)
SEASON_CARRY_OVER = 0.70


def win_probability(elo_a, elo_b):
    """Expected win probability for team A vs team B (before HFA adjustment)."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def mov_multiplier(margin, elo_diff):
    """
    Margin of victory multiplier — diminishing returns above ~14 pts.
    Borrowed from FiveThirtyEight NFL model, adjusted for CFB.
    """
    if margin <= 0:
        return 1.0
    return np.log(abs(margin) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))


def update_elo(home_elo, away_elo, home_score, away_score, neutral=False):
    """
    Update Elo ratings after a single game.
    Returns (new_home_elo, new_away_elo).
    """
    hfa = 0 if neutral else ELO_HFA
    home_elo_adj = home_elo + hfa

    expected_home = win_probability(home_elo_adj, away_elo)
    actual_home = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)

    margin = home_score - away_score
    elo_diff = home_elo_adj - away_elo
    k = K_FACTOR * mov_multiplier(margin, elo_diff)

    delta = k * (actual_home - expected_home)
    return round(home_elo + delta, 1), round(away_elo - delta, 1)


def initialize_season_elos(prior_elos, default=DEFAULT_ELO, carry_over=SEASON_CARRY_OVER):
    """
    Regress Elo ratings toward the mean at the start of a new season.
    prior_elos: dict of {team: elo}
    Returns dict of {team: new_season_elo}
    """
    return {
        team: round(elo * carry_over + default * (1 - carry_over), 1)
        for team, elo in prior_elos.items()
    }


def run_season_elos(games_df, initial_elos=None):
    """
    Process a full season's game results to produce final Elo ratings.

    games_df: DataFrame with columns [homeTeam, awayTeam, homePoints, awayPoints,
                                       neutralSite, week]
    initial_elos: dict {team: elo} — defaults to 1500 for all teams

    Returns:
        ratings_df: DataFrame with [team, elo] (end-of-season)
        history_df: DataFrame with per-game Elo changes (for trend charts)
    """
    elos = dict(initial_elos) if initial_elos else {}
    history = []

    games = games_df.sort_values("week").copy()
    games = games[games["homePoints"].notna() & games["awayPoints"].notna()]

    for _, row in games.iterrows():
        home = row["homeTeam"]
        away = row["awayTeam"]
        neutral = row.get("neutralSite", False)

        home_elo = elos.get(home, DEFAULT_ELO)
        away_elo = elos.get(away, DEFAULT_ELO)

        new_home, new_away = update_elo(
            home_elo, away_elo,
            row["homePoints"], row["awayPoints"],
            neutral=neutral
        )

        history.append({
            "week": row.get("week"),
            "homeTeam": home, "awayTeam": away,
            "homePoints": row["homePoints"], "awayPoints": row["awayPoints"],
            "pre_home_elo": home_elo, "pre_away_elo": away_elo,
            "post_home_elo": new_home, "post_away_elo": new_away,
        })

        elos[home] = new_home
        elos[away] = new_away

    ratings_df = pd.DataFrame([
        {"team": team, "elo": elo} for team, elo in elos.items()
    ]).sort_values("elo", ascending=False).reset_index(drop=True)

    history_df = pd.DataFrame(history)
    return ratings_df, history_df


def elo_to_spread(home_elo, away_elo, neutral=False, points_per_100_elo=3.0):
    """
    Convert Elo difference to a predicted point spread.
    Default: 100 Elo points ≈ 3 points on the spread.
    Negative means home team favored.
    """
    hfa_pts = 0 if neutral else 2.5
    diff = (home_elo - away_elo) / 100 * points_per_100_elo
    return round(-(diff + hfa_pts), 1)   # negative = home favored (standard spread format)
