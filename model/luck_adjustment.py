"""
luck_adjustment.py — regresses teams toward their underlying win probability
rather than their raw record.

Why this matters: a team's record can diverge from what their per-game
postgame win probability says they "should" have won — close-game luck,
late-game execution in one-score games, etc. Teams running hot on close-game
luck tend to regress toward the mean; teams getting unlucky tend to bounce
back. This nudges the composite rating toward process (win probability) over
results (the record), the same idea as Pythagorean win expectation in other
sports.

Data source: CFBD /games — homePostgameWinProbability / awayPostgameWinProbability,
already pulled by fetch_completed_games(), no extra API call needed.
"""

import pandas as pd

POINTS_PER_LUCKY_WIN = 2.5   # composite-point penalty per "extra" win vs. expectation
MAX_ADJ              = 5.0   # clip — small early-season samples shouldn't swing more than this


def compute_luck_factor(games_df):
    """
    Returns DataFrame[team, actual_wins, expected_wins, games, luck, luck_adj].
    luck = actual_wins - expected_wins (positive = overperforming / lucky).
    luck_adj = composite-point adjustment to apply (negative for lucky teams,
    positive for unlucky teams), clipped to +/- MAX_ADJ.
    """
    if games_df is None or games_df.empty or \
            "homePostgameWinProbability" not in games_df.columns:
        return pd.DataFrame()

    df = games_df.dropna(subset=["homePostgameWinProbability", "awayPostgameWinProbability",
                                  "homePoints", "awayPoints"])
    if df.empty:
        return pd.DataFrame()

    home_won = df["homePoints"] > df["awayPoints"]
    home_rows = pd.DataFrame({
        "team":     df["homeTeam"],
        "win":      home_won.astype(float),
        "win_prob": df["homePostgameWinProbability"],
    })
    away_rows = pd.DataFrame({
        "team":     df["awayTeam"],
        "win":      (~home_won).astype(float),
        "win_prob": df["awayPostgameWinProbability"],
    })
    long = pd.concat([home_rows, away_rows], ignore_index=True)

    agg = long.groupby("team").agg(
        actual_wins=("win", "sum"),
        expected_wins=("win_prob", "sum"),
        games=("win", "count"),
    ).reset_index()

    agg["luck"] = agg["actual_wins"] - agg["expected_wins"]
    agg["luck_adj"] = (-POINTS_PER_LUCKY_WIN * agg["luck"]).clip(-MAX_ADJ, MAX_ADJ)
    return agg


def apply_luck_adjustment(ratings_df, games_df):
    """
    Nudge composite ratings toward process (win probability) over results.

    ratings_df: output of build_composite_ratings() — needs 'team' and 'composite'.
    Adds 'luck' (actual wins - expected wins) and 'luck_flag' columns,
    and shifts 'composite' by the resulting point adjustment.
    """
    ratings_df = ratings_df.copy()
    ratings_df["luck"] = 0.0
    ratings_df["luck_flag"] = None

    luck_df = compute_luck_factor(games_df)
    if luck_df.empty:
        return ratings_df

    ratings_df = ratings_df.drop(columns=["luck"]).merge(
        luck_df[["team", "luck", "luck_adj"]], on="team", how="left"
    )
    ratings_df["luck"] = ratings_df["luck"].fillna(0.0)
    ratings_df["luck_adj"] = ratings_df["luck_adj"].fillna(0.0)

    ratings_df["composite"] = ratings_df["composite"] + ratings_df["luck_adj"]

    flagged = ratings_df["luck_adj"].abs() > 0.05
    ratings_df.loc[flagged, "luck_flag"] = ratings_df.loc[flagged].apply(
        lambda r: f"{'Lucky' if r['luck_adj'] < 0 else 'Unlucky'} "
                  f"({r['luck']:+.2f} wins vs. expected, adj={r['luck_adj']:+.1f}pts)",
        axis=1,
    )
    ratings_df = ratings_df.drop(columns=["luck_adj"])
    return ratings_df
