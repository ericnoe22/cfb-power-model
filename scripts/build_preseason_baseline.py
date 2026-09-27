"""
build_preseason_baseline.py — one-time snapshot of each team's genuine
preseason composite rating, using TODAY's composite formula fed with
genuinely pre-Week-1 inputs (same technique as backtest_week1_2026.py).

Why not just use the earliest historical CSV in git history? The composite
formula itself has changed since preseason (Sagarin weighting bug fix, luck
adjustment, EPA weight, FCS opponent-adjustment fix, nationalAverages bug
fix...) — diffing an old-formula rating against a new-formula rating would
mix "the team got better/worse" with "we changed how we compute this",
which is not what a start-of-year power-rating tracker should show.

Run once per season, right after Week 1 (needs the git-recoverable
preseason Sagarin/FPI snapshot, which only survives for a limited time
after it gets overwritten by the live refetch).

Usage:
    python scripts/build_preseason_baseline.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from config import CURRENT_SEASON
from model.power_rankings import build_composite_ratings
from backtest_week1_2026 import _load_preseason_inputs  # noqa: E402


def main():
    sp_df, fpi_df, sagarin_df, talent_df, returning_df, elo_df = _load_preseason_inputs()

    ratings = build_composite_ratings(
        sp_df=sp_df, fpi_df=fpi_df, sagarin_df=sagarin_df,
        elo_df=elo_df, returning_df=returning_df, talent_df=talent_df,
        epa_df=None, games_df=None, sp_rank_prev_df=None,
        week=0, season=CURRENT_SEASON, apply_coaching=True,
    )

    # cache/sp_plus_2026_week1.csv predates the nationalAverages pseudo-team
    # fix in cfbd_fetcher.fetch_sp_plus() — it's read directly here, bypassing
    # that filter, so drop it explicitly.
    ratings = ratings[ratings["team"] != "nationalAverages"]
    out = ratings[["team", "composite"]].rename(columns={"composite": "preseason_composite"})
    out_path = f"cache/composite_preseason_{CURRENT_SEASON}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} teams -> {out_path}")
    print(out.sort_values("preseason_composite", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
