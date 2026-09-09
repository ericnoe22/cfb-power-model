"""
backtest_week1_2026.py — grades the model's actual Week 1 2026 predictions
against Vegas, using genuinely pre-game (no-leakage) inputs.

Ratings are rebuilt from the exact snapshots that existed BEFORE Week 1
kicked off:
  - SP+      : cache/sp_plus_2026_week1.csv (captured Sep 1, pre-Week-1)
  - FPI      : git commit 8e5f580 (last commit before this week's live refetch)
  - Sagarin  : git commit 8e5f580 (same — Sagarin has no historical archive,
               so the pre-refetch commit is the only surviving preseason copy)
  - Elo      : 2025 final Elo, regressed to the mean (same method as production)
  - Talent / returning production: season-level, unchanged by games played

No EPA/PPA (0% weight preseason) and no luck adjustment (no games played yet)
— matches what the live model would have actually output.

Usage:
    python scripts/backtest_week1_2026.py
"""

import io
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import EDGE_THRESHOLD_SPREAD, EDGE_THRESHOLD_TOTAL
from data.cfbd_fetcher import fetch_talent, fetch_returning_production, fetch_elo
from data.team_names import normalize
from model.elo import initialize_season_elos
from model.power_rankings import build_composite_ratings
from model.game_predictor import predict_game

YEAR = 2026
WEEK = 1
PRESEASON_COMMIT = "8e5f580"   # last commit before this week's live Sagarin/SP+/FPI refetch
JUICE = -110
BREAKEVEN = 110 / 210


def _git_show(commit, path):
    out = subprocess.run(["git", "show", f"{commit}:{path}"],
                          capture_output=True, text=True, check=True).stdout
    return out


def _load_preseason_inputs():
    sp_df = pd.read_csv("cache/sp_plus_2026_week1.csv")

    fpi_df = pd.read_csv(io.StringIO(_git_show(PRESEASON_COMMIT, "cache/fpi_2026.csv")))
    fpi_df = fpi_df[["team", "fpi"]] if "fpi" in fpi_df.columns else pd.DataFrame()

    sagarin_df = pd.DataFrame(json.loads(_git_show(PRESEASON_COMMIT, "cache/sagarin_2026.json")))

    talent_df = fetch_talent(year=YEAR)
    returning_df = fetch_returning_production(year=YEAR)

    prior_elo = fetch_elo(year=YEAR - 1)
    regressed = initialize_season_elos(dict(zip(prior_elo["team"], prior_elo["elo"])))
    elo_df = pd.DataFrame([{"team": t, "elo": e} for t, e in regressed.items()])

    return sp_df, fpi_df, sagarin_df, talent_df, returning_df, elo_df


def _load_neutral_site_map():
    with open("cache/games_2026_regular.json") as f:
        games = json.load(f)
    return {
        (normalize(g["homeTeam"]), normalize(g["awayTeam"])): bool(g.get("neutralSite"))
        for g in games if g.get("week") == WEEK
    }


def roi(wins, losses):
    total = wins + losses
    if total == 0:
        return 0.0
    return round((wins / total - BREAKEVEN) * 100, 1)


def main():
    print(f"\n{'='*60}")
    print(f"  Week {WEEK} {YEAR} — Model vs. Vegas (pre-game ratings, no leakage)")
    print(f"{'='*60}\n")

    sp_df, fpi_df, sagarin_df, talent_df, returning_df, elo_df = _load_preseason_inputs()
    print(f"  Preseason inputs: SP+ {len(sp_df)} | FPI {len(fpi_df)} | "
          f"Sagarin {len(sagarin_df)} | Talent {len(talent_df)} | "
          f"Returning {len(returning_df)} | Elo {len(elo_df)}")

    ratings = build_composite_ratings(
        sp_df=sp_df, fpi_df=fpi_df, sagarin_df=sagarin_df,
        elo_df=elo_df, returning_df=returning_df, talent_df=talent_df,
        epa_df=None, games_df=None, sp_rank_prev_df=None,
        week=0, season=YEAR, apply_coaching=True,
    )
    print(f"  Pre-Week-1 composite built: {len(ratings)} teams | "
          f"Top 3: {', '.join(ratings.head(3)['team'].tolist())}\n")

    # predict_game() normalizes its home/away args via data.team_names.normalize()
    # before matching against ratings_df["team"]. ratings_df itself carries CFBD's
    # raw naming (e.g. "UL Monroe", "Massachusetts"), which differs from the
    # normalized alias for a couple of teams — normalize ratings_df's team column
    # too so those don't spuriously drop out of this backtest as "missing ratings".
    ratings = ratings.copy()
    ratings["team"] = ratings["team"].map(normalize)

    lines = pd.read_csv("cache/lines_2026.csv")
    lines = lines[(lines["week"] == WEEK) & lines["spread"].notna() & lines["homeScore"].notna()].copy()
    print(f"  Week {WEEK} games with a Vegas line + final score: {len(lines)}\n")

    neutral_map = _load_neutral_site_map()

    records = []
    for _, row in lines.iterrows():
        home = normalize(str(row["homeTeam"]))
        away = normalize(str(row["awayTeam"]))
        neutral = neutral_map.get((home, away), False)

        pred = predict_game(home, away, ratings, neutral=neutral, week=WEEK)
        if pred.get("predicted_spread") is None:
            continue

        model_spread = pred["predicted_spread"]
        model_total  = pred["predicted_total"]
        vegas_spread = float(row["spread"])
        vegas_total  = float(row["overUnder"]) if pd.notna(row.get("overUnder")) else None
        actual_home  = float(row["homeScore"])
        actual_away  = float(row["awayScore"])
        actual_margin = actual_home - actual_away
        actual_total  = actual_home + actual_away

        edge_spread = model_spread - vegas_spread
        edge_total  = (model_total - vegas_total) if vegas_total is not None else None

        home_covered = actual_margin > -vegas_spread
        push_spread  = actual_margin == -vegas_spread
        bet_home = edge_spread < -EDGE_THRESHOLD_SPREAD
        bet_away = edge_spread > EDGE_THRESHOLD_SPREAD
        won_spread = None
        if (bet_home or bet_away) and not push_spread:
            won_spread = (bet_home and home_covered) or (bet_away and not home_covered)

        won_total = None
        if edge_total is not None and vegas_total:
            bet_over  = edge_total > EDGE_THRESHOLD_TOTAL
            bet_under = edge_total < -EDGE_THRESHOLD_TOTAL
            push_total = actual_total == vegas_total
            if (bet_over or bet_under) and not push_total:
                went_over = actual_total > vegas_total
                won_total = (bet_over and went_over) or (bet_under and not went_over)

        records.append({
            "homeTeam": row["homeTeam"], "awayTeam": row["awayTeam"],
            "model_spread": round(model_spread, 1), "vegas_spread": vegas_spread,
            "actual_margin": actual_margin, "edge_spread": round(edge_spread, 1),
            "model_total": round(model_total, 1), "vegas_total": vegas_total,
            "actual_total": actual_total, "edge_total": round(edge_total, 1) if edge_total is not None else None,
            "bet_spread": "Away" if bet_away else ("Home" if bet_home else None),
            "won_spread": won_spread, "won_total": won_total,
            "model_spread_err": abs(model_spread - (-actual_margin)),
            "vegas_spread_err": abs(vegas_spread - (-actual_margin)),
        })

    df = pd.DataFrame(records)
    df.to_csv("outputs/backtest_week1_2026.csv", index=False)

    print(f"{'─'*60}")
    print(f"  Graded {len(df)} games")
    print(f"{'─'*60}\n")

    print("  ── Raw accuracy: whose spread was closer to the actual margin? ──")
    print(f"  Model MAE:  {df['model_spread_err'].mean():.2f} pts")
    print(f"  Vegas MAE:  {df['vegas_spread_err'].mean():.2f} pts")
    beat_vegas = (df["model_spread_err"] < df["vegas_spread_err"]).sum()
    tied       = (df["model_spread_err"] == df["vegas_spread_err"]).sum()
    print(f"  Model closer than Vegas: {beat_vegas}/{len(df)} games "
          f"({tied} ties)\n")

    print(f"  ── ATS record when model disagreed with Vegas by >{EDGE_THRESHOLD_SPREAD} pts ──")
    graded = df[df["won_spread"].notna()]
    wins = (graded["won_spread"] == True).sum()
    losses = (graded["won_spread"] == False).sum()
    win_pct = wins / (wins + losses) * 100 if (wins + losses) else 0
    print(f"  Bets made: {wins + losses} | Record: {wins}-{losses} ({win_pct:.1f}%) | "
          f"ROI: {roi(wins, losses):+.1f}%\n")

    print(f"  ── O/U record when model disagreed with Vegas by >{EDGE_THRESHOLD_TOTAL} pts ──")
    graded_t = df[df["won_total"].notna()]
    wins_t = (graded_t["won_total"] == True).sum()
    losses_t = (graded_t["won_total"] == False).sum()
    win_pct_t = wins_t / (wins_t + losses_t) * 100 if (wins_t + losses_t) else 0
    print(f"  Bets made: {wins_t + losses_t} | Record: {wins_t}-{losses_t} ({win_pct_t:.1f}%) | "
          f"ROI: {roi(wins_t, losses_t):+.1f}%\n")

    print("  ── Biggest misses (model spread vs. actual margin) ──")
    worst = df.reindex(df["model_spread_err"].sort_values(ascending=False).index).head(5)
    for _, r in worst.iterrows():
        print(f"  {r['awayTeam']:<22} @ {r['homeTeam']:<22} "
              f"model {r['model_spread']:+.1f}  vegas {r['vegas_spread']:+.1f}  "
              f"actual {-r['actual_margin']:+.1f}")

    print(f"\n  Full results → outputs/backtest_week1_2026.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()
