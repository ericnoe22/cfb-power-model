"""
week2_edge_board.py — generates the model-vs-Vegas edge board for Week 2,
2026 games right now (pre-game). Week 2 hasn't been played yet, so this is
NOT a graded backtest (no actual results exist to grade against) — it's the
live board of where the model currently disagrees with the market, using
the exact same pipeline as the Betting Edges page in app.py.

Usage:
    python scripts/week2_edge_board.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import load_ratings, load_live_lines, load_situational_tendencies
from model.game_predictor import predict_all_games
from model.situational_factors import apply_adjustments_to_predictions
from model.edge_finder import find_edges
from data.team_names import normalize

WEEK = 2


def main():
    ratings_df = load_ratings()
    print(f"Ratings loaded: {len(ratings_df)} teams")

    live_lines, source = load_live_lines()
    print(f"Live lines source: {source} ({len(live_lines)} games)")

    if "week" not in live_lines.columns:
        print("No week column on live lines — aborting.")
        return

    wk = live_lines[live_lines["week"] == WEEK].copy()
    print(f"Week {WEEK} games with lines: {len(wk)}")
    if wk.empty:
        return

    neutral_col = "neutralSite_y" if "neutralSite_y" in wk.columns else \
                  "neutralSite" if "neutralSite" in wk.columns else None
    sched = wk[["homeTeam", "awayTeam"]].copy()
    sched["neutralSite"] = wk[neutral_col] if neutral_col else False

    predicted = predict_all_games(sched, ratings_df)

    matchup_ou, team_ou, team_ats = load_situational_tendencies()
    predicted = apply_adjustments_to_predictions(predicted, matchup_ou, team_ou, team_ats)
    if "predicted_total_adj" in predicted.columns:
        predicted["predicted_total"] = predicted["predicted_total_adj"]

    lines_for_edge = wk[["homeTeam", "awayTeam", "spread", "overUnder"]].rename(
        columns={"spread": "vegas_spread", "overUnder": "vegas_total"}
    )
    edges = find_edges(predicted, lines_for_edge)

    os.makedirs("outputs", exist_ok=True)
    edges.to_csv("outputs/week2_edge_board_2026.csv", index=False)

    cols = ["homeTeam", "awayTeam", "predicted_spread", "vegas_spread", "edge_spread",
            "predicted_total", "vegas_total", "edge_total", "bet_spread", "bet_total", "edge_grade"]
    cols = [c for c in cols if c in edges.columns]

    print(f"\n{'='*70}\n  Week {WEEK} 2026 — Model vs. Vegas (live, pre-game)\n{'='*70}\n")
    print(f"  Predicted: {len(predicted)} | Matched to a Vegas line: {edges['vegas_spread'].notna().sum()}")

    graded = edges[edges["vegas_spread"].notna()].copy()
    print(f"\n  Spread edge distribution: mean={graded['edge_spread'].mean():.2f}  "
          f"std={graded['edge_spread'].std():.2f}  "
          f"|edge|>3pt: {(graded['edge_spread'].abs() > 3).sum()}/{len(graded)}")
    if "edge_total" in graded.columns:
        gt = graded[graded["edge_total"].notna()]
        print(f"  Total edge distribution:  mean={gt['edge_total'].mean():.2f}  "
              f"std={gt['edge_total'].std():.2f}  "
              f"|edge|>3pt: {(gt['edge_total'].abs() > 3).sum()}/{len(gt)}")

    print(f"\n  ── Biggest spread disagreements with Vegas ──")
    top = graded.reindex(graded["edge_spread"].abs().sort_values(ascending=False).index).head(10)
    for _, r in top.iterrows():
        print(f"  {r['awayTeam']:<22} @ {r['homeTeam']:<22} "
              f"model {r['predicted_spread']:+.1f}  vegas {r['vegas_spread']:+.1f}  "
              f"edge {r['edge_spread']:+.1f}  [{r.get('bet_spread')}]  grade={r.get('edge_grade')}")

    print(f"\n  Full board -> outputs/week2_edge_board_2026.csv")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
