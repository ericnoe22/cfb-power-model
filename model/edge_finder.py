"""
edge_finder.py — compares model predictions to Vegas lines to surface betting edges.

"Edge" = where the model's number differs from the market line by more than a threshold.
The bigger the edge, the stronger the signal (but also consider confidence score).

Output columns:
  edge_spread   — model_spread minus vegas_spread (negative = model likes home, positive = model likes away)
  edge_total    — model_total minus vegas_total (positive = model likes the Over)
  bet_spread    — which side to bet ("Home -X", "Away +X", or None)
  bet_total     — "Over" or "Under" or None
  edge_grade    — A/B/C/D based on edge size and confidence
"""

import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import EDGE_THRESHOLD_SPREAD, EDGE_THRESHOLD_TOTAL


def find_edges(predictions_df, lines_df):
    """
    Merge model predictions with Vegas lines and compute edges.

    predictions_df: output of predict_all_games() — has homeTeam, awayTeam,
                    predicted_spread, predicted_total
    lines_df: has homeTeam, awayTeam, spread (Vegas), overUnder (Vegas)

    Returns a merged DataFrame with edge columns.
    """
    if lines_df.empty:
        predictions_df["vegas_spread"] = np.nan
        predictions_df["vegas_total"] = np.nan
        predictions_df["edge_spread"] = np.nan
        predictions_df["edge_total"] = np.nan
        predictions_df["bet_spread"] = None
        predictions_df["bet_total"] = None
        predictions_df["edge_grade"] = None
        return predictions_df

    # Normalize line column names
    line_rename = {}
    for col in lines_df.columns:
        if col.lower() in ("spread", "line", "home_spread"):
            line_rename[col] = "vegas_spread"
        if col.lower() in ("overunder", "over_under", "total", "ou"):
            line_rename[col] = "vegas_total"
    lines_clean = lines_df.rename(columns=line_rename)

    # Ensure numeric
    for c in ["vegas_spread", "vegas_total"]:
        if c in lines_clean.columns:
            lines_clean[c] = pd.to_numeric(lines_clean[c], errors="coerce")

    merge_cols = [c for c in ["homeTeam", "awayTeam", "vegas_spread", "vegas_total"]
                  if c in lines_clean.columns]
    merged = predictions_df.merge(lines_clean[merge_cols],
                                  on=["homeTeam", "awayTeam"], how="left")

    # ── Spread edge ────────────────────────────────────────────────────────
    # Convention: spread is from home team's perspective (negative = home favored)
    # edge_spread < 0 → model more bullish on home than Vegas → bet home
    # edge_spread > 0 → model less bullish on home than Vegas → bet away
    if "vegas_spread" in merged.columns and "predicted_spread" in merged.columns:
        merged["edge_spread"] = merged["predicted_spread"] - merged["vegas_spread"]
    else:
        merged["edge_spread"] = np.nan

    # ── Total edge ─────────────────────────────────────────────────────────
    if "vegas_total" in merged.columns and "predicted_total" in merged.columns:
        merged["edge_total"] = merged["predicted_total"] - merged["vegas_total"]
    else:
        merged["edge_total"] = np.nan

    # ── Betting recommendation ─────────────────────────────────────────────
    def spread_rec(row):
        e = row.get("edge_spread")
        if pd.isna(e) or abs(e) < EDGE_THRESHOLD_SPREAD:
            return None
        home = row.get("homeTeam", "Home")
        away = row.get("awayTeam", "Away")
        vspread = row.get("vegas_spread", 0)
        if e < 0:
            # Model more bullish on home than Vegas → home is undervalued → bet home
            tag = f"{home} {vspread:+.1f}" if not pd.isna(vspread) else home
        else:
            # Model less bullish on home → away is undervalued → bet away
            adj = -vspread
            tag = f"{away} {adj:+.1f}" if not pd.isna(vspread) else away
        return tag

    def total_rec(row):
        e = row.get("edge_total")
        if pd.isna(e) or abs(e) < EDGE_THRESHOLD_TOTAL:
            return None
        return "Over" if e > 0 else "Under"

    merged["bet_spread"] = merged.apply(spread_rec, axis=1)
    merged["bet_total"]  = merged.apply(total_rec, axis=1)

    # ── Edge grade ─────────────────────────────────────────────────────────
    def grade(row):
        se = abs(row.get("edge_spread") or 0)
        te = abs(row.get("edge_total") or 0)
        conf = row.get("confidence", 0.5)
        max_edge = max(se, te)
        score = max_edge * conf
        if score >= 8:   return "A+"
        if score >= 6:   return "A"
        if score >= 4.5: return "B"
        if score >= 3:   return "C"
        return None

    merged["edge_grade"] = merged.apply(grade, axis=1)

    return merged


def summarize_edges(edges_df):
    """Return only games with actionable betting edges, sorted by grade."""
    grade_order = {"A+": 0, "A": 1, "B": 2, "C": 3}
    has_edge = edges_df[
        edges_df["bet_spread"].notna() | edges_df["bet_total"].notna()
    ].copy()
    if has_edge.empty:
        return has_edge
    has_edge["_grade_order"] = has_edge["edge_grade"].map(grade_order).fillna(99)
    return has_edge.sort_values("_grade_order").drop(columns=["_grade_order"])


def track_ats_performance(completed_games_df):
    """
    After games are complete, evaluate how the model did ATS and vs. the total.

    completed_games_df: needs homePoints, awayPoints, predicted_spread,
                        vegas_spread, predicted_total, vegas_total
    Returns summary dict: {ats_wins, ats_losses, ats_pushes, ou_wins, ou_losses, roi_pct}
    """
    df = completed_games_df.copy()
    df = df[df["homePoints"].notna() & df["awayPoints"].notna()]
    if df.empty:
        return {}

    # ATS: did the bet_spread side cover?
    df["actual_margin"] = df["homePoints"] - df["awayPoints"]  # positive = home won

    # If we bet the home team ATS (edge_spread > threshold):
    # Home covers if actual_margin > -vegas_spread (i.e. beats the spread)
    df["home_covered"] = df["actual_margin"] > -df["vegas_spread"]
    df["push_spread"] = df["actual_margin"] == -df["vegas_spread"]

    # Our bet: negative edge_spread = model more bullish on home → bet home
    df["our_bet_home"] = df["edge_spread"].fillna(0) < 0
    df["won_ats"] = (
        (df["our_bet_home"] & df["home_covered"]) |
        (~df["our_bet_home"] & ~df["home_covered"])
    )
    df["push_ats"] = df["push_spread"]

    ats_df = df[df["bet_spread"].notna()]
    ats_wins   = int(ats_df["won_ats"].sum())
    ats_losses = int((~ats_df["won_ats"] & ~ats_df["push_ats"]).sum())
    ats_pushes = int(ats_df["push_ats"].sum())

    # O/U
    df["actual_total"] = df["homePoints"] + df["awayPoints"]
    df["went_over"] = df["actual_total"] > df["vegas_total"]
    df["push_total"] = df["actual_total"] == df["vegas_total"]
    df["bet_over"] = df["bet_total"] == "Over"
    df["won_ou"] = (
        (df["bet_over"] & df["went_over"]) |
        (~df["bet_over"] & ~df["went_over"])
    )
    ou_df = df[df["bet_total"].notna()]
    ou_wins   = int(ou_df["won_ou"].sum())
    ou_losses = int((~ou_df["won_ou"] & ~ou_df["push_total"]).sum())
    ou_pushes = int(ou_df["push_total"].sum())

    total_bets = ats_wins + ats_losses + ou_wins + ou_losses
    total_wins = ats_wins + ou_wins
    roi = round((total_wins / max(total_bets, 1) - 0.5238) * 100, 1)  # -110 juice breakeven

    return {
        "ats_wins": ats_wins, "ats_losses": ats_losses, "ats_pushes": ats_pushes,
        "ou_wins":  ou_wins,  "ou_losses":  ou_losses,  "ou_pushes": ou_pushes,
        "total_bets": total_bets,
        "win_pct": round(total_wins / max(total_bets, 1) * 100, 1),
        "roi_pct": roi,
    }
