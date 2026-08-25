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
        se   = abs(row.get("edge_spread") or 0)
        te   = abs(row.get("edge_total")  or 0)
        conf = row.get("confidence", 0.5)
        max_edge = max(se, te)
        score = max_edge * conf

        # ── Situational signal boosters ────────────────────────────────
        # Each is documented to have historical backing; adds up to +1.5
        # to the score when present, which can lift a B to A or A to A+.

        # 1. Home underdog in conference: public fades home dogs,
        #    sharp money tends to take them — documented +EV angle.
        vegas_sp = row.get("vegas_spread")
        is_conf  = str(row.get("conferenceGame", "")).lower() in ("true", "1", "yes")
        home_dog_conf = (
            vegas_sp is not None and
            not pd.isna(vegas_sp) and
            float(vegas_sp) > 3.0 and   # home team is underdog by 3+
            is_conf and
            (row.get("edge_spread") or 0) < 0  # model also leans home
        )
        if home_dog_conf:
            score += 0.5

        # 2. Wind threshold on totals: >15 mph suppresses scoring reliably.
        wind = row.get("weather_wind_mph")
        has_total_bet = abs(te) >= EDGE_THRESHOLD_TOTAL
        wind_under = (
            has_total_bet and
            wind is not None and
            not pd.isna(wind) and
            float(wind) > 15 and
            (row.get("edge_total") or 0) < 0  # model leans under
        )
        if wind_under:
            score += 0.5

        # 3. Rating system agreement already embedded in confidence via
        #    predict_game(), but reward games where both SP+ edge and FPI
        #    edge exist and agree with the bet (extra signal stacking).
        sp_edge  = row.get("sp_plus_edge")   # set below if available
        fpi_edge = row.get("fpi_edge")
        if (sp_edge is not None and fpi_edge is not None and
                not pd.isna(sp_edge) and not pd.isna(fpi_edge)):
            bet_home = (row.get("edge_spread") or 0) < 0
            sp_home  = float(sp_edge)  < 0
            fpi_home = float(fpi_edge) < 0
            if sp_home == fpi_home == bet_home:
                score += 0.5

        # 4. Pick Six Previews expert agreement: Ciancia's Game Grader
        #    implied spread leans the same side vs. Vegas as the model.
        #    Full boost for real P4 grades; half boost when a G6 team's
        #    grade is imputed from SP+ (noisier signal).
        if row.get("pick_six_agrees") is True:
            score += 0.25 if row.get("pick_six_estimated") else 0.5

        if score >= 8:   return "A+"
        if score >= 6:   return "A"
        if score >= 4.5: return "B"
        if score >= 3:   return "C"
        return None

    # Pre-compute per-system spread edges where component data is available
    if "home_sp_norm" in merged.columns and "away_sp_norm" in merged.columns:
        merged["sp_plus_edge"] = (
            (merged["home_sp_norm"] - merged["away_sp_norm"]) * -1
            - merged["vegas_spread"].fillna(0)
        )
    if "home_fpi_norm" in merged.columns and "away_fpi_norm" in merged.columns:
        merged["fpi_edge"] = (
            (merged["home_fpi_norm"] - merged["away_fpi_norm"]) * -1
            - merged["vegas_spread"].fillna(0)
        )

    # Pre-compute Pick Six expert agreement columns.
    # Agreement = Ciancia's Game Grader implied spread leans the same side
    # vs. Vegas as the model does, by at least 1 point. G6 teams use grades
    # imputed from SP+ (flagged estimated → reduced boost in grade()).
    try:
        from data.pick_six_loader import get_game_grader_map
        from data.team_intel import _canonical
        gg_map = get_game_grader_map()
        if gg_map:
            GG_PTS = 0.6   # spread points per Game Grader point
            HFA    = 2.5

            def _ps_cols(row):
                h = gg_map.get(_canonical(str(row.get("homeTeam", ""))))
                a = gg_map.get(_canonical(str(row.get("awayTeam", ""))))
                vegas = row.get("vegas_spread")
                model_edge = row.get("edge_spread")
                if h is None or a is None or pd.isna(vegas) or pd.isna(model_edge):
                    return pd.Series([None, None])
                neutral = str(row.get("neutralSite", "")).lower() in ("true", "1", "yes")
                implied = -((h["gg"] - a["gg"]) * GG_PTS + (0 if neutral else HFA))
                ps_edge = implied - float(vegas)
                agrees = (abs(ps_edge) >= 1.0 and
                          (ps_edge < 0) == (model_edge < 0))
                estimated = h["estimated"] or a["estimated"]
                return pd.Series([agrees, estimated])

            merged[["pick_six_agrees", "pick_six_estimated"]] = merged.apply(_ps_cols, axis=1)
    except Exception:
        pass

    merged["edge_grade"] = merged.apply(grade, axis=1)

    return merged


def summarize_edges(edges_df, min_grade="A", conf_only=True):
    """
    Return only games with actionable betting edges, sorted by grade.

    Defaults reflect the best-performing filter combination from backtesting
    (2022-2024): A/A+ grades on conference games only → +1.0% ROI.

    min_grade:  minimum grade to include ("A+", "A", "B", "C", or None for all)
    conf_only:  if True, restrict to conference games (when data is available)
    """
    grade_order = {"A+": 0, "A": 1, "B": 2, "C": 3}
    min_grades  = {"A+": {"A+"}, "A": {"A+", "A"}, "B": {"A+", "A", "B"},
                   "C": {"A+", "A", "B", "C"}}.get(min_grade or "", None)

    has_edge = edges_df[
        edges_df["bet_spread"].notna() | edges_df["bet_total"].notna()
    ].copy()

    if has_edge.empty:
        return has_edge

    # ── Grade filter ──────────────────────────────────────────────────────
    if min_grades is not None and "edge_grade" in has_edge.columns:
        has_edge = has_edge[has_edge["edge_grade"].isin(min_grades)]

    # ── Conference filter ─────────────────────────────────────────────────
    if conf_only:
        if "conferenceGame" in has_edge.columns:
            conf_mask = has_edge["conferenceGame"].astype(str).str.lower().isin(
                ("true", "1", "yes")
            )
            if conf_mask.any():
                has_edge = has_edge[conf_mask]
        elif "homeConference" in has_edge.columns and "awayConference" in has_edge.columns:
            same_conf = (
                has_edge["homeConference"].notna() &
                (has_edge["homeConference"] == has_edge["awayConference"])
            )
            if same_conf.any():
                has_edge = has_edge[same_conf]

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
