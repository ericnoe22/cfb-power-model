"""
game_predictor.py — predicts spreads and totals for upcoming games.

Prediction logic:
  predicted_spread = (away_composite - home_composite) - HFA + weather_spread_adj
    (negative = home team favored, matching standard sportsbook convention)

  predicted_total  = avg_points_per_team * 2 + offensive_bonus + weather_total_adj

The model is intentionally transparent so you can see exactly why it predicts what it does.
"""

import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import HOME_FIELD_ADVANTAGE
from data.team_names import normalize

# Average FBS points per team per game. Use 26.5 (total ~53) as the neutral
# baseline — most competitive games land 47-57, so this is a sensible midpoint.
BASELINE_POINTS_PER_TEAM = 26.5


MAX_FBS_SPREAD = 45.0   # no FBS vs FBS game should ever exceed this in magnitude


def predict_spread(home_composite, away_composite, neutral=False):
    """
    Predict the point spread from the home team's perspective.
    Negative = home team favored (standard convention).

    home_composite, away_composite: composite ratings (SP+ scale, points vs avg)
    Capped at ±MAX_FBS_SPREAD to prevent unrealistic outputs for extreme mismatches.
    """
    hfa = 0.0 if neutral else HOME_FIELD_ADVANTAGE
    diff = home_composite - away_composite + hfa
    spread = -diff
    spread = max(-MAX_FBS_SPREAD, min(MAX_FBS_SPREAD, spread))
    return round(spread, 1)


def predict_total(home_composite, away_composite,
                  home_off_rating=None, away_off_rating=None,
                  home_def_rating=None, away_def_rating=None,
                  league_avg_off=25.7,
                  weather_adj=0.0):
    """
    Predict the game total (combined score).

    Preferred path: uses SP+ offense.rating and defense.rating directly.
      expected_home = home_off + away_def - league_avg_off
      expected_away = away_off + home_def - league_avg_off
      total = expected_home + expected_away

    This correctly suppresses totals when two elite defenses meet (e.g. Texas vs
    Ohio State), instead of the naive composite-only approach which over-predicts.

    Fallback: composite-based estimate when SP+ breakdown is unavailable.
    weather_adj: negative number reduces total (wind, cold, precipitation).
    """
    if (home_off_rating is not None and away_off_rating is not None
            and home_def_rating is not None and away_def_rating is not None):
        # SP+ offense/defense breakdown — most accurate
        home_expected = home_off_rating + away_def_rating - league_avg_off
        away_expected = away_off_rating + home_def_rating - league_avg_off
        # Per-team floor/ceiling: even the worst FBS offense scores 17;
        # even the best offense against the worst defense caps at 38 —
        # SP+ extreme values compound unrealistically without this ceiling.
        home_expected = max(17.0, min(38.0, home_expected))
        away_expected = max(17.0, min(38.0, away_expected))
        base_total = home_expected + away_expected
        # Total floor: two evenly matched G5 teams still combine for ~42 pts;
        # the per-team floor of 17 alone bottoms out too low for those matchups.
        base_total = max(42.0, base_total)
    else:
        # Fallback: composite ratings with conservative scaling factor.
        # Composites reflect overall team quality (offense + defense), so high
        # composites on both sides don't necessarily mean more scoring — elite
        # defenses cancel out elite offenses. Use 0.05 multiplier + cap at 58.
        base_total = BASELINE_POINTS_PER_TEAM * 2 + (home_composite + away_composite) * 0.05
        base_total = min(base_total, 58.0)

    base_total += weather_adj
    return round(base_total, 1)


def _week_confidence(week):
    """
    Confidence multiplier based on how many weeks of in-season data exist.
    Preseason SP+/FPI are projections; week 6+ ratings incorporate real results.
    """
    if week is None or week == 0:
        return 0.60   # preseason
    if week <= 2:
        return 0.70
    if week <= 4:
        return 0.80
    if week <= 7:
        return 0.90
    return 1.00       # week 8+ — ratings are battle-tested


def predict_game(home_team, away_team, ratings_df, neutral=False,
                 weather_adj_total=0.0, weather_adj_spread=0.0, week=None):
    """
    Predict spread and total for a single game.

    ratings_df: DataFrame with columns [team, composite] (output of build_composite_ratings)
    Returns dict with predicted_spread, predicted_total, home_edge_rating, confidence.
    """
    home_team = normalize(home_team)
    away_team = normalize(away_team)
    home_row = ratings_df[ratings_df["team"] == home_team]
    away_row = ratings_df[ratings_df["team"] == away_team]

    if home_row.empty or away_row.empty:
        return {
            "home_team": home_team, "away_team": away_team,
            "predicted_spread": None, "predicted_total": None,
            "error": f"Missing ratings for: "
                     f"{home_team if home_row.empty else ''} "
                     f"{away_team if away_row.empty else ''}".strip()
        }

    h_comp = home_row["composite"].values[0]
    a_comp = away_row["composite"].values[0]

    def _val(row, col):
        return row[col].values[0] if col in row.columns and not pd.isna(row[col].values[0]) else None

    h_off = _val(home_row, "offense.rating")
    a_off = _val(away_row, "offense.rating")
    h_def = _val(home_row, "defense.rating")
    a_def = _val(away_row, "defense.rating")

    spread = predict_spread(h_comp, a_comp, neutral) + weather_adj_spread
    total  = predict_total(h_comp, a_comp,
                           home_off_rating=h_off, away_off_rating=a_off,
                           home_def_rating=h_def, away_def_rating=a_def,
                           weather_adj=weather_adj_total)

    # ── Confidence ────────────────────────────────────────────────────────
    # Combines two signals:
    #   1. Week-based reliability (preseason ratings are projections, not results)
    #   2. Rating-system agreement (SP+ and FPI leaning same direction = more signal)
    def _has_val(row, col):
        vals = row[col].values if col in row.columns else []
        return len(vals) > 0 and not pd.isna(vals[0])

    week_conf = _week_confidence(week)

    # Agreement bonus: do SP+ and FPI both favour the same side?
    h_sp_norm = home_row["sp_plus_norm"].values[0] if "sp_plus_norm" in home_row.columns else None
    a_sp_norm = away_row["sp_plus_norm"].values[0] if "sp_plus_norm" in away_row.columns else None
    h_fpi_norm = home_row["fpi_norm"].values[0]    if "fpi_norm"     in home_row.columns else None
    a_fpi_norm = away_row["fpi_norm"].values[0]    if "fpi_norm"     in away_row.columns else None

    agreement_bonus = 0.0
    if None not in (h_sp_norm, a_sp_norm, h_fpi_norm, a_fpi_norm):
        sp_home_favoured  = (h_sp_norm  - a_sp_norm)  > 0
        fpi_home_favoured = (h_fpi_norm - a_fpi_norm) > 0
        if sp_home_favoured == fpi_home_favoured:
            agreement_bonus = 0.10   # both systems agree on winner

    confidence = round(min(1.0, week_conf + agreement_bonus), 2)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "neutral": neutral,
        "home_composite": round(h_comp, 2),
        "away_composite": round(a_comp, 2),
        "predicted_spread": round(spread, 1),
        "predicted_total":  round(total, 1),
        "weather_total_adj": weather_adj_total,
        "confidence": confidence,
    }


def predict_all_games(schedule_df, ratings_df, fcs_lookup=None, week=None):
    """
    Run predictions for every game in the schedule.
    schedule_df: needs homeTeam, awayTeam, neutralSite columns.
    fcs_lookup: optional output of build_fcs_lookup() for FCS opponent games.
    Returns schedule_df with prediction columns appended.
    """
    from model.fcs_adjustment import is_fcs_game, predict_fcs_game

    # Load FCS lookup if not provided
    if fcs_lookup is None:
        try:
            from model.fcs_adjustment import build_fcs_lookup
            fcs_lookup = build_fcs_lookup()
        except Exception:
            fcs_lookup = {}

    # Normalize team names in the schedule before predicting so merge keys align
    schedule_df = schedule_df.copy()
    schedule_df["_home"] = schedule_df["homeTeam"].map(normalize)
    schedule_df["_away"] = schedule_df["awayTeam"].map(normalize)

    # Normalize ratings team names to match — ensures "UTSA" → "UT San Antonio" etc.
    # are found correctly rather than being misclassified as FCS opponents.
    ratings_norm = ratings_df.copy()
    ratings_norm["team"] = ratings_norm["team"].map(normalize)

    preds = []
    for _, row in schedule_df.iterrows():
        home = row["_home"]
        away = row["_away"]

        # Detect FBS vs FCS matchup
        fcs, fbs_team, fcs_team, fbs_is_home = is_fcs_game(home, away, ratings_norm)

        if fcs and fcs_lookup:
            fbs_row = ratings_norm[ratings_norm["team"] == fbs_team]
            fbs_comp = float(fbs_row["composite"].values[0]) if not fbs_row.empty else 0.0
            pred = predict_fcs_game(fbs_team, fbs_is_home, fbs_comp, fcs_lookup)
            pred["home_team"] = home
            pred["away_team"] = away
            pred["home_composite"] = fbs_comp if fbs_is_home else None
            pred["away_composite"] = None if fbs_is_home else fbs_comp
            pred["weather_total_adj"] = 0.0
        else:
            game_week = week or (int(row["week"]) if pd.notna(row.get("week")) else None)
            pred = predict_game(
                home_team=home,
                away_team=away,
                ratings_df=ratings_norm,
                neutral=row.get("neutralSite", False),
                weather_adj_total=row.get("weather_total_adj", 0.0),
                weather_adj_spread=row.get("weather_spread_adj", 0.0),
                week=game_week,
            )
            pred["home_team"] = home
            pred["away_team"] = away
            pred["fcs_note"] = None

        preds.append(pred)

    pred_df = pd.DataFrame(preds)
    merge_cols = [c for c in [
        "home_team", "away_team", "predicted_spread", "predicted_total",
        "home_composite", "away_composite", "weather_total_adj",
        "confidence", "fcs_note",
    ] if c in pred_df.columns]

    result = schedule_df.merge(
        pred_df[merge_cols],
        left_on=["_home", "_away"],
        right_on=["home_team", "away_team"],
        how="left"
    )
    result = result.drop(columns=["_home", "_away"], errors="ignore")
    return result
