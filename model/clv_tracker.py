"""
clv_tracker.py — closing line value (CLV) and pick log.

CLV is the gold standard for measuring whether a betting model has genuine edge.
If your picks consistently beat the closing line, sharps agree with you — the
model is finding real signal. If CLV is negative, the model is fighting sharp money.

Pick lifecycle:
  1. Log the pick when you place the bet (model_spread, line_at_bet, bet_side)
  2. Update with closing line on game day (closing_line)
  3. Update with result after the game (actual_margin → won/loss)

Storage: outputs/pick_log.csv
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "pick_log.csv")

COLUMNS = [
    "logged_at",       # ISO timestamp when pick was logged
    "season",          # e.g. 2026
    "week",            # int
    "home_team",
    "away_team",
    "bet_side",        # "Home" or "Away" (spread) / "Over" / "Under" (total)
    "bet_type",        # "spread" or "total"
    "model_spread",    # model's predicted spread (home perspective)
    "model_total",     # model's predicted total
    "edge_grade",      # A+/A/B/C at time of logging
    "line_at_bet",     # the spread/total you actually got when placing the bet
    "closing_line",    # what DK closed at before kickoff (fill after game day)
    "actual_margin",   # home_score - away_score (fill after game)
    "actual_total",    # home_score + away_score (fill after game)
    "won",             # True/False/None (None = pending)
    "clv",             # line_at_bet - closing_line (positive = beat the close)
    "notes",           # optional free text
]

BREAKEVEN = 110 / 210   # standard -110 juice


def _load() -> pd.DataFrame:
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH, dtype={"won": object})
    return pd.DataFrame(columns=COLUMNS)


def _save(df: pd.DataFrame):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


def _game_key(season, week, home_team, away_team, bet_type):
    return (str(season), str(week), str(home_team), str(away_team), str(bet_type))


def log_pick(
    season: int,
    week: int,
    home_team: str,
    away_team: str,
    bet_side: str,
    bet_type: str,
    model_spread: float,
    model_total: float,
    line_at_bet: float,
    edge_grade: str = "",
    notes: str = "",
) -> bool:
    """
    Log a new pick. Returns False if a pick for this game+type already exists.

    line_at_bet: the actual number you bet — for spreads this is the spread you
                 received (e.g. -3.5 if you bet the home team at -3.5).
    """
    df = _load()

    # Prevent duplicate entries for the same game+bet_type
    exists = (
        (df["season"].astype(str) == str(season)) &
        (df["week"].astype(str) == str(week)) &
        (df["home_team"] == home_team) &
        (df["away_team"] == away_team) &
        (df["bet_type"] == bet_type)
    )
    if exists.any():
        return False

    row = {
        "logged_at":    datetime.now(timezone.utc).isoformat(),
        "season":       season,
        "week":         week,
        "home_team":    home_team,
        "away_team":    away_team,
        "bet_side":     bet_side,
        "bet_type":     bet_type,
        "model_spread": model_spread,
        "model_total":  model_total,
        "edge_grade":   edge_grade,
        "line_at_bet":  line_at_bet,
        "closing_line": None,
        "actual_margin": None,
        "actual_total":  None,
        "won":          None,
        "clv":          None,
        "notes":        notes,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)
    return True


def update_closing_line(
    season: int,
    week: int,
    home_team: str,
    away_team: str,
    bet_type: str,
    closing_line: float,
) -> bool:
    """
    Record the closing line for a pick. Computes CLV automatically.

    CLV convention:
      Spread bets: CLV = line_at_bet − closing_line (from our side's perspective)
        - Bet home at -3, closes at -5 → CLV = +2 (we got a better number)
        - Bet away at +3, closes at +1 → CLV = +2 (same logic, flipped)
      Total bets:
        - Bet Over at 47, closes at 49 → CLV = +2 (line moved our way)
        - Bet Under at 47, closes at 45 → CLV = +2
    """
    df = _load()
    mask = (
        (df["season"].astype(str) == str(season)) &
        (df["week"].astype(str) == str(week)) &
        (df["home_team"] == home_team) &
        (df["away_team"] == away_team) &
        (df["bet_type"] == bet_type)
    )
    if not mask.any():
        return False

    df.loc[mask, "closing_line"] = closing_line

    # Compute CLV
    for idx in df[mask].index:
        row = df.loc[idx]
        lat = pd.to_numeric(row["line_at_bet"], errors="coerce")
        cl  = float(closing_line)
        if pd.isna(lat):
            continue

        bt = str(row["bet_type"])
        bs = str(row["bet_side"])
        if bt == "spread":
            # bet_side is "Home" or "Away"
            if bs == "Home":
                # We bet the home team. Better number = lower home spread (more negative)
                # e.g. got -3 vs close -5 → we got 2 extra points on home → CLV = +2
                df.loc[idx, "clv"] = round(cl - lat, 1)
            else:
                # Bet away. Our side's number is -line_at_bet for away.
                # got +3 (line=-3) vs closed +1 (line=-1) → CLV = +2
                df.loc[idx, "clv"] = round(lat - cl, 1)
        elif bt == "total":
            if bs == "Over":
                # We want total to go high. Lower close = worse for us.
                df.loc[idx, "clv"] = round(cl - lat, 1)
            else:
                # Under. Higher close = worse for us.
                df.loc[idx, "clv"] = round(lat - cl, 1)

    _save(df)
    return True


def update_outcome(
    season: int,
    week: int,
    home_team: str,
    away_team: str,
    home_score: float,
    away_score: float,
) -> int:
    """
    Record the final score for a game. Updates ALL pick types for that game.
    Returns the number of rows updated.
    """
    df = _load()
    mask = (
        (df["season"].astype(str) == str(season)) &
        (df["week"].astype(str) == str(week)) &
        (df["home_team"] == home_team) &
        (df["away_team"] == away_team)
    )
    if not mask.any():
        return 0

    margin = home_score - away_score
    total  = home_score + away_score
    df.loc[mask, "actual_margin"] = margin
    df.loc[mask, "actual_total"]  = total

    for idx in df[mask].index:
        row   = df.loc[idx]
        bt    = str(row["bet_type"])
        bs    = str(row["bet_side"])
        lat   = pd.to_numeric(row["line_at_bet"], errors="coerce")
        if pd.isna(lat):
            continue

        if bt == "spread":
            # Home covers if actual_margin > -line_at_bet (spread from home's perspective)
            push = (margin == -lat)
            if push:
                df.loc[idx, "won"] = None   # push
            else:
                home_covered = margin > -lat
                df.loc[idx, "won"] = str(
                    (bs == "Home" and home_covered) or
                    (bs == "Away" and not home_covered)
                )
        elif bt == "total":
            push = (total == lat)
            if push:
                df.loc[idx, "won"] = None
            else:
                went_over = total > lat
                df.loc[idx, "won"] = str(
                    (bs == "Over" and went_over) or
                    (bs == "Under" and not went_over)
                )

    _save(df)
    return int(mask.sum())


def load_log() -> pd.DataFrame:
    """Return the full pick log, parsed with correct dtypes."""
    df = _load()
    for col in ["model_spread", "model_total", "line_at_bet",
                "closing_line", "actual_margin", "actual_total", "clv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_stats(df: pd.DataFrame = None) -> dict:
    """
    Compute summary stats over all completed picks.
    Returns a dict suitable for display.
    """
    if df is None:
        df = load_log()

    settled = df[df["won"].notna() & (df["won"] != "None")].copy()
    settled["_won"] = settled["won"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    settled = settled[settled["_won"].notna()]

    n_picks = len(settled)
    if n_picks == 0:
        return {"n_picks": 0, "win_pct": None, "roi": None, "avg_clv": None}

    wins = settled["_won"].sum()
    roi  = round((wins / n_picks - BREAKEVEN) * 100, 1)

    clv_vals = pd.to_numeric(settled["clv"], errors="coerce").dropna()
    avg_clv  = round(clv_vals.mean(), 2) if not clv_vals.empty else None

    # By grade
    by_grade = {}
    for grade in ["A+", "A", "B", "C"]:
        sub = settled[settled["edge_grade"] == grade]
        if len(sub) == 0:
            continue
        g_wins = sub["_won"].sum()
        by_grade[grade] = {
            "n":    len(sub),
            "wins": int(g_wins),
            "pct":  round(g_wins / len(sub) * 100, 1),
            "roi":  round((g_wins / len(sub) - BREAKEVEN) * 100, 1),
        }

    # CLV positive rate
    clv_pos_rate = None
    if not clv_vals.empty:
        clv_pos_rate = round((clv_vals > 0).sum() / len(clv_vals) * 100, 1)

    return {
        "n_picks":      n_picks,
        "wins":         int(wins),
        "losses":       int(n_picks - wins),
        "win_pct":      round(wins / n_picks * 100, 1),
        "roi":          roi,
        "avg_clv":      avg_clv,
        "clv_pos_rate": clv_pos_rate,
        "by_grade":     by_grade,
        "n_pending":    int((df["won"].isna() | (df["won"] == "None")).sum()),
    }
