"""
backtest.py — runs the power model against 2024 season data to validate
accuracy and calibrate weights before betting real money.

What it tests:
  - Builds preseason 2024 composite ratings (same method as 2026)
  - Predicts spread + total for every FBS game with a Vegas line
  - Compares predictions to actual results
  - Reports ATS record, O/U record, ROI by edge grade

Usage:
    python backtest.py              # full 2024 season
    python backtest.py --year 2023  # test a different season
    python backtest.py --save       # save detailed results to outputs/
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data.cfbd_fetcher import (
    fetch_sp_plus, fetch_fpi, fetch_elo, fetch_talent,
    fetch_returning_production, fetch_lines,
)
from model.elo import initialize_season_elos
from model.power_rankings import build_composite_ratings
from model.game_predictor import predict_game
from model.edge_finder import find_edges
from data.team_names import normalize
from config import EDGE_THRESHOLD_SPREAD, EDGE_THRESHOLD_TOTAL


JUICE = -110   # standard juice; breakeven win rate = 110/210 = 52.38%
BREAKEVEN = 110 / 210


def roi(wins, losses):
    """ROI at standard -110 juice."""
    total = wins + losses
    if total == 0:
        return 0.0
    return round((wins / total - BREAKEVEN) * 100, 1)


def grade_edge(spread_edge, total_edge, confidence):
    se = abs(spread_edge) if not pd.isna(spread_edge) else 0
    te = abs(total_edge)  if not pd.isna(total_edge)  else 0
    score = max(se, te) * confidence
    if score >= 8:   return "A+"
    if score >= 6:   return "A"
    if score >= 4.5: return "B"
    if score >= 3:   return "C"
    return None


def build_ratings(year, elo_overrides=None):
    """
    Build composite ratings for a given year.
    elo_overrides: dict {team: elo} — if provided, replaces the preseason Elo
                   (used for in-season week-by-week predictions).
    """
    prior = year - 1
    print(f"  Building {year} preseason composite ratings (using {prior} final data)...")

    sp_df  = _try_fetch(fetch_sp_plus,             year, prior, "SP+")
    fpi_df = _try_fetch(fetch_fpi,                  year, prior, "FPI")
    tal_df = _try_fetch(fetch_talent,               year, prior, "Talent")
    ret_df = _try_fetch(fetch_returning_production, year, prior, "Returning prod")

    # Elo: use provided overrides (in-season) or regress prior season to mean (preseason)
    elo_df = pd.DataFrame()
    if elo_overrides:
        elo_df = pd.DataFrame([{"team": t, "elo": e} for t, e in elo_overrides.items()])
    else:
        try:
            prior_elo = fetch_elo(year=prior)
            regressed = initialize_season_elos(dict(zip(prior_elo["team"], prior_elo["elo"])))
            elo_df = pd.DataFrame([{"team": t, "elo": e} for t, e in regressed.items()])
        except Exception as e:
            print(f"    ⚠️  Elo fetch failed: {e}")

    # Normalize column names
    if not sp_df.empty:
        sp_df = sp_df.rename(columns={"rating": "sp_plus"}) if "sp_plus" not in sp_df.columns else sp_df
        sp_df = sp_df.rename(columns={"sp_plus": "rating"})
    if not fpi_df.empty and "fpi" in fpi_df.columns:
        fpi_df = fpi_df[["team", "fpi"]]
    if not ret_df.empty:
        for col in ["percentPPA", "returning_prod"]:
            if col in ret_df.columns:
                ret_df = ret_df.rename(columns={col: "returning_prod"})
                break

    composite = build_composite_ratings(
        sp_df=sp_df if not sp_df.empty else None,
        fpi_df=fpi_df if not fpi_df.empty else None,
        elo_df=elo_df if not elo_df.empty else None,
        returning_df=ret_df if not ret_df.empty else None,
        talent_df=tal_df if not tal_df.empty else None,
        week=0, season=year,
    )
    print(f"    ✅ {len(composite)} teams rated | Top 3: "
          + ", ".join(composite.head(3)["team"].tolist()))
    return composite


def fetch_weekly_ratings(year, week):
    """
    Pull SP+ and FPI as published entering a given week.
    Uses week-1 ratings for week 1 (preseason), week N-1 for week N onwards.
    Caches each week to avoid repeated API calls.
    """
    import requests as _req
    headers = {"Authorization": f"Bearer {__import__('config').CFBD_API_KEY}"}
    base = "https://api.collegefootballdata.com"

    # SP+ published after week N reflects that week's games — use week N for predictions
    # of week N+1. For week 1 we use the preseason (week 0/1) values.
    # Prefer manual (hand-entered) over API cache
    manual_sp  = f"cache/sp_plus_{year}_week{week}_manual.csv"
    cache_sp   = f"cache/sp_plus_{year}_week{week}.csv"
    cache_fpi  = f"cache/fpi_{year}_week{week}.csv"
    manual_fpi = f"cache/fpi_{year}_week{week}_manual.csv"

    if os.path.exists(manual_sp):
        sp_df = pd.read_csv(manual_sp)
    elif os.path.exists(cache_sp):
        sp_df = pd.read_csv(cache_sp)
    else:
        r = _req.get(f"{base}/ratings/sp",  headers=headers,
                     params={"year": year, "week": week})
        sp_df = pd.json_normalize(r.json()) if r.status_code == 200 else pd.DataFrame()
        if not sp_df.empty:
            sp_df.to_csv(cache_sp, index=False)

    if os.path.exists(manual_fpi):
        fpi_df = pd.read_csv(manual_fpi)
    elif os.path.exists(cache_fpi):
        fpi_df = pd.read_csv(cache_fpi)
    else:
        r = _req.get(f"{base}/ratings/fpi", headers=headers,
                     params={"year": year, "week": week})
        fpi_df = pd.json_normalize(r.json()) if r.status_code == 200 else pd.DataFrame()
        if not fpi_df.empty:
            fpi_df.to_csv(cache_fpi, index=False)

    return sp_df, fpi_df


def build_weekly_ratings(year, week, base_ratings, elo_overrides=None):
    """
    Build composite ratings using the SP+/FPI published for a specific week.
    Keeps talent/returning prod from the base ratings (season-level only).
    """
    sp_df, fpi_df = fetch_weekly_ratings(year, week)
    if sp_df.empty:
        return base_ratings   # fall back to preseason if weekly not available

    sp_df = sp_df.rename(columns={"rating": "rating"})  # already named correctly
    fpi_clean = fpi_df[["team", "fpi"]] if (not fpi_df.empty and "fpi" in fpi_df.columns) else pd.DataFrame()

    # Pull talent/returning from base (not available weekly)
    tal_df = base_ratings[["team","talent"]].rename(columns={"talent":"talent"}) \
             if "talent" in base_ratings.columns else pd.DataFrame()
    ret_df = base_ratings[["team","returning_prod"]] \
             if "returning_prod" in base_ratings.columns else pd.DataFrame()

    # Elo: use overrides if provided
    if elo_overrides:
        elo_df = pd.DataFrame([{"team": t, "elo": e} for t, e in elo_overrides.items()])
    elif "elo" in base_ratings.columns:
        elo_df = base_ratings[["team", "elo"]]
    else:
        elo_df = pd.DataFrame()

    composite = build_composite_ratings(
        sp_df=sp_df,
        fpi_df=fpi_clean if not fpi_clean.empty else None,
        elo_df=elo_df if not elo_df.empty else None,
        returning_df=ret_df if not ret_df.empty else None,
        talent_df=tal_df if not tal_df.empty else None,
        week=week, season=year,
    )
    return composite if not composite.empty else base_ratings


def run_backtest(year=2024, lines_path=None, save=False):
    print(f"\n{'='*58}")
    print(f"  CFB Power Model — Back-Test ({year} Season)")
    print(f"{'='*58}\n")

    # ── Load preseason base ratings (talent, returning prod, preseason Elo) ──
    base_ratings = build_ratings(year)
    if base_ratings.empty:
        print("❌ Could not build ratings — aborting.")
        return

    # ── Load lines + results ──────────────────────────────────────────
    lines_df = pd.DataFrame()

    # 1. Explicit path override
    if lines_path and os.path.exists(lines_path):
        lines_df = pd.read_csv(lines_path)
        print(f"  Lines file: {lines_path} ({len(lines_df)} rows)")

    # 2. Cache file
    if lines_df.empty:
        cache_path = f"cache/lines_{year}.csv"
        if os.path.exists(cache_path):
            lines_df = pd.read_csv(cache_path)
            print(f"  Lines cache: {cache_path} ({len(lines_df)} rows)")

    # 3. Fetch from CFBD API and cache
    if lines_df.empty:
        print(f"  Fetching {year} lines from CFBD API...")
        try:
            lines_df = fetch_lines(year=year, force_refresh=True)
            if not lines_df.empty:
                os.makedirs("cache", exist_ok=True)
                lines_df.to_csv(f"cache/lines_{year}.csv", index=False)
                print(f"  ✅ {len(lines_df)} line rows fetched and cached")
        except Exception as e:
            print(f"  ❌ CFBD lines fetch failed: {e}")

    if lines_df.empty:
        print("❌ No lines data available — aborting.")
        return

    # ── Filter to FBS vs FBS with a spread and final score ───────────
    fbs = lines_df[
        (lines_df.get("homeClassification", pd.Series(["fbs"]*len(lines_df))) == "fbs") &
        (lines_df.get("awayClassification", pd.Series(["fbs"]*len(lines_df))) == "fbs") &
        lines_df["spread"].notna() &
        lines_df["homeScore"].notna()
    ].copy()

    # Prefer DraftKings; fall back to any provider. One row per game.
    if "provider" in fbs.columns:
        dk = fbs[fbs["provider"] == "DraftKings"]
        fbs = dk if not dk.empty else fbs
    fbs = fbs.drop_duplicates(subset=["homeTeam", "awayTeam", "week"])

    fbs = fbs.drop_duplicates(subset=["homeTeam", "awayTeam", "week"]).copy()
    print(f"  Games to evaluate: {len(fbs)} FBS matchups\n")

    # ── Load per-game pre-game Elo if available (in-season accuracy) ─────
    elo_by_game = {}
    elo_candidates = [f"games_with_sp_cleaned.csv", f"cache/games_{year}_regular.json"]
    for path in elo_candidates:
        if os.path.exists(path) and path.endswith(".csv"):
            eg = pd.read_csv(path)
            if "homePregameElo" in eg.columns:
                for _, r in eg.iterrows():
                    key = (normalize(str(r.get("homeTeam",""))),
                           normalize(str(r.get("awayTeam",""))),
                           int(r.get("week", 0)))
                    elo_by_game[key] = {
                        "home_elo": r.get("homePregameElo"),
                        "away_elo": r.get("awayPregameElo"),
                    }
                print(f"  In-season Elo: loaded {len(elo_by_game)} per-game entries from {path}")
                break

    # ── Pre-cache weekly ratings for all weeks in the dataset ─────────
    weeks = sorted(fbs["week"].dropna().unique().astype(int))
    print(f"  Fetching weekly SP+/FPI for weeks {weeks[0]}–{weeks[-1]}...")
    weekly_ratings_cache = {}
    for wk in weeks:
        weekly_ratings_cache[wk] = build_weekly_ratings(year, wk, base_ratings)
    print(f"  ✅ Weekly ratings ready\n")

    # ── Run predictions ───────────────────────────────────────────────
    records = []
    for _, row in fbs.iterrows():
        home = normalize(str(row["homeTeam"]))
        away = normalize(str(row["awayTeam"]))
        week = int(row.get("week", 0))

        # Use weekly SP+/FPI ratings for this game's week
        ratings = weekly_ratings_cache.get(week, base_ratings)

        # Inject per-game Elo into ratings if available
        game_ratings = ratings.copy()
        game_key = (home, away, week)
        if game_key in elo_by_game:
            elos = elo_by_game[game_key]
            if pd.notna(elos.get("home_elo")) and pd.notna(elos.get("away_elo")):
                # Only update the Elo-derived component for these two teams
                for team, elo_val in [(home, elos["home_elo"]), (away, elos["away_elo"])]:
                    mask = game_ratings["team"] == team
                    if mask.any():
                        game_ratings.loc[mask, "elo"] = elo_val
                        # Recompute composite for just these two teams
                        from config import RATING_WEIGHTS
                        w = RATING_WEIGHTS
                        game_ratings.loc[mask, "elo_norm"] = (elo_val - 1500) / 30
                        game_ratings.loc[mask, "composite"] = (
                            w["sp_plus"]        * game_ratings.loc[mask, "sp_plus_norm"] +
                            w["fpi"]            * game_ratings.loc[mask, "fpi_norm"] +
                            w["elo"]            * game_ratings.loc[mask, "elo_norm"] +
                            w["returning_prod"] * game_ratings.loc[mask, "returning_norm"] +
                            w["talent"]         * game_ratings.loc[mask, "talent_norm"]
                        )

        pred = predict_game(home, away, game_ratings,
                            neutral=row.get("neutralSite", False))

        if pred.get("predicted_spread") is None:
            continue

        model_spread = pred["predicted_spread"]
        model_total  = pred["predicted_total"]
        vegas_spread = float(row["spread"])
        vegas_total  = float(row["overUnder"]) if pd.notna(row.get("overUnder")) else None
        actual_home  = float(row["homeScore"])
        actual_away  = float(row["awayScore"]) if pd.notna(row.get("awayScore")) else None

        edge_spread = model_spread - vegas_spread
        edge_total  = (model_total - vegas_total) if vegas_total else None
        confidence  = pred.get("confidence", 0.5)
        grade       = grade_edge(edge_spread, edge_total or 0, confidence)

        # ATS result: did our pick cover?
        # spread convention: negative = home favored
        # home covers if actual_margin > -vegas_spread
        actual_margin = actual_home - (actual_away or 0)
        home_covered  = actual_margin > -vegas_spread
        push_spread   = actual_margin == -vegas_spread

        # Our spread pick:
        # edge_spread < 0 → model more bullish on home than Vegas → bet home
        # edge_spread > 0 → model less bullish on home than Vegas → bet away
        bet_home    = edge_spread < -EDGE_THRESHOLD_SPREAD
        bet_away    = edge_spread > EDGE_THRESHOLD_SPREAD
        has_spread_bet = bet_home or bet_away

        if has_spread_bet and not push_spread:
            won_spread = (bet_home and home_covered) or (bet_away and not home_covered)
        else:
            won_spread = None

        # O/U result
        won_total = None
        if edge_total is not None and vegas_total and actual_away is not None:
            actual_total = actual_home + actual_away
            bet_over  = edge_total > EDGE_THRESHOLD_TOTAL
            bet_under = edge_total < -EDGE_THRESHOLD_TOTAL
            has_total_bet = bet_over or bet_under
            push_total = actual_total == vegas_total
            if has_total_bet and not push_total:
                went_over = actual_total > vegas_total
                won_total = (bet_over and went_over) or (bet_under and not went_over)

        records.append({
            "week":          row.get("week"),
            "homeTeam":      row["homeTeam"],
            "awayTeam":      row["awayTeam"],
            "home_score":    int(actual_home),
            "away_score":    int(actual_away) if actual_away is not None else None,
            "model_spread":  round(model_spread, 1),
            "vegas_spread":  vegas_spread,
            "edge_spread":   round(edge_spread, 1),
            "model_total":   round(model_total, 1),
            "vegas_total":   vegas_total,
            "edge_total":    round(edge_total, 1) if edge_total is not None else None,
            "actual_margin": round(actual_margin, 1),
            "actual_total":  round(actual_home + (actual_away or 0), 1),
            "confidence":    confidence,
            "grade":         grade,
            "bet_spread":    "Away" if bet_away else ("Home" if bet_home else None),
            "bet_total":     "Over" if (edge_total or 0) > EDGE_THRESHOLD_TOTAL else
                             ("Under" if (edge_total or 0) < -EDGE_THRESHOLD_TOTAL else None),
            "won_spread":    won_spread,
            "won_total":     won_total,
        })

    results_df = pd.DataFrame(records)

    # ── Summary stats ─────────────────────────────────────────────────
    _print_summary(results_df, year)

    if save:
        os.makedirs("outputs", exist_ok=True)
        out = f"outputs/backtest_{year}.csv"
        results_df.to_csv(out, index=False)
        print(f"\n  Full results saved → {out}")

    return results_df


def _print_summary(df, year):
    print(f"\n{'─'*58}")
    print(f"  RESULTS — {year} Season")
    print(f"{'─'*58}")

    grades = ["A+", "A", "B", "C", None]

    # ── Spread ────────────────────────────────────────────────────────
    spread_bets = df[df["bet_spread"].notna() & df["won_spread"].notna()].copy()
    print(f"\n  SPREAD BETS  ({len(spread_bets)} total picks)")
    print(f"  {'Grade':<6} {'Picks':>6} {'W':>5} {'L':>5} {'Win%':>7} {'ROI':>7}")
    print(f"  {'─'*42}")

    for grade in grades:
        if grade is None:
            sub = spread_bets
            label = "ALL"
        else:
            sub = spread_bets[spread_bets["grade"] == grade]
            label = grade
        if sub.empty:
            continue
        w = int((sub["won_spread"] == True).sum())
        l = int((sub["won_spread"] == False).sum())
        if w + l == 0:
            continue
        pct = round(w / (w+l) * 100, 1)
        r = roi(w, l)
        flag = " ✅" if r > 0 else (" ❌" if r < -5 else "")
        print(f"  {label:<6} {w+l:>6} {w:>5} {l:>5} {pct:>6.1f}% {r:>+6.1f}%{flag}")

    # ── Totals ────────────────────────────────────────────────────────
    total_bets = df[df["bet_total"].notna() & df["won_total"].notna()].copy()
    print(f"\n  TOTAL BETS  ({len(total_bets)} total picks)")
    print(f"  {'Grade':<6} {'Picks':>6} {'W':>5} {'L':>5} {'Win%':>7} {'ROI':>7}")
    print(f"  {'─'*42}")

    for grade in grades:
        if grade is None:
            sub = total_bets
            label = "ALL"
        else:
            sub = total_bets[total_bets["grade"] == grade]
            label = grade
        if sub.empty:
            continue
        w = int((sub["won_total"] == True).sum())
        l = int((sub["won_total"] == False).sum())
        if w + l == 0:
            continue
        pct = round(w / (w+l) * 100, 1)
        r = roi(w, l)
        flag = " ✅" if r > 0 else (" ❌" if r < -5 else "")
        print(f"  {label:<6} {w+l:>6} {w:>5} {l:>5} {pct:>6.1f}% {r:>+6.1f}%{flag}")

    # ── Combined ──────────────────────────────────────────────────────
    all_bets = pd.concat([
        spread_bets.rename(columns={"won_spread": "won"})[["grade","won"]],
        total_bets.rename(columns={"won_total":  "won"})[["grade","won"]],
    ])
    w = int((all_bets["won"] == True).sum())
    l = int((all_bets["won"] == False).sum())
    r = roi(w, l)
    print(f"\n  COMBINED: {w+l} picks | {w}-{l} | {round(w/(w+l)*100,1)}% | ROI {r:+.1f}%")
    print(f"  (Break-even at -110 juice = 52.4%)")

    # ── Weekly breakdown ──────────────────────────────────────────────
    print(f"\n  WEEK-BY-WEEK (spread bets)")
    print(f"  {'Wk':<4} {'Picks':>6} {'W':>5} {'L':>5} {'Win%':>7} {'ROI':>7}")
    print(f"  {'─'*36}")
    for wk in sorted(spread_bets["week"].dropna().unique()):
        sub = spread_bets[spread_bets["week"] == wk]
        w = int((sub["won_spread"] == True).sum())
        l = int((sub["won_spread"] == False).sum())
        if w + l == 0:
            continue
        pct = round(w / (w+l) * 100, 1)
        r = roi(w, l)
        flag = " ✅" if r > 0 else ""
        print(f"  {int(wk):<4} {w+l:>6} {w:>5} {l:>5} {pct:>6.1f}% {r:>+6.1f}%{flag}")

    print(f"\n{'─'*58}\n")


def _try_fetch(fn, year, prior, label):
    try:
        df = fn(year=year)
        if not df.empty:
            return df
    except Exception:
        pass
    try:
        df = fn(year=prior)
        if not df.empty:
            print(f"    ℹ️  {label}: using {prior} final (no {year} preseason data)")
            return df
    except Exception:
        pass
    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  type=int, default=2024)
    parser.add_argument("--lines", type=str, default=None, help="Path to lines CSV")
    parser.add_argument("--save",  action="store_true", help="Save detailed results to outputs/")
    args = parser.parse_args()

    run_backtest(year=args.year, lines_path=args.lines, save=args.save)
