"""
update_weekly.py — run this each week to refresh all data.

Usage:
    python update_weekly.py              # update current season
    python update_weekly.py --week 5     # update through a specific week
    python update_weekly.py --force      # force re-fetch (ignore cache)
    python update_weekly.py --year 2024  # update a different season
"""

import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import CURRENT_SEASON

from data.cfbd_fetcher import (
    fetch_games, fetch_completed_games,
    fetch_sp_plus, fetch_fpi, fetch_elo, fetch_talent,
    fetch_returning_production, fetch_consensus_lines,
    fetch_teams, fetch_coaches,
    fetch_ppa_teams, fetch_epa_per_play, fetch_advanced_box_scores,
    fetch_team_game_stats,
)
from data.owls_fetcher import fetch_ncaaf_lines
from data.sagarin_fetcher import fetch_sagarin
from model.elo import run_season_elos, initialize_season_elos
from model.power_rankings import build_composite_ratings
from model.game_quality import compute_game_quality_flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  type=int, default=CURRENT_SEASON)
    parser.add_argument("--week",  type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    year  = args.year
    force = args.force
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("cache",   exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  CFB Power Model — Weekly Update ({year})")
    print(f"{'='*50}\n")

    # ── 1. Pull games ─────────────────────────────────────────────────────
    print("📥 Fetching game results...")
    try:
        games_df = fetch_completed_games(year=year, force_refresh=force)
        print(f"   ✅ {len(games_df)} completed games found")
    except Exception as e:
        print(f"   ❌ Failed to fetch games: {e}")
        games_df = pd.DataFrame()

    # ── 2. Update Elo ─────────────────────────────────────────────────────
    if not games_df.empty:
        print("\n📊 Updating Elo ratings...")
        try:
            # Load this season's starting Elo baseline. run_season_elos() replays
            # ALL completed games in games_df from this baseline every time it's
            # called, so the baseline must be frozen once per season and never
            # re-derived from cache/elo_current.csv — that file is this script's
            # own OUTPUT (this season's live, game-updated ratings), and running
            # initialize_season_elos() on it again would regress already-in-progress
            # ratings back toward 1500 on every weekly run, compounding indefinitely.
            initial_elos = {}
            preseason_csv = f"cache/elo_preseason_{year}.csv"
            prior_json    = f"cache/elo_{year-1}.json"
            prior_csv     = f"cache/elo_current.csv"
            if os.path.exists(preseason_csv):
                preseason_df = pd.read_csv(preseason_csv)
                initial_elos = dict(zip(preseason_df["team"], preseason_df["elo"]))
                print(f"   → Using frozen {year} preseason Elo baseline ({len(initial_elos)} teams) [{preseason_csv}]")
            elif os.path.exists(prior_json):
                import json as _json
                with open(prior_json) as _f:
                    _records = _json.load(_f)
                prior_elos = {r["team"]: r["elo"] for r in _records if "team" in r and "elo" in r}
                initial_elos = initialize_season_elos(prior_elos)
                pd.DataFrame([{"team": t, "elo": e} for t, e in initial_elos.items()]).to_csv(preseason_csv, index=False)
                print(f"   → Initialized from {year-1} Elo ratings ({len(initial_elos)} teams) [{prior_json}], froze baseline → {preseason_csv}")
            elif os.path.exists(prior_csv):
                prior_df = pd.read_csv(prior_csv)
                prior_elos = dict(zip(prior_df["team"], prior_df["elo"]))
                initial_elos = initialize_season_elos(prior_elos)
                pd.DataFrame([{"team": t, "elo": e} for t, e in initial_elos.items()]).to_csv(preseason_csv, index=False)
                print(f"   → Initialized from cached Elo ({len(initial_elos)} teams) [{prior_csv}], froze baseline → {preseason_csv}")

            elo_ratings, elo_history = run_season_elos(games_df, initial_elos)
            elo_ratings.to_csv(f"cache/elo_current.csv", index=False)
            elo_history.to_csv(f"cache/elo_history_{year}.csv", index=False)
            print(f"   ✅ Elo updated for {len(elo_ratings)} teams")
            print(f"   Top 5: {elo_ratings.head(5)[['team','elo']].to_string(index=False)}")
        except Exception as e:
            print(f"   ❌ Elo update failed: {e}")
            elo_ratings = pd.DataFrame()
    else:
        elo_ratings = pd.DataFrame()
        print("\n⚠️  No games found — skipping Elo update")

    # ── 2b. Advanced box scores + scoreboard/efficiency mismatch flags ────
    print(f"\n📥 Fetching advanced box scores ({year})...")
    box_df = pd.DataFrame()
    try:
        # Always pull the whole season to date (not just this week) so the
        # flags file stays a complete game log; CFBD doesn't revise past
        # weeks, so re-fetching is cheap and just extends the cache.
        box_df = fetch_advanced_box_scores(year=year, force_refresh=force)
        if not box_df.empty:
            box_df.to_csv(f"cache/advanced_box_{year}.csv", index=False)
            print(f"   ✅ Advanced box scores: {len(box_df)} team-game rows")
        else:
            print("   ⚠️  No advanced box score data (preseason or Patreon tier off)")
    except Exception as e:
        print(f"   ❌ Advanced box score fetch failed: {e}")

    # ── 2c. Traditional box score stats (turnovers) for the luck flag ─────
    # /games/teams requires a week per call, so loop over every week that
    # already has completed games; each week is cached individually so
    # re-runs only fetch the current week.
    print(f"\n📥 Fetching turnover data ({year})...")
    team_game_stats_df = pd.DataFrame()
    try:
        weeks_played = sorted(games_df["week"].dropna().unique().tolist()) if not games_df.empty else []
        weekly_frames = []
        for wk in weeks_played:
            wk_df = fetch_team_game_stats(year=year, week=int(wk), force_refresh=force)
            if not wk_df.empty:
                weekly_frames.append(wk_df)
        if weekly_frames:
            team_game_stats_df = pd.concat(weekly_frames, ignore_index=True)
            team_game_stats_df.to_csv(f"cache/team_game_stats_{year}.csv", index=False)
            print(f"   ✅ Team game stats: {len(team_game_stats_df)} team-game rows across {len(weeks_played)} weeks")
        else:
            print("   ⚠️  No team game stats — preseason or unavailable")
    except Exception as e:
        print(f"   ❌ Team game stats fetch failed: {e}")

    # ── 2d. Compute scoreboard/efficiency + turnover-luck flags ───────────
    if not box_df.empty:
        try:
            flags_df = compute_game_quality_flags(games_df, box_df, team_game_stats_df)
            if not flags_df.empty:
                flags_df.to_csv(f"outputs/game_quality_flags_{year}.csv", index=False)
                n_flagged = flags_df["flag"].notna().sum()
                n_to_flagged = flags_df["turnover_flag"].notna().sum()
                print(f"   ✅ Quality-of-win flags: {n_flagged} efficiency, {n_to_flagged} turnover-luck (of {len(flags_df)} team-games)")
        except Exception as e:
            print(f"   ❌ Quality flag computation failed: {e}")

    # ── 3. Pull SP+ (manual import takes priority over CFBD API) ─────────────
    print(f"\n📥 Fetching SP+ ratings ({year})...")
    sp_df = pd.DataFrame()
    if args.week:
        manual_sp = f"cache/sp_plus_{year}_week{args.week}_manual.csv"
        if os.path.exists(manual_sp):
            sp_df = pd.read_csv(manual_sp)
            print(f"   ✅ Manual SP+ loaded for {len(sp_df)} teams (week {args.week})")
    if sp_df.empty:
        try:
            sp_df = fetch_sp_plus(year=year, force_refresh=force)
            sp_df.to_csv(f"cache/sp_plus_{year}.csv", index=False)
            print(f"   ✅ SP+ loaded from CFBD for {len(sp_df)} teams")
        except Exception as e:
            print(f"   ❌ SP+ fetch failed: {e}")
            fallback = f"ratings_sp_{year}.csv"
            if os.path.exists(fallback):
                sp_df = pd.read_csv(fallback)
                print(f"   → Using fallback: {fallback}")
            else:
                sp_df = pd.DataFrame()

    # ── 4. Pull FPI (manual import takes priority over CFBD API) ─────────
    # CFBD's /ratings/fpi mirrors ESPN's FPI on its own sync schedule, which
    # can lag ESPN's live site by several points right after games finish
    # (confirmed: force_refresh against CFBD returned identical, stale
    # values while ESPN's site had already moved). When that happens, save
    # a fresher pull as cache/fpi_{year}_week{N}_manual.csv (team, fpi
    # columns) and it's used instead — same pattern as the SP+ manual import.
    print(f"\n📥 Fetching FPI ratings ({year})...")
    fpi_df = pd.DataFrame()
    if args.week:
        manual_fpi = f"cache/fpi_{year}_week{args.week}_manual.csv"
        if os.path.exists(manual_fpi):
            fpi_df = pd.read_csv(manual_fpi)
            print(f"   ✅ Manual FPI loaded for {len(fpi_df)} teams (week {args.week})")
    if fpi_df.empty:
        try:
            fpi_df = fetch_fpi(year=year, force_refresh=force)
            if not fpi_df.empty and "fpi" in fpi_df.columns:
                fpi_df.to_csv(f"cache/fpi_{year}.csv", index=False)
                print(f"   ✅ FPI loaded for {len(fpi_df)} teams")
            else:
                # Fall back to cached file from setup_season.py
                fpi_cache = f"cache/fpi_{year}.csv"
                if os.path.exists(fpi_cache):
                    fpi_df = pd.read_csv(fpi_cache)
                    print(f"   → Using cached FPI ({len(fpi_df)} teams)")
                else:
                    print("   ⚠️  FPI not available — preseason or API issue")
        except Exception as e:
            print(f"   ❌ FPI fetch failed: {e}")
            fpi_cache = f"cache/fpi_{year}.csv"
            if os.path.exists(fpi_cache):
                fpi_df = pd.read_csv(fpi_cache)
                print(f"   → Using cached FPI ({len(fpi_df)} teams)")

    # ── 5. Pull talent + returning production ─────────────────────────────
    print(f"\n📥 Fetching talent & returning production ({year})...")
    try:
        talent_df = fetch_talent(year=year, force_refresh=force)
        print(f"   ✅ Talent data: {len(talent_df)} teams")
    except Exception as e:
        print(f"   ❌ Talent fetch failed: {e}")
        talent_df = pd.DataFrame()

    try:
        returning_df = fetch_returning_production(year=year, force_refresh=force)
        print(f"   ✅ Returning production: {len(returning_df)} teams")
    except Exception as e:
        print(f"   ❌ Returning production fetch failed: {e}")
        returning_df = pd.DataFrame()

    # ── 4b. Pull Sagarin ratings ──────────────────────────────────────────
    print(f"\n📥 Fetching Sagarin ratings ({year})...")
    sagarin_df = pd.DataFrame()
    try:
        sagarin_df = fetch_sagarin(year=year, force_refresh=force)
        print(f"   ✅ Sagarin: {len(sagarin_df)} FBS teams")
    except Exception as e:
        print(f"   ❌ Sagarin fetch failed: {e}")

    # ── 4c. Pull opponent-adjusted EPA/PPA (Patreon tier) ─────────────────
    print(f"\n📥 Fetching opponent-adjusted EPA/PPA ({year})...")
    epa_df = pd.DataFrame()
    try:
        ppa_df = fetch_ppa_teams(year=year, force_refresh=force)
        if not ppa_df.empty:
            ppa_df.to_csv(f"cache/ppa_teams_{year}.csv", index=False)
            print(f"   ✅ PPA data: {len(ppa_df)} teams")
            epa_df = ppa_df
        else:
            # Fall back to EPA per play from advanced stats
            epa_per_play = fetch_epa_per_play(year=year, force_refresh=force)
            if not epa_per_play.empty:
                epa_per_play.to_csv(f"cache/epa_per_play_{year}.csv", index=False)
                print(f"   ✅ EPA/play data: {len(epa_per_play)} teams")
                epa_df = epa_per_play
            else:
                print("   ⚠️  No EPA data — preseason or unavailable")
    except Exception as e:
        print(f"   ❌ EPA fetch failed: {e}")
        epa_df = pd.DataFrame()

    # ── 4d. Pull defensive havoc rate (season-level advanced stats) ───────
    # Distinct signal from opponent-adjusted PPA above (~0.70 correlated, not
    # redundant) — always fetched, since it lives in /stats/season/advanced,
    # a different endpoint from /ppa/teams and not returned by the PPA path.
    print(f"\n📥 Fetching defensive havoc rate ({year})...")
    havoc_df = pd.DataFrame()
    try:
        if "havoc_total" in epa_df.columns:
            havoc_df = epa_df[["team", "havoc_total"]].copy()
        else:
            adv_df = fetch_epa_per_play(year=year, force_refresh=force)
            if not adv_df.empty and "havoc_total" in adv_df.columns:
                havoc_df = adv_df[["team", "havoc_total"]].copy()
        if not havoc_df.empty:
            havoc_df.to_csv(f"cache/havoc_{year}.csv", index=False)
            print(f"   ✅ Havoc rate: {len(havoc_df)} teams")
        else:
            print("   ⚠️  No havoc data — preseason or unavailable")
    except Exception as e:
        print(f"   ❌ Havoc fetch failed: {e}")
        havoc_df = pd.DataFrame()

    # ── 5. Build composite ratings ────────────────────────────────────────
    print("\n⚙️  Building composite power ratings...")
    try:
        # Use current Elo if freshly computed, else load prebuilt
        elo_for_model = elo_ratings
        if elo_for_model.empty:
            elo_path = "cache/elo_current.csv"
            if os.path.exists(elo_path):
                elo_for_model = pd.read_csv(elo_path)

        # Normalize SP+ DataFrame column names
        if not sp_df.empty:
            for candidate in ["rating", "overall", "value"]:
                if candidate in sp_df.columns:
                    sp_df = sp_df.rename(columns={candidate: "rating"})
                    break

        # Prepare FPI for composite builder (expects 'team' + 'fpi' columns)
        fpi_for_model = fpi_df[["team", "fpi"]].copy() if (not fpi_df.empty and "fpi" in fpi_df.columns) else None

        composite = build_composite_ratings(
            sp_df=sp_df if not sp_df.empty else None,
            fpi_df=fpi_for_model,
            sagarin_df=sagarin_df if not sagarin_df.empty else None,
            elo_df=elo_for_model if not elo_for_model.empty else None,
            returning_df=returning_df if not returning_df.empty else None,
            talent_df=talent_df if not talent_df.empty else None,
            epa_df=epa_df if not epa_df.empty else None,
            havoc_df=havoc_df if not havoc_df.empty else None,
            week=args.week,
            season=year,
        )

        if not composite.empty:
            out_path = f"outputs/power_ratings_{year}_week{args.week or 'current'}.csv"
            composite.to_csv(out_path, index=False)
            # Also save as the "current" file the dashboard reads
            composite.to_csv(f"{year}_power_rating_cleaned.csv", index=False)
            print(f"   ✅ Composite ratings saved → {out_path}")
            print(f"\n   Top 10:")
            print(composite[["rank","team","composite","sp_plus","elo"]].head(10).to_string(index=False))
        else:
            print("   ⚠️  Could not build composite — check input data")
    except Exception as e:
        print(f"   ❌ Composite build failed: {e}")
        import traceback; traceback.print_exc()

    # ── 6. Pull live lines from Owls Insight ─────────────────────────────
    print(f"\n📥 Fetching live NCAAF lines (Owls Insight)...")
    try:
        live_lines, multibook_lines = fetch_ncaaf_lines()
        if not live_lines.empty:
            live_lines.to_csv("cache/lines_live.csv", index=False)
            games_with_lines = live_lines[live_lines["spread"].notna()].shape[0]
            print(f"   ✅ {len(live_lines)} upcoming games | {games_with_lines} with spreads")
        if not multibook_lines.empty:
            multibook_lines.to_csv("cache/lines_multibook.csv", index=False)
        else:
            print("   ⚠️  No live lines available (offseason or API issue)")
    except Exception as e:
        print(f"   ❌ Owls Insight fetch failed: {e}")

    # ── 7. Pull historical lines (for performance tracking) ───────────────
    if not games_df.empty:
        print(f"\n📥 Fetching historical lines ({year})...")
        try:
            lines_df = fetch_consensus_lines(year=year, force_refresh=force)
            if not lines_df.empty:
                lines_df.to_csv(f"cache/lines_{year}.csv", index=False)
                print(f"   ✅ Lines saved for {len(lines_df)} games")
        except Exception as e:
            print(f"   ❌ Lines fetch failed: {e}")

    # ── 8. Snapshot weekly ratings BEFORE this week's games ──────────────
    if args.week:
        print(f"\n📸 Snapshotting SP+/FPI for week {args.week} (pre-game capture)...")
        try:
            from scripts.capture_weekly_ratings import capture as _capture
            _capture(year=year, week=args.week, force=force)
        except Exception as e:
            print(f"   ⚠️  Weekly snapshot failed: {e}")

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  ✅ Update complete!")
    print(f"  → Open the dashboard: streamlit run app.py")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
