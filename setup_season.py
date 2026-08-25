"""
setup_season.py — one-time setup script to initialize a new season.

Run this now (preseason) and again in July/August when SP+/FPI are published:
    python setup_season.py           # initial setup
    python setup_season.py --force   # re-fetch when preseason ratings drop

What it pulls:
  - 2026 schedule (already available)
  - 2025 final SP+ / FPI / Elo / Talent / Returning Production (best available baseline)
  - Venues (for weather lookups)
  - Team metadata (conferences)

What it builds:
  - Preseason 2026 composite power ratings
  - Schedule merged with team ratings
"""

import argparse
import os
import sys
import pandas as pd
import json

sys.path.insert(0, os.path.dirname(__file__))
from config import CURRENT_SEASON, CFBD_API_KEY
from data.cfbd_fetcher import (
    fetch_schedule, fetch_sp_plus, fetch_fpi, fetch_elo,
    fetch_talent, fetch_returning_production,
    fetch_teams, fetch_venues,
)
from model.elo import initialize_season_elos
from model.power_rankings import build_composite_ratings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  type=int, default=CURRENT_SEASON)
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    year       = args.year
    prior_year = year - 1
    force      = args.force

    os.makedirs("cache",   exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  CFB Power Model — Season Setup ({year})")
    print(f"{'='*55}\n")

    # ── 1. Schedule ───────────────────────────────────────────────────────
    print(f"📅 Fetching {year} schedule...")
    try:
        schedule_df = fetch_schedule(year=year, force_refresh=force)
        before = len(schedule_df)
        schedule_df = schedule_df.drop_duplicates(subset=["homeTeam", "awayTeam", "week"])
        dupes = before - len(schedule_df)
        out = f"schedule_{year}_fbs.csv"
        schedule_df.to_csv(out, index=False)
        max_week = int(schedule_df["week"].max()) if "week" in schedule_df.columns else "?"
        dupe_note = f" ({dupes} duplicates removed)" if dupes else ""
        print(f"   ✅ {len(schedule_df)} games | Weeks 1–{max_week} | saved → {out}{dupe_note}")
    except Exception as e:
        print(f"   ❌ Schedule fetch failed: {e}")
        schedule_df = pd.DataFrame()

    # ── 2. SP+ ────────────────────────────────────────────────────────────
    print(f"\n📊 Fetching SP+ ratings...")
    # Manual override takes priority even over --force (use when CFBD hasn't published yet)
    sp_manual_path = f"cache/sp_plus_{year}_manual.csv"
    sp_df = pd.DataFrame()
    if os.path.exists(sp_manual_path):
        sp_df = pd.read_csv(sp_manual_path)
        print(f"   ✅ Manual SP+ loaded for {year} ({len(sp_df)} teams) from {sp_manual_path}")
    else:
        sp_df = _fetch_with_fallback("SP+", fetch_sp_plus, year, prior_year, force)
    if not sp_df.empty:
        # Normalize the main rating column name
        sp_df = sp_df.rename(columns={"rating": "sp_plus"}) if "rating" in sp_df.columns else sp_df
        sp_df.to_csv(f"cache/sp_plus_{year}.csv", index=False)
        print(f"   → Key columns: sp_plus, offense.rating, defense.rating, sos")

    # ── 3. FPI ────────────────────────────────────────────────────────────
    print(f"\n📊 Fetching FPI ratings...")
    fpi_df = _fetch_with_fallback("FPI", fetch_fpi, year, prior_year, force)
    if not fpi_df.empty:
        fpi_df.to_csv(f"cache/fpi_{year}.csv", index=False)

    # ── 4. Elo — regress prior season to mean ─────────────────────────────
    print(f"\n📊 Fetching Elo ratings...")
    elo_df = pd.DataFrame()
    try:
        prior_elo_df = fetch_elo(year=prior_year, force_refresh=force)
        if not prior_elo_df.empty:
            prior_elos = dict(zip(prior_elo_df["team"], prior_elo_df["elo"]))
            regressed  = initialize_season_elos(prior_elos)
            elo_df = pd.DataFrame([{"team": t, "elo": e} for t, e in regressed.items()])
            elo_df.to_csv("cache/elo_current.csv", index=False)
            top5 = elo_df.sort_values("elo", ascending=False).head(5)
            print(f"   ✅ {prior_year} Elo regressed to {year} preseason ({len(elo_df)} teams)")
            for _, r in top5.iterrows():
                print(f"      {r['team']}: {r['elo']:.0f}")
    except Exception as e:
        print(f"   ❌ Elo fetch failed: {e}")

    # ── 5. Talent ─────────────────────────────────────────────────────────
    print(f"\n📊 Fetching talent (247Sports composite)...")
    talent_df = _fetch_with_fallback("Talent", fetch_talent, year, prior_year, force)
    if not talent_df.empty:
        talent_df.to_csv(f"cache/talent_{year}.csv", index=False)

    # ── 6. Returning production ───────────────────────────────────────────
    print(f"\n📊 Fetching returning production...")
    returning_df = _fetch_with_fallback("Returning production",
                                        fetch_returning_production, year, prior_year, force)
    if not returning_df.empty:
        # Rename percentPPA → returning_prod for the model
        returning_df = returning_df.rename(columns={"percentPPA": "returning_prod",
                                                     "season": "year"})
        returning_df.to_csv(f"cache/returning_{year}.csv", index=False)

    # ── 7. Venues (for weather lookups) ───────────────────────────────────
    print(f"\n📍 Fetching venue locations...")
    try:
        venues_df = fetch_venues(force_refresh=force)
        venues_df.to_csv("cache/venues.csv", index=False)
        print(f"   ✅ {len(venues_df)} venues saved")
    except Exception as e:
        print(f"   ❌ Venues fetch failed: {e}")

    # ── 8. Team metadata ──────────────────────────────────────────────────
    print(f"\n🏟️  Fetching FBS team metadata...")
    try:
        teams_df = fetch_teams(force_refresh=force)
        teams_df.to_csv("cache/teams_fbs.csv", index=False)
        print(f"   ✅ {len(teams_df)} FBS teams saved")
    except Exception as e:
        print(f"   ❌ Teams fetch failed: {e}")

    # ── 9. Build composite ratings ────────────────────────────────────────
    print(f"\n⚙️  Building {year} preseason composite power ratings...")

    # Prepare FPI: the model expects a 'fpi' column + 'team'
    fpi_for_model = fpi_df[["team", "fpi"]].copy() if (not fpi_df.empty and "fpi" in fpi_df.columns) else pd.DataFrame()

    # Prepare SP+: model expects 'team' + 'rating'
    sp_for_model = sp_df.rename(columns={"sp_plus": "rating"}) if (not sp_df.empty and "sp_plus" in sp_df.columns) else sp_df

    try:
        composite = build_composite_ratings(
            sp_df=sp_for_model      if not sp_for_model.empty  else None,
            fpi_df=fpi_for_model    if not fpi_for_model.empty else None,
            elo_df=elo_df           if not elo_df.empty        else None,
            returning_df=returning_df if not returning_df.empty else None,
            talent_df=talent_df     if not talent_df.empty     else None,
            week=0,
            season=year,
        )

        if not composite.empty:
            preseason_path = f"outputs/power_ratings_{year}_preseason.csv"
            dashboard_path = f"{year}_power_rating_cleaned.csv"
            composite.to_csv(preseason_path, index=False)
            composite.to_csv(dashboard_path, index=False)
            print(f"   ✅ Saved → {preseason_path}")
            print(f"   ✅ Dashboard file updated → {dashboard_path}")
            cols = [c for c in ["rank","team","composite","sp_plus","fpi","elo"] if c in composite.columns]
            print(f"\n   Preseason Top 15 ({year}):")
            print(composite[cols].head(15).to_string(index=False))
        else:
            print("   ⚠️  Could not build composite — check SP+ data above")

    except Exception as e:
        print(f"   ❌ Composite build failed: {e}")
        import traceback; traceback.print_exc()
        composite = pd.DataFrame()

    # ── 10. Merge schedule with power ratings ─────────────────────────────
    if not schedule_df.empty and not composite.empty:
        print(f"\n📎 Merging {year} schedule with power ratings...")
        try:
            ratings_slim = composite[["team","composite","sp_plus"]].copy()

            sched_merged = schedule_df.merge(
                ratings_slim.rename(columns={"team":"homeTeam", "composite":"home_composite", "sp_plus":"home_SP"}),
                on="homeTeam", how="left"
            ).merge(
                ratings_slim.rename(columns={"team":"awayTeam", "composite":"away_composite", "sp_plus":"away_SP"}),
                on="awayTeam", how="left"
            )
            out2 = f"{year}_schedule_with_power.csv"
            sched_merged.to_csv(out2, index=False)
            print(f"   ✅ Saved → {out2}")
        except Exception as e:
            print(f"   ⚠️  Schedule merge failed: {e}")

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ✅ {year} setup complete!")
    print(f"\n  Files created:")
    print(f"    schedule_{year}_fbs.csv          ← full schedule")
    print(f"    {year}_power_rating_cleaned.csv  ← dashboard ratings")
    print(f"    {year}_schedule_with_power.csv   ← schedule + ratings merged")
    print(f"    cache/sp_plus_{year}.csv         ← SP+ (w/ offense/defense breakdown)")
    print(f"    cache/fpi_{year}.csv             ← FPI")
    print(f"    cache/elo_current.csv            ← Elo (regressed to mean)")
    print(f"\n  Next steps:")
    print(f"    streamlit run app.py             ← open the dashboard")
    print(f"\n  When preseason SP+/FPI drop in July/August:")
    print(f"    python setup_season.py --force   ← re-fetches everything")
    print(f"{'='*55}\n")


def _fetch_with_fallback(label, fetch_fn, year, prior_year, force):
    """Try fetching year, fall back to prior_year if empty."""
    try:
        df = fetch_fn(year=year, force_refresh=force)
        if not df.empty:
            print(f"   ✅ {year} {label} loaded ({len(df)} teams)")
            return df
    except Exception:
        pass

    print(f"   ⚠️  {year} {label} not yet published — using {prior_year} final ratings as baseline")
    try:
        df = fetch_fn(year=prior_year, force_refresh=force)
        if not df.empty:
            print(f"   ✅ {prior_year} {label} loaded ({len(df)} teams)")
            return df
    except Exception as e:
        print(f"   ❌ {prior_year} {label} also failed: {e}")

    return pd.DataFrame()


if __name__ == "__main__":
    main()
