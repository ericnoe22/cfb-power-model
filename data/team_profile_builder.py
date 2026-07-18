"""
team_profile_builder.py — builds per-team context profiles for game preview generation.

Sources (in order of priority):
  1. CFBD 2025 player season stats  — factual, complete, last full season
  2. CFBD 2025 rosters               — position/class info
  3. coaching_changes_2026_manual.csv — our manually maintained coaching data
  4. coaches_2025.csv                 — coach names for non-changed programs
  5. 2026_power_rating_cleaned.csv    — returning production, talent scores

Output: cache/team_profiles_2026.json
  {
    "Ohio State": {
      "head_coach": "Ryan Day",
      "coaching_change": false,
      "coaching_note": "",
      "top_qb": {"name": "Julian Sayin", "pass_yds": 3323, "pass_td": 28, "pass_int": 5},
      "top_rb": {"name": "Bo Jackson", "rush_yds": 1035, "rush_td": 12},
      "top_wr": {"name": "Jeremiah Smith", "rec_yds": 1086, "rec_td": 10, "rec": 68},
      "top_defender": {"name": "Jack Sawyer", "sacks": 12.5, "tfl": 16.0},
      "returning_prod": 82.3,
      "talent": 95.2,
      "user_notes": ""   # manual override — user can add transfer info, storylines, etc.
    },
    ...
  }
"""

import os
import json
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.cfbd_fetcher import _get

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_PATH  = os.path.join(CACHE_DIR, "team_profiles_2026.json")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _pivot_stats(stats_list, category, stat_type):
    """Return df of (player, team, value) for a given category+statType, sorted desc."""
    rows = [
        {"player": s["player"], "team": s["team"], "val": _safe_float(s["stat"])}
        for s in stats_list
        if s.get("category") == category and s.get("statType") == stat_type
    ]
    return pd.DataFrame(rows).sort_values("val", ascending=False) if rows else pd.DataFrame()


def _top_player(df, team):
    """Return the top-stat player row for a given team."""
    t = df[df["team"] == team]
    return t.iloc[0] if not t.empty else None


# ── Fetch raw data ───────────────────────────────────────────────────────────

def _fetch_all_stats(year=2025):
    print(f"  Fetching {year} player stats (all teams)...")
    stats = _get("/stats/player/season", {"year": year, "seasonType": "regular"})
    if not stats:
        print("  WARNING: No stats returned")
        return []
    print(f"  Got {len(stats)} stat rows")
    return stats


def _fetch_coaches():
    """Load coach names: start from coaches_2025.csv, override with 2026 changes."""
    coaches = {}

    coaches_path = os.path.join(CACHE_DIR, "coaches_2025.csv")
    if os.path.exists(coaches_path):
        df = pd.read_csv(coaches_path)
        for _, row in df.iterrows():
            name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
            if name and row.get("school"):
                coaches[row["school"]] = {"name": name, "changed": False, "note": ""}

    changes_path = os.path.join(CACHE_DIR, "coaching_changes_2026_manual.csv")
    if os.path.exists(changes_path):
        df = pd.read_csv(changes_path)
        for _, row in df.iterrows():
            team = row.get("team", "")
            new_coach = row.get("new_coach", "")
            if team and new_coach:
                coaches[team] = {
                    "name": new_coach,
                    "changed": True,
                    "note": str(row.get("notes", "")),
                }

    return coaches


def _fetch_ratings():
    """Load returning production and talent from ratings CSV."""
    path = os.path.join(ROOT, "2026_power_rating_cleaned.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        out[row["team"]] = {
            "returning_prod": _safe_float(row.get("returning_prod")),
            "talent":         _safe_float(row.get("talent")),
        }
    return out


# ── Build profiles ───────────────────────────────────────────────────────────

def build_team_profiles(year=2025, force=False):
    """
    Build team profiles and save to cache/team_profiles_2026.json.
    Skips teams that already have a profile unless force=True.
    Returns the profiles dict.
    """
    # Load existing profiles if any
    existing = {}
    if os.path.exists(OUT_PATH) and not force:
        with open(OUT_PATH) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing profiles from cache.")

    ratings = _fetch_ratings()
    coaches = _fetch_coaches()
    teams   = sorted(ratings.keys())

    # Only fetch stats if we need to build new profiles
    teams_to_build = [t for t in teams if t not in existing] if not force else teams
    if not teams_to_build:
        print("All profiles already built. Pass force=True to rebuild.")
        return existing

    print(f"Building profiles for {len(teams_to_build)} teams...")
    stats = _fetch_all_stats(year)

    # Pre-pivot the stats we care about (one pass each)
    pass_yds  = _pivot_stats(stats, "passing",  "YDS")
    pass_td   = _pivot_stats(stats, "passing",  "TD")
    pass_int  = _pivot_stats(stats, "passing",  "INT")
    rush_yds  = _pivot_stats(stats, "rushing",  "YDS")
    rush_td   = _pivot_stats(stats, "rushing",  "TD")
    rec_yds   = _pivot_stats(stats, "receiving","YDS")
    rec_td    = _pivot_stats(stats, "receiving","TD")
    rec_no    = _pivot_stats(stats, "receiving","REC")
    sacks     = _pivot_stats(stats, "defensive","SACKS")
    tfl       = _pivot_stats(stats, "defensive","TFL")
    tackles   = _pivot_stats(stats, "defensive","TOT")

    profiles = dict(existing)

    for team in teams_to_build:
        r = ratings.get(team, {})
        c = coaches.get(team, {"name": "Unknown", "changed": False, "note": ""})

        # QB
        qb_row = _top_player(pass_yds, team)
        top_qb = None
        if qb_row is not None:
            qb_name = qb_row["player"]
            top_qb = {
                "name":      qb_name,
                "pass_yds":  int(qb_row["val"]),
                "pass_td":   int(_top_player(pass_td[pass_td["player"] == qb_name], team)["val"])
                             if not pass_td[pass_td["player"] == qb_name].empty else 0,
                "pass_int":  int(_top_player(pass_int[pass_int["player"] == qb_name], team)["val"])
                             if not pass_int[pass_int["player"] == qb_name].empty else 0,
            }

        # RB
        rb_row = _top_player(rush_yds, team)
        top_rb = None
        if rb_row is not None:
            rb_name = rb_row["player"]
            top_rb = {
                "name":     rb_name,
                "rush_yds": int(rb_row["val"]),
                "rush_td":  int(_top_player(rush_td[rush_td["player"] == rb_name], team)["val"])
                            if not rush_td[rush_td["player"] == rb_name].empty else 0,
            }

        # WR (top by receiving yards)
        wr_row = _top_player(rec_yds, team)
        top_wr = None
        if wr_row is not None:
            wr_name = wr_row["player"]
            top_wr = {
                "name":    wr_name,
                "rec_yds": int(wr_row["val"]),
                "rec_td":  int(_top_player(rec_td[rec_td["player"] == wr_name], team)["val"])
                           if not rec_td[rec_td["player"] == wr_name].empty else 0,
                "rec":     int(_top_player(rec_no[rec_no["player"] == wr_name], team)["val"])
                           if not rec_no[rec_no["player"] == wr_name].empty else 0,
            }

        # Top defender (by sacks first, then TFL, then tackles)
        def_row = _top_player(sacks, team)
        top_defender = None
        if def_row is not None and def_row["val"] >= 1.0:
            d_name = def_row["player"]
            tfl_val = _top_player(tfl[tfl["player"] == d_name], team)
            top_defender = {
                "name":  d_name,
                "sacks": float(def_row["val"]),
                "tfl":   float(tfl_val["val"]) if tfl_val is not None else 0.0,
            }
        else:
            # Fall back to TFL leader
            tfl_row = _top_player(tfl, team)
            if tfl_row is not None:
                d_name = tfl_row["player"]
                top_defender = {
                    "name":  d_name,
                    "sacks": 0.0,
                    "tfl":   float(tfl_row["val"]),
                }

        profiles[team] = {
            "head_coach":      c["name"],
            "coaching_change": c["changed"],
            "coaching_note":   c["note"],
            "top_qb":          top_qb,
            "top_rb":          top_rb,
            "top_wr":          top_wr,
            "top_defender":    top_defender,
            # returning_prod is a fraction (0.0–1.0); store as pct for readability
            "returning_prod_pct": round(r.get("returning_prod", 0.0) * 100, 1),
            # talent is raw recruit composite (e.g. 973 = elite)
            "talent_score":    round(r.get("talent", 0.0), 1),
            "user_notes":      existing.get(team, {}).get("user_notes", ""),
        }

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"Saved {len(profiles)} team profiles → {OUT_PATH}")
    return profiles


def load_team_profiles():
    """Load profiles from cache. Returns empty dict if not built yet."""
    if not os.path.exists(OUT_PATH):
        return {}
    with open(OUT_PATH) as f:
        return json.load(f)


def format_profile_for_prompt(team, profiles):
    """
    Return a compact one-line context string for injection into synopsis prompts.
    e.g. "Ryan Day (new coach); QB Julian Sayin (3323 yds, 28 TD); RB Bo Jackson (1035 yds);
          WR Jeremiah Smith (1086 yds, 10 TD); DL Jack Sawyer (12.5 sacks); RetProd: 82.3"
    """
    p = profiles.get(team)
    if not p:
        return ""

    parts = []

    coach = p.get("head_coach", "")
    if coach and coach != "Unknown":
        tag = " (new coach)" if p.get("coaching_change") else ""
        parts.append(f"{coach}{tag}")

    qb = p.get("top_qb")
    if qb and qb.get("pass_yds", 0) > 100:
        parts.append(f"QB {qb['name']} ({qb['pass_yds']} pass yds, {qb['pass_td']} TD)")

    rb = p.get("top_rb")
    if rb and rb.get("rush_yds", 0) > 50:
        parts.append(f"RB {rb['name']} ({rb['rush_yds']} rush yds)")

    wr = p.get("top_wr")
    if wr and wr.get("rec_yds", 0) > 50:
        parts.append(f"WR {wr['name']} ({wr['rec_yds']} rec yds)")

    defender = p.get("top_defender")
    if defender:
        if defender.get("sacks", 0) >= 1.0:
            parts.append(f"pass rusher {defender['name']} ({defender['sacks']} sacks)")
        elif defender.get("tfl", 0) >= 3.0:
            parts.append(f"LB/DL {defender['name']} ({defender['tfl']} TFL)")

    ret = p.get("returning_prod_pct", 0)
    if ret:
        ret_label = "high" if ret > 60 else ("mid" if ret > 40 else "low")
        parts.append(f"{ret:.0f}% production returning ({ret_label})")

    talent = p.get("talent_score", 0)
    if talent > 850:
        tier = "elite" if talent > 950 else "high"
        parts.append(f"talent {tier} ({talent:.0f})")

    notes = p.get("user_notes", "").strip()
    if notes:
        parts.append(notes)

    return "; ".join(parts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build 2026 team profiles from 2025 CFBD data")
    parser.add_argument("--force", action="store_true", help="Rebuild all profiles even if cached")
    parser.add_argument("--team",  help="Show profile for a specific team after building")
    args = parser.parse_args()

    profiles = build_team_profiles(year=2025, force=args.force)

    if args.team:
        p = profiles.get(args.team)
        if p:
            print(f"\n{args.team}:")
            print(json.dumps(p, indent=2))
            print(f"\nPrompt context: {format_profile_for_prompt(args.team, profiles)}")
        else:
            print(f"Team '{args.team}' not found. Available: {sorted(profiles.keys())[:10]}...")
