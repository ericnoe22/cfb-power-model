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
from data.cfbd_fetcher import _get, fetch_transfer_portal

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


def _fetch_portal(year=2026):
    """
    Load transfer portal data for the upcoming season.
    Returns (departures, incoming):
      departures: {(origin_team, "First Last"): destination_or_""}
      incoming:   {team: [{"name", "position", "origin", "rating"}, ...]} top-rated first
    """
    try:
        df = fetch_transfer_portal(year=year)
    except Exception as e:
        print(f"  WARNING: transfer portal fetch failed: {e}")
        return {}, {}
    if df.empty:
        return {}, {}

    df["name"] = (df["firstName"].fillna("") + " " + df["lastName"].fillna("")).str.strip()
    departures = {
        (row["origin"], row["name"]): row.get("destination") or ""
        for _, row in df.iterrows() if row.get("origin")
    }
    incoming = {}
    with_dest = df[df["destination"].notna()].copy()
    with_dest["rating"] = pd.to_numeric(with_dest["rating"], errors="coerce").fillna(0.0)
    for team, grp in with_dest.groupby("destination"):
        top = grp.sort_values("rating", ascending=False).head(3)
        incoming[team] = [
            {"name": r["name"], "position": r.get("position") or "",
             "origin": r.get("origin") or "", "rating": float(r["rating"])}
            for _, r in top.iterrows()
        ]
    return departures, incoming


def _tag_departure(player, team, departures):
    """If the player left via the portal, add a 'departed_to' field."""
    if player and (team, player["name"]) in departures:
        player["departed_to"] = departures[(team, player["name"])] or "unknown"
    return player


def _validate_roster(player, team_roster):
    """
    Cross-check a player against the ESPN 2026 roster.
    If they're not found and weren't already flagged via portal,
    mark as not on 2026 roster (graduated, silent transfer, etc.).
    """
    if not player or not team_roster or player.get("departed_to"):
        return player
    from data.espn_fetcher import is_on_roster
    if not is_on_roster(player["name"], team_roster):
        player["departed_to"] = "not on 2026 roster"
    return player


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
    departures, incoming = _fetch_portal(year=year + 1)

    # Load ESPN 2026 rosters for roster validation
    try:
        from data.espn_fetcher import load_fbs_rosters, get_qb_room
        espn_rosters = load_fbs_rosters()
        print(f"  ESPN rosters loaded: {len(espn_rosters)} teams")
    except Exception as e:
        print(f"  WARNING: ESPN rosters unavailable ({e}) — skipping roster validation")
        espn_rosters = {}

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

        # Flag any 2025 stat leaders who left via the transfer portal
        top_qb       = _tag_departure(top_qb, team, departures)
        top_rb       = _tag_departure(top_rb, team, departures)
        top_wr       = _tag_departure(top_wr, team, departures)
        top_defender = _tag_departure(top_defender, team, departures)

        # Validate against ESPN 2026 roster — catches graduates and silent transfers
        team_roster = espn_rosters.get(team, {})
        top_qb       = _validate_roster(top_qb, team_roster)
        top_rb       = _validate_roster(top_rb, team_roster)
        top_wr       = _validate_roster(top_wr, team_roster)
        top_defender = _validate_roster(top_defender, team_roster)

        # Build QB room from ESPN 2026 roster (most senior first)
        qb_room = []
        if team_roster:
            qb_room = [
                {"name": qb["name"], "class": qb["class"], "jersey": qb["jersey"]}
                for qb in get_qb_room(team_roster)
            ]

        prev = existing.get(team, {})
        profiles[team] = {
            "head_coach":      c["name"],
            "coaching_change": c["changed"],
            "coaching_note":   c["note"],
            "top_qb":          top_qb,
            "top_rb":          top_rb,
            "top_wr":          top_wr,
            "top_defender":    top_defender,
            "qb_room":         qb_room,
            "key_transfers_in": incoming.get(team, []),
            # returning_prod is a fraction (0.0–1.0); store as pct for readability
            "returning_prod_pct": round(r.get("returning_prod", 0.0) * 100, 1),
            # talent is raw recruit composite (e.g. 973 = elite)
            "talent_score":    round(r.get("talent", 0.0), 1),
            # manually editable fields — preserved across rebuilds
            "oc_name":    prev.get("oc_name", ""),
            "dc_name":    prev.get("dc_name", ""),
            "user_notes": prev.get("user_notes", ""),
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
        parts.append(f"HC {coach}{tag}")

    oc = p.get("oc_name", "").strip()
    if oc:
        parts.append(f"OC {oc}")

    dc = p.get("dc_name", "").strip()
    if dc:
        parts.append(f"DC {dc}")

    def _dest(player):
        dest = player.get("departed_to", "")
        if not dest:
            return ""
        if dest in ("unknown", "not on 2026 roster"):
            return " (transferred out)"
        return f" (transferred to {dest})"

    # QB — lead with the 2026 reality, not 2025 stats of a player who is gone
    qb = p.get("top_qb")
    qb_room = p.get("qb_room", [])
    if qb and qb.get("pass_yds", 0) > 100:
        if qb.get("departed_to"):
            # 2025 starter is gone — lead with the 2026 QB room
            if qb_room:
                starter = qb_room[0]
                depth_str = (", " + ", ".join(f"{q['name']} ({q['class']})" for q in qb_room[1:3])
                             if len(qb_room) > 1 else "")
                dest_note = _dest(qb)
                parts.append(
                    f"2026 QB: {starter['name']} ({starter['class']})"
                    f" — new starter after {qb['name']} departed{dest_note}"
                    f"{depth_str}"
                )
            else:
                parts.append(f"2026 QB situation unclear — {qb['name']} departed{_dest(qb)}")
        else:
            # Returning starter — look up class year from qb_room, then show 2025 stats
            qb_class = next(
                (q["class"] for q in qb_room if q["name"].lower().replace(".", "") ==
                 qb["name"].lower().replace(".", "")),
                "returning"
            )
            parts.append(f"2026 QB: {qb['name']} ({qb_class}"
                         f", {qb['pass_yds']} pass yds / {qb['pass_td']} TD in 2025)")

    # RB — only include if still on the team
    rb = p.get("top_rb")
    if rb and rb.get("rush_yds", 0) > 50:
        if not rb.get("departed_to"):
            parts.append(f"2026 RB: {rb['name']} ({rb['rush_yds']} rush yds in 2025)")
        else:
            parts.append(f"lost top RB {rb['name']}{_dest(rb)}")

    # WR — only include if still on the team
    wr = p.get("top_wr")
    if wr and wr.get("rec_yds", 0) > 50:
        if not wr.get("departed_to"):
            parts.append(f"2026 WR: {wr['name']} ({wr['rec_yds']} rec yds in 2025)")
        else:
            parts.append(f"lost top WR {wr['name']}{_dest(wr)}")

    # Defender — only include if still on the team
    defender = p.get("top_defender")
    if defender:
        if not defender.get("departed_to"):
            if defender.get("sacks", 0) >= 1.0:
                parts.append(f"2026 pass rusher: {defender['name']} ({defender['sacks']} sacks in 2025)")
            elif defender.get("tfl", 0) >= 3.0:
                parts.append(f"2026 LB/DL: {defender['name']} ({defender['tfl']} TFL in 2025)")
        else:
            parts.append(f"lost top defender {defender['name']}{_dest(defender)}")

    transfers_in = p.get("key_transfers_in", [])
    if transfers_in:
        adds = ", ".join(f"{t['position']} {t['name']} (from {t['origin']})" for t in transfers_in)
        parts.append(f"key transfer additions: {adds}")

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
