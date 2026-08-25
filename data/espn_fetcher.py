"""
espn_fetcher.py — pulls 2026 preseason rosters from ESPN's public API.

CFBD /roster is empty preseason; ESPN has current rosters with position,
class year, jersey number, and active/IR/suspended status.

Cached to cache/espn_rosters_2026.json — one dict entry per CFBD team name.
"""

import os
import sys
import json
import time
import requests
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.team_names import normalize
from data.cfbd_fetcher import fetch_teams
from config import CURRENT_SEASON

BASE_URL  = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
ID_MAP_PATH     = os.path.join(CACHE_DIR, "espn_team_ids.json")
ROSTERS_PATH    = os.path.join(CACHE_DIR, f"espn_rosters_{CURRENT_SEASON}.json")

# Position abbreviation → slot in roster output
_POS_GROUP = {
    "QB":  "qbs",
    "RB":  "rbs", "FB": "rbs",
    "WR":  "wrs",
    "TE":  "tes",
    "OL":  "ols", "OT": "ols", "OG": "ols", "C": "ols",
    "DL":  "dls", "DE": "dls", "DT": "dls", "NT": "dls",
    "LB":  "lbs", "ILB": "lbs", "OLB": "lbs",
    "DB":  "dbs", "CB": "dbs", "S": "dbs", "SS": "dbs", "FS": "dbs",
    "EDGE": "dls",
    "K": "specialists", "P": "specialists", "LS": "specialists",
}

_CLASS_ORDER = {"Graduate": 5, "Senior": 4, "Junior": 3, "Sophomore": 2, "Freshman": 1}


def _get(url, params=None, retries=2):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=12)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        if attempt < retries:
            time.sleep(1.0)
    return None


# ── Team ID map ────────────────────────────────────────────────────────────

def build_espn_id_map(force_refresh=False):
    """
    Return {cfbd_school_name: espn_team_id} for all 138 FBS teams.
    Cached to cache/espn_team_ids.json.
    """
    if not force_refresh and os.path.exists(ID_MAP_PATH):
        with open(ID_MAP_PATH) as f:
            return json.load(f)

    cfbd = fetch_teams()
    cfbd_names = cfbd["school"].tolist()

    data = _get(f"{BASE_URL}/teams", params={"limit": 900})
    if not data:
        raise RuntimeError("Could not fetch ESPN team list")

    espn_teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])

    # normalized location → ESPN id
    espn_by_norm = {
        normalize(t["team"]["location"]): t["team"]["id"]
        for t in espn_teams
        if t.get("team", {}).get("location")
    }

    id_map = {}
    missing = []
    for name in cfbd_names:
        espn_id = espn_by_norm.get(normalize(name))
        if espn_id:
            id_map[name] = espn_id
        else:
            missing.append(name)

    if missing:
        print(f"WARNING: no ESPN ID found for: {missing}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(ID_MAP_PATH, "w") as f:
        json.dump(id_map, f, indent=2)

    print(f"ESPN ID map: {len(id_map)}/{len(cfbd_names)} teams matched → {ID_MAP_PATH}")
    return id_map


# ── Single-team roster ─────────────────────────────────────────────────────

def fetch_espn_roster(espn_id):
    """
    Fetch roster for one ESPN team ID.
    Returns a dict with position-keyed player lists, or None on failure.
    """
    data = _get(f"{BASE_URL}/teams/{espn_id}/roster")
    if not data:
        return None

    roster = {slot: [] for slot in set(_POS_GROUP.values())}
    roster["other"] = []
    all_names = set()

    for group in data.get("athletes", []):
        group_name = group.get("position", "")  # "offense", "defense", "specialTeam", etc.
        is_active = group_name not in ("injuredReserveOrOut", "suspended")

        for player in group.get("items", []):
            name     = player.get("displayName", "")
            pos_abbr = player.get("position", {}).get("abbreviation", "")
            class_yr = player.get("experience", {}).get("displayValue", "")
            jersey   = player.get("jersey", "")
            status   = player.get("status", {}).get("type", "active")

            all_names.add(name)
            entry = {
                "name":    name,
                "pos":     pos_abbr,
                "class":   class_yr,
                "jersey":  jersey,
                "active":  is_active and status == "active",
            }

            slot = _POS_GROUP.get(pos_abbr, "other")
            roster[slot].append(entry)

    # Sort each group by class year desc (Seniors first), then jersey number
    for slot in roster:
        roster[slot].sort(
            key=lambda p: (_CLASS_ORDER.get(p["class"], 0), -999 if not p["jersey"].isdigit() else -int(p["jersey"])),
            reverse=True
        )

    roster["all_names"] = sorted(all_names)
    roster["season"]    = data.get("season", {}).get("year", CURRENT_SEASON)
    roster["source"]    = "espn"

    # Return None if ESPN gave us an empty roster (data gap, not a real team)
    if not all_names:
        return None
    return roster


def _fetch_cfbd_roster_as_fallback(team, year=None):
    """
    Build a minimal roster dict from the CFBD /roster endpoint.
    Used when ESPN has no data for a team (e.g., Charlotte, Troy).

    Only populates all_names — position groups are left empty since CFBD
    doesn't map cleanly to our slot structure.  Portal departure data still
    runs on top of this so the main transfer moves are still caught.
    """
    if year is None:
        year = CURRENT_SEASON - 1  # most recent completed season
    try:
        from data.cfbd_fetcher import _get as cfbd_get
        data = cfbd_get("/roster", {"year": year, "team": team}, force_refresh=False)
    except Exception:
        data = []
    if not data:
        return None

    all_names = [
        f"{p.get('firstName','')} {p.get('lastName','')}".strip()
        for p in data
        if p.get("firstName") or p.get("lastName")
    ]
    return {
        "all_names": sorted(all_names),
        "qbs": [], "rbs": [], "wrs": [], "tes": [], "ols": [],
        "dls": [], "lbs": [], "dbs": [], "specialists": [], "other": [],
        "source": f"cfbd_{year}",
        "season": year,
    }


# ── Full FBS roster pull ───────────────────────────────────────────────────

def fetch_all_fbs_rosters(force_refresh=False, delay=0.25):
    """
    Fetch rosters for all 138 FBS teams. Cached to cache/espn_rosters_{year}.json.

    force_refresh=True rebuilds from scratch.
    delay: seconds between requests (be polite to ESPN's public API).

    Returns dict: {cfbd_team_name: roster_dict}
    """
    existing = {}
    if not force_refresh and os.path.exists(ROSTERS_PATH):
        with open(ROSTERS_PATH) as f:
            existing = json.load(f)

    id_map = build_espn_id_map()
    to_fetch = {name: eid for name, eid in id_map.items() if name not in existing}

    if not to_fetch:
        print(f"All {len(existing)} rosters already cached.")
        return existing

    print(f"Fetching {len(to_fetch)} rosters from ESPN...")
    rosters = dict(existing)
    errors  = []

    for i, (cfbd_name, espn_id) in enumerate(to_fetch.items(), 1):
        roster = fetch_espn_roster(espn_id)
        if roster:
            rosters[cfbd_name] = roster
        else:
            # ESPN returned empty — try CFBD prior-season roster as fallback
            fallback = _fetch_cfbd_roster_as_fallback(cfbd_name)
            if fallback:
                rosters[cfbd_name] = fallback
                print(f"  [{i}/{len(to_fetch)}] ESPN empty, CFBD fallback used: {cfbd_name} ({len(fallback['all_names'])} names)")
            else:
                errors.append(cfbd_name)
                print(f"  [{i}/{len(to_fetch)}] FAILED (no source): {cfbd_name}")
            continue

        if i % 20 == 0 or i == len(to_fetch):
            print(f"  [{i}/{len(to_fetch)}] done ({cfbd_name})")

        time.sleep(delay)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(ROSTERS_PATH, "w") as f:
        json.dump(rosters, f, indent=2)

    print(f"\nCached {len(rosters)} rosters → {ROSTERS_PATH}")
    if errors:
        print(f"Failed ({len(errors)}): {errors}")
    return rosters


def load_fbs_rosters():
    """Load cached rosters. Returns empty dict if not yet fetched."""
    if not os.path.exists(ROSTERS_PATH):
        return {}
    with open(ROSTERS_PATH) as f:
        return json.load(f)


# ── Helpers for team_profile_builder ──────────────────────────────────────

def get_qb_room(roster):
    """Return active QBs sorted by seniority (most experienced first)."""
    return [p for p in roster.get("qbs", []) if p.get("active", True)]


def is_on_roster(player_name, roster):
    """
    Check if a player name (from 2025 CFBD stats) is still on the 2026 ESPN roster.
    Normalizes periods from initials so 'C.J. Carr' matches 'CJ Carr'.
    """
    if not player_name or not roster:
        return False
    import re
    def _norm(n):
        return re.sub(r'\s+', ' ', n.replace('.', '').lower()).strip()

    target = _norm(player_name)
    return any(target == _norm(n) for n in roster.get("all_names", []))


def top_active_player_by_position(roster, slot):
    """Return the most senior active player in a position slot, or None."""
    players = [p for p in roster.get(slot, []) if p.get("active", True)]
    return players[0] if players else None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch 2026 ESPN rosters for all FBS teams")
    parser.add_argument("--force",  action="store_true", help="Re-fetch all rosters")
    parser.add_argument("--team",   help="Show roster for a specific team after fetch")
    parser.add_argument("--ids",    action="store_true", help="Rebuild ESPN team ID map")
    args = parser.parse_args()

    if args.ids:
        build_espn_id_map(force_refresh=True)

    rosters = fetch_all_fbs_rosters(force_refresh=args.force)

    if args.team:
        r = rosters.get(args.team)
        if r:
            print(f"\n{args.team} QB room:")
            for qb in get_qb_room(r):
                print(f"  #{qb['jersey']} {qb['name']} ({qb['class']})")
            print(f"\nTotal players: {len(r.get('all_names', []))}")
        else:
            print(f"'{args.team}' not found. Available: {sorted(rosters.keys())[:10]}...")
