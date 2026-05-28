"""
owls_fetcher.py — pulls live betting lines from Owls Insight API.

Fetches spreads and totals for NCAAF, merges them into one row per game,
and returns a DataFrame matching the format the rest of the model expects.

Book priority: DraftKings → FanDuel → BetMGM → first available
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import OWLS_INSIGHT_API_KEY
from data.team_names import normalize

BASE_URL = "https://api.owlsinsight.com/api/v1"
HEADERS  = {"Authorization": f"Bearer {OWLS_INSIGHT_API_KEY}"}

# Book preference order for consensus line
BOOK_PRIORITY = ["draftkings", "fanduel", "betmgm", "bet365", "caesars",
                 "pinnacle", "circa", "westgate"]

# Owls Insight uses "School Mascot" format — strip known mascots
# to match CFBD team names (school name only)
MASCOT_STRIP = [
    # Multi-word mascots first (order matters — longest match wins)
    "Crimson Tide", "Nittany Lions", "Tar Heels", "Blue Devils", "Demon Deacons",
    "Golden Hurricane",
    "Yellow Jackets", "Yellowjackets", "Golden Hurricanes", "Golden Flashes",
    "Golden Eagles", "Golden Bears", "Blue Raiders", "Thundering Herd",
    "Rainbow Warriors", "Wolf Pack", "Wolfpack", "Scarlet Knights", "Red Raiders",
    "Horned Frogs", "Fighting Irish", "Fighting Illini", "Mountain Hawks",
    "Red Wolves", "Sun Devils", "Mean Green", "Green Wave", "Hilltoppers",
    # Single-word mascots
    "Volunteers", "Bulldogs", "Wildcats", "Tigers", "Rebels", "Gators",
    "Seminoles", "Hurricanes", "Buckeyes", "Wolverines", "Spartans", "Hawkeyes",
    "Badgers", "Cornhuskers", "Sooners", "Cowboys", "Longhorns", "Aggies",
    "Razorbacks", "Gamecocks", "Cavaliers", "Hokies", "Mustangs", "Bears",
    "Cougars", "Utes", "Ducks", "Beavers", "Trojans", "Bruins", "Cardinal",
    "Huskies", "Buffaloes", "Rams", "Falcons", "Panthers", "Eagles", "Owls",
    "Knights", "Monarchs", "Flames", "Blazers", "Chargers", "Jaguars",
    "Aztecs", "Warriors", "Broncos", "Lobos", "Miners", "Roadrunners",
    "Colonels", "Raiders", "Catamounts", "Chanticleers", "Warhawks", "Mavericks",
    "Bobcats", "Mountaineers", "Zips", "Flashes", "Cardinals", "Penguins",
    "Lions", "Grizzlies", "Bison", "Jackrabbits", "Cyclones", "Boilermakers",
    "Wolfpack", "Leathernecks", "Redbirds", "Salukis", "Norse", "Braves",
    "Hoosiers", "Jayhawks", "Bearcats", "Bulls", "Rockets", "Chippewas",
    "Bearkats", "Pirates", "Toreros", "Dolphins", "Spiders", "Dukes",
    "Seahawks", "Thunderbirds", "Lumberjacks", "49ers", "Retrievers",
    "Redhawks", "Owls", "Hilltoppers", "Flames", "Mocs", "Norse",
    "Penguins", "Ospreys", "Flames", "Buccaneers",
]


def _clean_team_name(raw_name):
    """
    Strip mascot from 'School Mascot' format → 'School'.
    Falls back to normalize() which handles known CFBD variants.
    """
    name = raw_name.strip()
    # Strip venue notes like "(Neutral Venue)"
    if "(" in name:
        name = name[:name.index("(")].strip()
    # Try stripping known mascots (longest match first to avoid partial strips)
    for mascot in sorted(MASCOT_STRIP, key=len, reverse=True):
        if name.endswith(mascot):
            cleaned = name[: -len(mascot)].strip()
            if cleaned:
                return normalize(cleaned)
    return normalize(name)


def _best_line(game_data, market_key, book_priority=BOOK_PRIORITY):
    """
    Extract the best available line from a game's bookmakers list.
    Returns (spread_or_total, home_price, away_price) or None.
    """
    bookmakers = game_data.get("bookmakers", [])
    book_map = {b["key"]: b for b in bookmakers}

    # Try books in priority order
    for book in book_priority:
        if book in book_map:
            for market in book_map[book].get("markets", []):
                if market["key"] == market_key:
                    return market["outcomes"], book_map[book]["title"]

    # Fallback: first available
    for b in bookmakers:
        for market in b.get("markets", []):
            if market["key"] == market_key:
                return market["outcomes"], b["title"]

    return None, None


def fetch_ncaaf_lines(force_refresh=False):
    """
    Fetch current NCAAF spreads, totals, and moneylines from Owls Insight.

    Returns two DataFrames as a tuple:
      consensus_df  — one row per game, best available line per market
      multibook_df  — one row per game × book, all markets side by side
    """
    odds_raw = _get_endpoint("ncaaf/odds")
    if not odds_raw:
        return pd.DataFrame(), pd.DataFrame()

    all_games = _flatten_by_game(odds_raw)

    consensus_rows = []
    multibook_rows = []

    for game_id, g in all_games.items():
        home_raw = g.get("home_team", "")
        away_raw = g.get("away_team", "")
        home = _clean_team_name(home_raw)
        away = _clean_team_name(away_raw)
        commence = g.get("commence_time", "")

        book_data = {}   # book_key → {spread, home_ml, away_ml, total}

        for bm in g.get("bookmakers", []):
            book_key   = bm["key"]
            book_title = bm["title"]
            spread = home_ml = away_ml = total = None

            for mkt in bm.get("markets", []):
                if mkt["key"] == "spreads":
                    for o in mkt["outcomes"]:
                        if _clean_team_name(o["name"]) == home:
                            spread = float(o["point"])
                elif mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        if _clean_team_name(o["name"]) == home:
                            home_ml = int(o["price"])
                        else:
                            away_ml = int(o["price"])
                elif mkt["key"] == "totals":
                    for o in mkt["outcomes"]:
                        if o["name"] == "Over":
                            total = float(o["point"])

            book_data[book_key] = {
                "spread": spread, "home_ml": home_ml,
                "away_ml": away_ml, "total": total,
            }

            multibook_rows.append({
                "homeTeam": home, "awayTeam": away,
                "commence_time": commence,
                "book": book_title,
                "spread": spread, "home_ml": home_ml,
                "away_ml": away_ml, "overUnder": total,
            })

        # ── Consensus: best line per market ──────────────────────────
        # Spread: most favorable for the bettor = widest range available
        # (we store home-perspective; best home spread = most negative,
        #  but we want the consensus number, so use priority book)
        outcomes, spread_book = _best_line(g, "spreads")
        spread = None
        if outcomes:
            for o in outcomes:
                if _clean_team_name(o["name"]) == home:
                    spread = float(o["point"])
                    break

        outcomes, total_book = _best_line(g, "totals")
        over_under = None
        if outcomes:
            for o in outcomes:
                if o["name"] == "Over":
                    over_under = float(o["point"])
                    break

        outcomes, ml_book = _best_line(g, "h2h")
        home_ml = away_ml = None
        if outcomes:
            for o in outcomes:
                if _clean_team_name(o["name"]) == home:
                    home_ml = int(o["price"])
                else:
                    away_ml = int(o["price"])

        # Best available spread across books (home perspective — most negative = best home line)
        all_spreads = [v["spread"] for v in book_data.values() if v["spread"] is not None]
        best_home_spread = min(all_spreads) if all_spreads else spread  # lowest = best for home bettors
        best_away_spread = max(all_spreads) if all_spreads else ((-spread) if spread else None)

        # Best moneyline per side
        home_mls = [v["home_ml"] for v in book_data.values() if v["home_ml"] is not None]
        away_mls = [v["away_ml"] for v in book_data.values() if v["away_ml"] is not None]
        best_home_ml = max(home_mls) if home_mls else home_ml  # highest = best payout if favorite, least risk if dog
        best_away_ml = max(away_mls) if away_mls else away_ml

        # Best total: highest over line (best for over bettors), lowest for under
        all_totals = [v["total"] for v in book_data.values() if v["total"] is not None]
        best_over_total  = max(all_totals) if all_totals else over_under
        best_under_total = min(all_totals) if all_totals else over_under

        consensus_rows.append({
            "homeTeam":        home,
            "awayTeam":        away,
            "homeTeam_raw":    home_raw,
            "awayTeam_raw":    away_raw,
            "commence_time":   commence,
            "spread":          spread,
            "overUnder":       over_under,
            "spread_book":     spread_book,
            "total_book":      total_book,
            "home_ml":         home_ml,
            "away_ml":         away_ml,
            "best_home_spread": best_home_spread,
            "best_away_spread": best_away_spread,
            "best_home_ml":    best_home_ml,
            "best_away_ml":    best_away_ml,
            "best_over_total": best_over_total,
            "best_under_total":best_under_total,
            "books_available": len(book_data),
        })

    consensus_df  = pd.DataFrame(consensus_rows)
    multibook_df  = pd.DataFrame(multibook_rows)

    for df in [consensus_df, multibook_df]:
        if not df.empty and "commence_time" in df.columns:
            df["commence_dt"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")

    if not consensus_df.empty:
        consensus_df = consensus_df.sort_values("commence_dt").reset_index(drop=True)

    return consensus_df, multibook_df


def fetch_ncaaf_lines_consensus(force_refresh=False):
    """Convenience wrapper — returns only the consensus DataFrame."""
    consensus, _ = fetch_ncaaf_lines(force_refresh)
    return consensus


def fetch_ncaaf_schedule():
    """
    Fetch the full NCAAF schedule from Owls Insight (regardless of line availability).
    Returns a DataFrame with homeTeam, awayTeam, commence_time, week columns.
    """
    data = _get_endpoint("ncaaf/schedule")
    if not data:
        return pd.DataFrame()

    rows = []
    # Schedule endpoint returns a list or dict of games
    games = data if isinstance(data, list) else list(data.values())
    for g in games:
        home_raw = g.get("home_team", "")
        away_raw = g.get("away_team", "")
        rows.append({
            "homeTeam":      _clean_team_name(home_raw),
            "awayTeam":      _clean_team_name(away_raw),
            "homeTeam_raw":  home_raw,
            "awayTeam_raw":  away_raw,
            "commence_time": g.get("commence_time", ""),
            "id":            g.get("id", ""),
        })

    df = pd.DataFrame(rows)
    if not df.empty and "commence_time" in df.columns:
        df["commence_dt"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
        df = df.sort_values("commence_dt").reset_index(drop=True)
    return df


def fetch_ncaaf_win_totals():
    """
    Fetch season win total lines (futures) for NCAAF teams.
    Tries the futures endpoint; returns DataFrame with team, wins_line, over_odds,
    under_odds, book columns.  Returns empty DataFrame if unavailable.
    """
    # Try futures endpoint — may require higher subscription tier
    for path in ("ncaaf/futures", "ncaaf/odds/futures"):
        raw = _get_endpoint(path)
        if raw:
            break
    else:
        return pd.DataFrame()

    rows = []
    markets = raw if isinstance(raw, list) else list(raw.values())
    for item in markets:
        team_raw = item.get("team", item.get("name", item.get("home_team", "")))
        team = _clean_team_name(team_raw) if team_raw else ""
        for bm in item.get("bookmakers", [item]):
            book = bm.get("title", bm.get("key", ""))
            for mkt in bm.get("markets", [bm]):
                if "win" not in str(mkt.get("key", "")).lower():
                    continue
                for outcome in mkt.get("outcomes", []):
                    if outcome.get("name", "").lower() == "over":
                        rows.append({
                            "team":       team,
                            "wins_line":  outcome.get("point"),
                            "over_odds":  outcome.get("price"),
                            "under_odds": next(
                                (o.get("price") for o in mkt.get("outcomes", [])
                                 if o.get("name", "").lower() == "under"),
                                None
                            ),
                            "book": book,
                        })

    df = pd.DataFrame(rows)
    # Keep best (highest) over line per team across books
    if not df.empty:
        df = (
            df.sort_values("wins_line", ascending=False)
            .drop_duplicates(subset=["team"])
            .reset_index(drop=True)
        )
    return df


def _get_endpoint(path):
    """Call an Owls Insight endpoint, return parsed JSON data dict or {}."""
    try:
        r = requests.get(f"{BASE_URL}/{path}", headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json().get("data", {})
        print(f"  ⚠️  Owls Insight {path}: HTTP {r.status_code}")
    except Exception as e:
        print(f"  ⚠️  Owls Insight {path} failed: {e}")
    return {}


def _flatten_by_game(data_dict):
    """Convert book-keyed dict → game_id keyed dict (merging bookmakers lists)."""
    games = {}
    for book_key, book_games in data_dict.items():
        for g in book_games:
            gid = g["id"]
            if gid not in games:
                games[gid] = {**g, "bookmakers": []}
            # Merge bookmakers
            games[gid]["bookmakers"].extend(g.get("bookmakers", []))
    return games


def print_current_lines():
    """Print a quick summary of available NCAAF lines."""
    df = fetch_ncaaf_lines_consensus()
    if df.empty:
        print("No NCAAF lines available right now.")
        return

    print(f"\nNCAAF Lines — {len(df)} games via Owls Insight")
    print(f"{'Home':<25} {'Away':<25} {'Spread':>8} {'O/U':>6}  {'Book'}")
    print("─" * 75)
    for _, r in df.iterrows():
        sp  = f"{r['spread']:+.1f}" if pd.notna(r.get('spread')) else "  —"
        ou  = f"{r['overUnder']:.1f}"  if pd.notna(r.get('overUnder')) else " —"
        print(f"{r['homeTeam']:<25} {r['awayTeam']:<25} {sp:>8} {ou:>6}  {r.get('spread_book','?')}")
