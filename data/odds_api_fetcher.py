"""
odds_api_fetcher.py — Pulls supplementary data from The Odds API.

Used for:
  - NCAAF National Championship winner odds (season projections page)

Not used for game lines — Owls Insight handles that.
Docs: https://the-odds-api.com/liveapi/guides/v4/
"""

import os
import sys
import requests
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import ODDS_API_KEY
from data.owls_fetcher import _clean_team_name as _clean

BASE_URL    = "https://api.the-odds-api.com/v4"
BOOK_PRIO   = ["draftkings", "fanduel", "betmgm", "caesars", "pinnacle"]

# Pre-normalization overrides for names that are ambiguous after mascot stripping.
# Applied BEFORE _clean so the right school name survives.
_PRE_OVERRIDES = {
    "Miami Hurricanes":           "Miami",
    "Miami (FL)":                 "Miami",
    "Miami Florida":              "Miami",
    "Miami Ohio":                 "Miami (OH)",
    "Miami (Ohio)":               "Miami (OH)",
    "Miami (OH) RedHawks":        "Miami (OH)",
    "Miami RedHawks":             "Miami (OH)",
    "Miami Red Hawks":            "Miami (OH)",
    # Odds API occasionally appends mascots not in our strip list
    "Sacramento State Hornets":   "Sacramento State",
    "UMass Minutemen":            "Massachusetts",
    "Sam Houston State":              "Sam Houston",
    "Sam Houston State Bearkats":     "Sam Houston",
    "UT San Antonio Roadrunners": "UTSA",
    "Louisiana Monroe Warhawks":  "UL Monroe",
}

def _normalize_team(name: str) -> str:
    """Normalize an Odds API team name to our canonical CFBD name."""
    if name in _PRE_OVERRIDES:
        return _PRE_OVERRIDES[name]
    return _clean(name)


def _get(path, params=None):
    """GET a The Odds API endpoint, return parsed JSON or None."""
    params = params or {}
    params["apiKey"] = ODDS_API_KEY
    try:
        r = requests.get(f"{BASE_URL}/{path}", params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        print(f"  ⚠️  Odds API {path}: HTTP {r.status_code} — {r.text[:200]}")
    except Exception as e:
        print(f"  ⚠️  Odds API {path} failed: {e}")
    return None


def fetch_ncaaf_championship_odds() -> pd.DataFrame:
    """
    Fetch current NCAAF national championship winner odds from The Odds API.

    Returns a DataFrame with one row per team:
        team, best_odds, best_book, dk_odds, fd_odds, betmgm_odds, caesars_odds
    All odds are American format (+550, -110, etc.).
    Returns empty DataFrame if unavailable.
    """
    data = _get(
        "sports/americanfootball_ncaaf_championship_winner/odds/",
        params={
            "regions":    "us",
            "markets":    "outrights",
            "bookmakers": ",".join(BOOK_PRIO),
            "oddsFormat": "american",
        },
    )
    if not data or not isinstance(data, list):
        return pd.DataFrame()

    # data is a list with a single event (the championship)
    event = data[0]
    book_odds: dict[str, dict[str, int]] = {}  # book_key → {team → odds}

    for bm in event.get("bookmakers", []):
        book_key = bm["key"]
        for mkt in bm.get("markets", []):
            if mkt["key"] != "outrights":
                continue
            book_odds[book_key] = {
                _normalize_team(o["name"]): int(o["price"])
                for o in mkt.get("outcomes", [])
            }

    if not book_odds:
        return pd.DataFrame()

    # Collect all teams across all books
    all_teams = sorted(set(t for odds in book_odds.values() for t in odds))

    rows = []
    for team in all_teams:
        row = {"team": team}
        available = []
        for book in BOOK_PRIO:
            odds_val = book_odds.get(book, {}).get(team)
            col = {"draftkings": "dk_odds", "fanduel": "fd_odds",
                   "betmgm": "betmgm_odds", "caesars": "caesars_odds"}.get(book)
            if col:
                row[col] = odds_val
            if odds_val is not None:
                available.append((odds_val, book))

        if available:
            # Best odds = highest payout (max American odds value)
            best_val, best_book = max(available, key=lambda x: x[0])
            row["best_odds"] = best_val
            row["best_book"] = best_book
        else:
            row["best_odds"] = None
            row["best_book"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "best_odds" in df.columns:
        df = df.sort_values("best_odds").reset_index(drop=True)  # lowest odds = favorite
    return df


def fetch_ncaaf_game_lines() -> pd.DataFrame:
    """
    Fetch current NCAAF game spreads, totals, and moneylines from The Odds API.

    Returns a DataFrame with one row per game (consensus: DraftKings if available,
    else best book), columns: homeTeam, awayTeam, spread, overUnder, home_ml,
    away_ml, commence_time, books_available.
    Returns empty DataFrame if unavailable.
    """
    data = _get(
        "sports/americanfootball_ncaaf/odds/",
        params={
            "regions":    "us",
            "markets":    "spreads,totals,h2h",
            "bookmakers": ",".join(BOOK_PRIO),
            "oddsFormat": "american",
        },
    )
    if not data or not isinstance(data, list):
        return pd.DataFrame()

    rows = []
    for event in data:
        home_raw = event.get("home_team", "")
        away_raw = event.get("away_team", "")
        home = _normalize_team(home_raw)
        away = _normalize_team(away_raw)
        commence = event.get("commence_time", "")

        book_map = {bm["key"]: bm for bm in event.get("bookmakers", [])}

        spread = over_under = home_ml = away_ml = None
        spread_book = total_book = ml_book = None

        for book in BOOK_PRIO:
            bm = book_map.get(book)
            if not bm:
                continue
            for mkt in bm.get("markets", []):
                if mkt["key"] == "spreads" and spread is None:
                    for o in mkt["outcomes"]:
                        if _normalize_team(o["name"]) == home:
                            spread = float(o["point"])
                            spread_book = bm["title"]
                            break
                elif mkt["key"] == "totals" and over_under is None:
                    for o in mkt["outcomes"]:
                        if o["name"] == "Over":
                            over_under = float(o["point"])
                            total_book = bm["title"]
                            break
                elif mkt["key"] == "h2h" and home_ml is None:
                    for o in mkt["outcomes"]:
                        if _normalize_team(o["name"]) == home:
                            home_ml = int(o["price"])
                        else:
                            away_ml = int(o["price"])
                    if home_ml is not None:
                        ml_book = bm["title"]

        if spread is None and over_under is None:
            continue  # skip games with no lines at all

        rows.append({
            "homeTeam":      home,
            "awayTeam":      away,
            "commence_time": commence,
            "spread":        spread,
            "overUnder":     over_under,
            "home_ml":       home_ml,
            "away_ml":       away_ml,
            "spread_book":   spread_book,
            "total_book":    total_book,
            "books_available": len(book_map),
        })

    df = pd.DataFrame(rows)
    if not df.empty and "commence_time" in df.columns:
        df["commence_dt"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
        df = df.sort_values("commence_dt").reset_index(drop=True)
    return df


def fmt_american_odds(v) -> str:
    """Format an American odds integer for display: +550, -110, etc."""
    try:
        v = int(v)
        return f"+{v}" if v > 0 else str(v)
    except (TypeError, ValueError):
        return "—"
