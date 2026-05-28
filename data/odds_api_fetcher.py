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
from data.team_names import normalize

BASE_URL    = "https://api.the-odds-api.com/v4"
BOOK_PRIO   = ["draftkings", "fanduel", "betmgm", "caesars", "pinnacle"]


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


def _normalize_team(raw: str) -> str:
    """Strip mascot from 'School Mascot' format and normalize."""
    # Simple approach: remove last word(s) if they look like a mascot.
    # Falls back to normalize() which handles known CFBD variants.
    parts = raw.strip().split()
    # Try progressively shorter names until normalize returns something clean
    for n in range(len(parts) - 1, 0, -1):
        candidate = " ".join(parts[:n])
        normed = normalize(candidate)
        if normed != candidate.lower():   # normalize changed it — good match
            return normed
    return normalize(raw)


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


def fmt_american_odds(v) -> str:
    """Format an American odds integer for display: +550, -110, etc."""
    try:
        v = int(v)
        return f"+{v}" if v > 0 else str(v)
    except (TypeError, ValueError):
        return "—"
