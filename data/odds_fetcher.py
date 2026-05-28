"""
odds_fetcher.py — pulls current betting lines for upcoming games.

Two modes:
  1. The Odds API (preferred) — add your free key to config.py
  2. Manual entry — lines can be typed directly into the Streamlit dashboard

CFBD's /lines endpoint handles *historical* lines (completed games).
This module handles *current* lines for upcoming games.
"""

import requests
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import ODDS_API_KEY

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY     = "americanfootball_ncaaf"


def fetch_upcoming_lines():
    """
    Fetch current spreads, totals, and moneylines for upcoming CFB games
    via The Odds API. Returns a DataFrame, or empty DataFrame if no key set.
    """
    if not ODDS_API_KEY:
        print("ℹ️  No Odds API key set. Use manual line entry in the dashboard.")
        print("   Get a free key at: https://the-odds-api.com")
        return pd.DataFrame()

    try:
        url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "spreads,totals,h2h",
            "oddsFormat": "american",
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"⚠️  Odds API error {resp.status_code}: {resp.text[:100]}")
            return pd.DataFrame()

        data = resp.json()
        rows = []
        for game in data:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            commence = game.get("commence_time", "")

            for book in game.get("bookmakers", []):
                book_key = book.get("key", "")
                if book_key not in ("fanduel", "draftkings", "betmgm", "consensus"):
                    continue
                for market in book.get("markets", []):
                    mtype = market.get("key")
                    if mtype == "spreads":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home:
                                rows.append({
                                    "homeTeam": home, "awayTeam": away,
                                    "commence_time": commence,
                                    "book": book_key,
                                    "spread": outcome.get("point"),
                                    "spread_juice": outcome.get("price"),
                                    "overUnder": None,
                                })
                    elif mtype == "totals":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == "Over":
                                rows.append({
                                    "homeTeam": home, "awayTeam": away,
                                    "commence_time": commence,
                                    "book": book_key,
                                    "spread": None,
                                    "overUnder": outcome.get("point"),
                                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Consensus: average across books
        consensus = (
            df.groupby(["homeTeam", "awayTeam", "commence_time"])
            .agg({"spread": "mean", "overUnder": "mean"})
            .reset_index()
        )
        consensus["book"] = "consensus"
        return consensus

    except Exception as e:
        print(f"⚠️  Odds fetch failed: {e}")
        return pd.DataFrame()


def parse_manual_lines(lines_input):
    """
    Parse a list of manually entered lines into a standard DataFrame.

    lines_input: list of dicts like:
        [{"homeTeam": "Alabama", "awayTeam": "Georgia",
          "spread": -7.0, "overUnder": 52.5}]
    """
    return pd.DataFrame(lines_input)
