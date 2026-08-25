"""
sagarin_fetcher.py — fetches Jeff Sagarin's college football ratings.

Source: http://sagarin.com/sports/cfsend.htm (updated weekly during the season).

Columns returned:
  team            — CFBD canonical team name
  sagarin_rating  — composite rating (predictor + schedule-adjusted performance)
  predictor       — pure efficiency predictor (best for spread prediction)
  sagarin_rank    — overall rank

Scale: ~40 (weak FCS) to ~105 (elite FBS), FBS average ~75.
Normalized to SP+ scale in power_rankings.py via z-score.
"""

import os
import re
import json
import sys
import requests
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CURRENT_SEASON

SAGARIN_URL = "http://sagarin.com/sports/cfsend.htm"
CACHE_DIR   = os.path.join(os.path.dirname(__file__), "..", "cache")

# Sagarin uses full/alternate names — map to CFBD canonical names
SAGARIN_NAME_MAP = {
    "Miami-Florida":          "Miami",
    "Southern California":    "USC",
    "Mississippi":            "Ole Miss",
    "Central Florida(UCF)":   "UCF",
    "Army West Point":        "Army",
    "Miami-Ohio":             "Miami (OH)",
    "Connecticut":            "UConn",
    "Louisiana-Lafayette":    "Louisiana",
    "Fla. International":     "Florida International",
    "LouisianaMonroe(ULM)":   "UL Monroe",
    # Additional variants seen historically
    "Sam Houston State":      "Sam Houston",
    "Brigham Young":          "BYU",
    "Hawai`i":                "Hawai'i",
    "Hawaii":                 "Hawai'i",
    "Nevada-Las Vegas":       "UNLV",
    "San Jose State":         "San José State",
    "Middle Tenn. State":     "Middle Tennessee",
    "Fla. Atlantic":          "Florida Atlantic",
    "So. Mississippi":        "Southern Miss",
    "Ga. Southern":           "Georgia Southern",
    "Appalachian State":      "App State",
    "UT-San Antonio":         "UTSA",
    "Tex.-San Antonio":       "UTSA",
    "N. Illinois":            "Northern Illinois",
    "W. Kentucky":            "Western Kentucky",
    "W. Michigan":            "Western Michigan",
    "E. Michigan":            "Eastern Michigan",
    "C. Michigan":            "Central Michigan",
    "Jacksonville St.":       "Jacksonville State",
    "Sam Hous. State":        "Sam Houston",
    "Kennesaw State":         "Kennesaw State",
}


def _normalize_sagarin_name(raw: str) -> str:
    """Map a Sagarin team name to its CFBD canonical name."""
    name = raw.strip()
    if name in SAGARIN_NAME_MAP:
        return SAGARIN_NAME_MAP[name]
    return name


def _parse_ratings(html: str) -> pd.DataFrame:
    """
    Parse the fixed-width Sagarin ratings block from the page HTML.

    Line format (preseason — all pipe-values equal):
      rank  team_name(20)  grade  =  rating  W  L  schedl(rank)  vs10  |  vs30  |  predictor rank  |  golden_mean rank  |  recent rank  |  strong_recent rank  CONF
    """
    # Pattern: leading spaces + rank + 2 spaces + 20-char name + grade + = + rating
    pattern = re.compile(
        r'^\s{1,3}(\d+)\s{2}(.{20})\s+([A-Z])\s+=\s+([\d.]+)\s+(.*)',
    )

    rows = []
    seen = set()

    for line in html.split('\n'):
        m = pattern.match(line)
        if not m:
            continue

        rank   = int(m.group(1))
        name   = m.group(2).strip()
        grade  = m.group(3)
        rating = float(m.group(4))

        if name in seen:
            continue
        seen.add(name)

        # Pipe layout: ... vs10 | vs30 | PREDICTOR rank | GOLDEN_MEAN rank | ...
        # predictor is in pipes[2] (0-indexed), golden_mean in pipes[3]
        predictor = rating  # default (identical preseason)
        pipes = line.split('|')
        if len(pipes) >= 3:
            pm = re.search(r'([\d.]+)', pipes[2])
            if pm:
                predictor = float(pm.group(1))

        cfbd_name = _normalize_sagarin_name(name)
        rows.append({
            'team':           cfbd_name,
            'sagarin_name':   name,
            'sagarin_grade':  grade,
            'sagarin_rating': rating,
            'predictor':      predictor,
            'sagarin_rank':   rank,
        })

    return pd.DataFrame(rows)


def fetch_sagarin(year=CURRENT_SEASON, force_refresh=False) -> pd.DataFrame:
    """
    Fetch Sagarin ratings, cache to cache/sagarin_{year}.json.
    Returns DataFrame with team, sagarin_rating, predictor, sagarin_rank.
    """
    cache_path = os.path.join(CACHE_DIR, f"sagarin_{year}.json")

    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        return pd.DataFrame(data)

    try:
        r = requests.get(SAGARIN_URL, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        print(f"  WARNING: Sagarin fetch failed: {e}")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return pd.DataFrame(json.load(f))
        return pd.DataFrame()

    df = _parse_ratings(html)
    if df.empty:
        print("  WARNING: Sagarin parse returned empty DataFrame")
        return df

    # Only keep FBS teams (grade A)
    df = df[df['sagarin_grade'] == 'A'].copy()
    df = df.drop(columns=['sagarin_grade'])

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2)

    print(f"  Sagarin: {len(df)} FBS teams fetched → {cache_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Sagarin CFB ratings")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = fetch_sagarin(force_refresh=args.force)
    print(f"\n{len(df)} teams")
    print(df.sort_values('sagarin_rank').head(25)[
        ['sagarin_rank', 'team', 'sagarin_rating', 'predictor']
    ].to_string(index=False))
