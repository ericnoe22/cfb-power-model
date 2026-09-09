"""
extended_sp_fetcher.py — fetches the fan-maintained "extended SP+" ratings
(FBS through D3/NAIA, ~770 teams) from a public Google Sheet.

Why: CFBD's SP+ only covers FBS. This sheet rates FCS/D2/D3/NAIA teams on
the same underlying scale, which lets model/fcs_adjustment.py predict FBS
vs FCS games against the specific opponent's strength instead of a
5-bucket historical tier average.

Source: https://docs.google.com/spreadsheets/d/1vwoVl-Dxy0es87Z9I1RTvFzr72Lb1fAkREfbLxbK-eg
        (gid 1878690143 = current-week all-levels ratings tab)
"""

import os
import sys
import requests
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CURRENT_SEASON

SHEET_ID  = "1vwoVl-Dxy0es87Z9I1RTvFzr72Lb1fAkREfbLxbK-eg"
SHEET_GID = "1878690143"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GID}"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")


def fetch_extended_sp_plus(year=CURRENT_SEASON, force_refresh=False) -> pd.DataFrame:
    """
    Fetch the extended (FBS-through-D3/NAIA) SP+ ratings, cache to
    cache/extended_sp_plus_{year}.csv. Returns DataFrame[team, ext_sp_plus].
    """
    cache_path = os.path.join(CACHE_DIR, f"extended_sp_plus_{year}.csv")

    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        if "team" in df.columns and "ext_sp_plus" in df.columns:
            return df

    try:
        r = requests.get(SHEET_URL, timeout=20)
        r.raise_for_status()
        from io import StringIO
        raw = pd.read_csv(StringIO(r.text))
    except Exception as e:
        print(f"  WARNING: Extended SP+ fetch failed: {e}")
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)
        return pd.DataFrame()

    if "Team" not in raw.columns or "SP+" not in raw.columns:
        print("  WARNING: Extended SP+ sheet format changed — expected 'Team'/'SP+' columns")
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)
        return pd.DataFrame()

    df = raw.rename(columns={"Team": "team", "SP+": "ext_sp_plus"})[["team", "ext_sp_plus"]]
    df = df.dropna(subset=["team", "ext_sp_plus"]).drop_duplicates(subset="team")

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  Extended SP+: {len(df)} teams fetched → {cache_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch extended (FBS-D3) SP+ ratings")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = fetch_extended_sp_plus(force_refresh=args.force)
    print(f"\n{len(df)} teams")
    print(df.head(15).to_string(index=False))
