"""
CFB Power Model REST API
Run with: uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path so model/data imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CURRENT_SEASON,
    EDGE_THRESHOLD_SPREAD,
    EDGE_THRESHOLD_TOTAL,
    RATING_WEIGHTS,
)
from model.power_rankings import (
    load_prebuilt_ratings,
    _normalize_elo,
    _normalize_returning,
    _normalize_talent,
    z_score,
)
from model.game_predictor import predict_all_games
from model.edge_finder import find_edges, summarize_edges

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CFB Power Model API",
    version="1.0.0",
    description=(
        "College football power ratings, game predictions, and betting edges. "
        f"Current season: {CURRENT_SEASON}."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Data helpers ─────────────────────────────────────────────────────────────

def _records(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-safe list of dicts (NaN → null)."""
    return json.loads(df.to_json(orient="records"))


def _compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite rating from a pre-merged ratings DataFrame.
    Mirrors the _compute_composite logic in app.py, without Streamlit.
    """
    w = RATING_WEIGHTS.copy()
    df = df.copy()

    df["sp_plus_norm"] = (
        df["sp_plus"].fillna(df["sp_plus"].mean())
        if "sp_plus" in df.columns else 0
    )
    df["fpi_norm"] = (
        df["fpi"].fillna(df["fpi"].mean())
        if "fpi" in df.columns and df["fpi"].notna().any()
        else df["sp_plus_norm"]
    )
    df["elo_norm"] = (
        _normalize_elo(df["elo"].fillna(1500))
        if "elo" in df.columns else 0
    )
    df["returning_norm"] = (
        _normalize_returning(df["returning_prod"].fillna(0.55))
        if "returning_prod" in df.columns else 0
    )
    df["talent_norm"] = (
        _normalize_talent(df["talent"].fillna(df["talent"].mean()))
        if "talent" in df.columns and df["talent"].notna().any() else 0
    )

    df["composite"] = (
        w["sp_plus"]        * df["sp_plus_norm"]   +
        w["fpi"]            * df["fpi_norm"]        +
        w["elo"]            * df["elo_norm"]        +
        w["returning_prod"] * df["returning_norm"]  +
        w["talent"]         * df["talent_norm"]
    )

    df = df.drop(columns=["rank"], errors="ignore")
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    return df.reset_index()


@lru_cache(maxsize=1)
def _ratings() -> pd.DataFrame:
    path = os.path.join(BASE_DIR, f"{CURRENT_SEASON}_power_rating_cleaned.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = load_prebuilt_ratings(path)
    df = _compute_composite(df)

    # Merge SP+ offense/defense breakdowns for accurate total predictions
    sp_path = os.path.join(BASE_DIR, f"cache/sp_plus_{CURRENT_SEASON}.csv")
    if os.path.exists(sp_path):
        try:
            sp = pd.read_csv(sp_path)
            if {"offense.rating", "defense.rating", "team"}.issubset(sp.columns):
                from data.team_names import normalize as _norm
                sp["team"] = sp["team"].map(_norm)
                df["team"] = df["team"].map(_norm)
                df = df.merge(
                    sp[["team", "offense.rating", "defense.rating"]],
                    on="team", how="left",
                )
        except Exception:
            pass
    return df


@lru_cache(maxsize=1)
def _schedule() -> pd.DataFrame:
    path = os.path.join(BASE_DIR, f"{CURRENT_SEASON}_schedule_with_power.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.drop_duplicates(subset=["homeTeam", "awayTeam", "week"])


def _live_lines() -> pd.DataFrame:
    cache_path = os.path.join(BASE_DIR, "cache/lines_live.csv")
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass
    return pd.DataFrame()


def _conf_map() -> dict:
    sched = _schedule()
    if sched.empty:
        return {}
    m = {}
    for _, row in sched[["homeTeam", "homeConference"]].dropna().iterrows():
        m[row["homeTeam"]] = row["homeConference"]
    for _, row in sched[["awayTeam", "awayConference"]].dropna().iterrows():
        m[row["awayTeam"]] = row["awayConference"]
    return m


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check."""
    ratings_ok = os.path.exists(
        os.path.join(BASE_DIR, f"{CURRENT_SEASON}_power_rating_cleaned.csv")
    )
    schedule_ok = os.path.exists(
        os.path.join(BASE_DIR, f"{CURRENT_SEASON}_schedule_with_power.csv")
    )
    lines_ok = os.path.exists(os.path.join(BASE_DIR, "cache/lines_live.csv"))
    return {
        "status": "ok",
        "season": CURRENT_SEASON,
        "data": {
            "ratings": ratings_ok,
            "schedule": schedule_ok,
            "live_lines": lines_ok,
        },
    }


@app.get("/ratings")
def ratings(
    conference: Optional[str] = Query(None, description="Filter by conference name"),
    limit: int = Query(134, ge=1, le=134, description="Max teams to return"),
):
    """
    Power rankings for all FBS teams, sorted by composite rating.
    Includes SP+, FPI, Elo, returning production, and talent components.
    """
    df = _ratings()
    if df.empty:
        raise HTTPException(status_code=503, detail="Ratings data not available.")

    df = df.copy()
    df["conference"] = df["team"].map(_conf_map()).fillna("Independent")

    if conference:
        df = df[df["conference"].str.lower() == conference.lower()]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No teams found for conference: {conference}")

    cols = [c for c in [
        "rank", "team", "composite", "conference",
        "sp_plus", "fpi", "elo", "returning_prod", "talent",
        "offense.rating", "defense.rating",
    ] if c in df.columns]

    return {
        "season": CURRENT_SEASON,
        "count": min(len(df), limit),
        "ratings": _records(df[cols].head(limit)),
    }


@app.get("/teams")
def teams(
    team: str = Query(..., description="Team name to look up"),
):
    """
    Full ratings breakdown for a single team.
    """
    df = _ratings()
    if df.empty:
        raise HTTPException(status_code=503, detail="Ratings data not available.")

    from data.team_names import normalize
    norm = normalize(team)
    row = df[df["team"] == norm]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Team not found: {team}")

    conf = _conf_map().get(norm, "Independent")
    record = _records(row)[0]
    record["conference"] = conf
    return record


@app.get("/schedule")
def schedule(
    week: Optional[int] = Query(None, ge=1, le=20, description="Week number (1-20)"),
    team: Optional[str] = Query(None, description="Filter by team name (home or away)"),
    predictions: bool = Query(True, description="Include model spread/total predictions"),
    lines: bool = Query(True, description="Include live Vegas lines"),
):
    """
    Season schedule with optional model predictions and live Vegas lines.
    Returns all games if week is omitted (may be slow — prefer specifying a week).
    """
    sched = _schedule()
    if sched.empty:
        raise HTTPException(status_code=503, detail="Schedule data not available.")

    df = sched.copy()

    if week is not None:
        df = df[df["week"] == week]
    if team:
        from data.team_names import normalize
        norm = normalize(team)
        mask = (
            df["homeTeam"].map(normalize) == norm
        ) | (
            df["awayTeam"].map(normalize) == norm
        )
        df = df[mask]

    if df.empty:
        return {"season": CURRENT_SEASON, "week": week, "count": 0, "games": []}

    if predictions and not _ratings().empty:
        df = predict_all_games(df, _ratings())

    if lines:
        live = _live_lines()
        if not live.empty:
            line_cols = [c for c in [
                "homeTeam", "awayTeam", "spread", "overUnder", "home_ml", "away_ml"
            ] if c in live.columns]
            live_clean = live[line_cols].rename(columns={
                "spread": "vegas_spread", "overUnder": "vegas_total"
            })
            df = df.merge(live_clean, on=["homeTeam", "awayTeam"], how="left")

    output_cols = [c for c in [
        "week", "homeTeam", "awayTeam", "neutralSite",
        "homeConference", "awayConference",
        "predicted_spread", "predicted_total",
        "home_composite", "away_composite",
        "vegas_spread", "vegas_total", "home_ml", "away_ml",
        "confidence", "fcs_note",
        "homePoints", "awayPoints",
    ] if c in df.columns]

    return {
        "season": CURRENT_SEASON,
        "week": week,
        "count": len(df),
        "games": _records(df[output_cols]),
    }


@app.get("/edges")
def edges(
    week: Optional[int] = Query(None, ge=1, le=20, description="Week number"),
    edges_only: bool = Query(True, description="Only return games with actionable edges (grade B or better)"),
):
    """
    Betting edges — games where the model disagrees with Vegas by more than the threshold.
    Requires live lines to be available (offseason returns empty list).
    """
    ratings_df = _ratings()
    if ratings_df.empty:
        raise HTTPException(status_code=503, detail="Ratings data not available.")

    live = _live_lines()
    if live.empty:
        return {
            "season": CURRENT_SEASON,
            "week": week,
            "count": 0,
            "edges": [],
            "message": "No live lines available — may be the offseason.",
        }

    df = live.copy()
    if week is not None and "week" in df.columns:
        df = df[df["week"] == week]

    if df.empty:
        return {"season": CURRENT_SEASON, "week": week, "count": 0, "edges": []}

    neutral_col = next(
        (c for c in ["neutralSite_y", "neutralSite"] if c in df.columns), None
    )
    live_sched = df[["homeTeam", "awayTeam"]].copy()
    live_sched["neutralSite"] = df[neutral_col] if neutral_col else False

    predicted = predict_all_games(live_sched, ratings_df)

    lines_for_edge = df[["homeTeam", "awayTeam", "spread", "overUnder"]].rename(
        columns={"spread": "vegas_spread", "overUnder": "vegas_total"}
    )

    edge_df = find_edges(predicted, lines_for_edge)

    if "week" in df.columns:
        week_map = dict(zip(
            zip(df["homeTeam"], df["awayTeam"]), df["week"]
        ))
        edge_df["week"] = edge_df.apply(
            lambda r: week_map.get((r.get("homeTeam"), r.get("awayTeam"))), axis=1
        )

    if edges_only:
        edge_df = summarize_edges(edge_df)

    output_cols = [c for c in [
        "week", "homeTeam", "awayTeam",
        "predicted_spread", "vegas_spread", "edge_spread", "bet_spread",
        "predicted_total", "vegas_total", "edge_total", "bet_total",
        "edge_grade", "confidence",
        "home_composite", "away_composite",
    ] if c in edge_df.columns]

    return {
        "season": CURRENT_SEASON,
        "week": week,
        "spread_threshold": EDGE_THRESHOLD_SPREAD,
        "total_threshold": EDGE_THRESHOLD_TOTAL,
        "count": len(edge_df),
        "edges": _records(edge_df[output_cols]),
    }


@app.get("/synopsis")
def synopsis(
    home: str = Query(..., description="Home team name"),
    away: str = Query(..., description="Away team name"),
    week: Optional[int] = Query(None, ge=1, le=20, description="Week number (used for cache key)"),
    force: bool = Query(False, description="Bypass cache and regenerate"),
):
    """
    AI-generated game preview using Claude.
    Requires ANTHROPIC_API_KEY to be set in environment / .env.
    Results are cached per week so repeated calls are free.
    """
    from model.synopsis_generator import generate_synopsis

    ratings_df = _ratings()

    def _rating(team: str, col: str):
        if ratings_df.empty:
            return None
        r = ratings_df[ratings_df["team"] == team]
        if r.empty or col not in r.columns:
            return None
        v = r[col].values[0]
        return None if pd.isna(v) else round(float(v), 2)

    from data.team_names import normalize
    home_norm = normalize(home)
    away_norm = normalize(away)

    game_data = {
        "homeTeam": home_norm,
        "awayTeam": away_norm,
        "home_composite": _rating(home_norm, "composite"),
        "away_composite": _rating(away_norm, "composite"),
        "home_off_rating": _rating(home_norm, "offense.rating"),
        "away_off_rating": _rating(away_norm, "offense.rating"),
        "home_def_rating": _rating(home_norm, "defense.rating"),
        "away_def_rating": _rating(away_norm, "defense.rating"),
        "week": week,
    }

    result = generate_synopsis(game_data, week=week, force=force)

    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Synopsis unavailable — set ANTHROPIC_API_KEY in .env or environment.",
        )

    return {
        "home": home_norm,
        "away": away_norm,
        "week": week,
        "synopsis": result,
    }


@app.get("/teams/profiles")
def team_profiles(
    team: Optional[str] = Query(None, description="Team name to look up a single profile"),
):
    """
    Team profiles: coaching staff, recruting tier, notes, and coordinator info.
    Returns all profiles if no team name is given.
    """
    profiles_path = os.path.join(BASE_DIR, f"cache/team_profiles_{CURRENT_SEASON}.json")
    if not os.path.exists(profiles_path):
        raise HTTPException(status_code=503, detail="Team profiles not available.")

    with open(profiles_path) as f:
        profiles: dict = json.load(f)

    if team:
        from data.team_names import normalize
        norm = normalize(team)
        # Try exact key first, then normalized match
        profile = profiles.get(norm) or next(
            (v for k, v in profiles.items() if normalize(k) == norm), None
        )
        if profile is None:
            raise HTTPException(status_code=404, detail=f"Profile not found: {team}")
        return {"team": norm, "profile": profile}

    return {
        "season": CURRENT_SEASON,
        "count": len(profiles),
        "profiles": profiles,
    }


@app.get("/picks/stats")
def picks_stats():
    """
    Aggregate performance stats for all logged picks: win %, ROI, avg CLV, by grade.
    """
    from model.clv_tracker import compute_stats
    stats = compute_stats()
    return stats


@app.get("/picks/log")
def picks_log(
    pending_only: bool = Query(False, description="Only return picks with no result yet"),
    bet_type: Optional[str] = Query(None, description="Filter by 'spread' or 'total'"),
    grade: Optional[str] = Query(None, description="Filter by edge grade (A+, A, B, C)"),
):
    """
    Full pick log with CLV and results for settled bets.
    """
    from model.clv_tracker import load_log
    df = load_log()
    if df.empty:
        return {"count": 0, "picks": []}

    if pending_only:
        df = df[df["won"].isna() | (df["won"] == "None")]
    if bet_type:
        df = df[df["bet_type"] == bet_type]
    if grade:
        df = df[df["edge_grade"] == grade]

    return {"count": len(df), "picks": _records(df)}
