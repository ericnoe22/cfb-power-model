"""
power_rankings.py — builds the composite power rating for every FBS team.

The composite is expressed in the same unit as SP+:
  "predicted points better/worse than an average FBS team per game"

Sources blended (weights configurable in config.py, see RATING_WEIGHTS):
  - SP+, FPI, Sagarin: efficiency/power-index ratings, each opponent-adjusted
  - Elo: form-based, updated after each result (partly ceded to epa_adj in-season)
  - Returning production, talent: preseason-only signals, fade in-season
  - Defensive havoc rate: disruption/aggression, distinct from PPA efficiency
  - Opponent-adjusted EPA/PPA: ramped in as real per-play data accumulates
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RATING_WEIGHTS, CURRENT_SEASON


# ── Normalization helpers ──────────────────────────────────────────────────

def z_score(series):
    """Standardize to mean=0, std=1, ignoring NaN. Returns 0 if no variation."""
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _normalize_elo(elo_series):
    """
    Convert Elo (avg ~1500, range ~1000–2000) to SP+-like point scale.
    Each 30 Elo points ≈ 1 point better per game (rough empirical calibration).
    """
    centered = elo_series - 1500
    return centered / 30


def _normalize_returning(rp_series):
    """
    Returning production (0–1). League average ~0.55.
    Scale so that going from 0.40 → 0.70 is roughly ±3 SP+ points.
    """
    centered = rp_series - rp_series.mean()
    return centered * 20   # amplify so meaningful but not dominant


def _normalize_talent(talent_series):
    """
    Talent composite (247Sports, ~280–1020). Convert to SP+ scale.
    """
    return z_score(talent_series) * 5   # 1 SD ≈ 5 SP+ points


def _normalize_sagarin(sagarin_series):
    """
    Sagarin Predictor (~40–105 for FBS, mean ~75). Convert to SP+ scale.
    1 SD ≈ 8 SP+ points — Sagarin spreads over a wider range than SP+.
    """
    return z_score(sagarin_series) * 8


def _normalize_havoc(havoc_series):
    """
    Defensive havoc rate (fraction of opponent plays with a TFL, forced
    fumble, INT, or pass breakup — higher = more disruptive defense).
    Convert to SP+ scale; kept modest since it's already ~0.70 correlated
    with opponent-adjusted defensive PPA and shouldn't double up on it.
    """
    return z_score(havoc_series) * 5   # 1 SD ≈ 5 SP+ points


# ── Main builder ───────────────────────────────────────────────────────────

def build_composite_ratings(
    sp_df=None,
    fpi_df=None,
    elo_df=None,
    returning_df=None,
    talent_df=None,
    epa_df=None,
    sagarin_df=None,
    havoc_df=None,
    week=None,
    season=CURRENT_SEASON,
    apply_coaching=True,
):
    """
    Merge all rating sources and compute a weighted composite.

    Each DataFrame should have at least a 'team' column.
    Pass week= to adjust the returning-production weight mid-season
    (it matters less once we have actual game data).

    Returns a DataFrame indexed by team with all ratings + composite.
    """
    weights = RATING_WEIGHTS.copy()

    # After week 4, shift returning-production weight to Elo (actual results matter more)
    if week and week >= 4:
        shift = min(weights["returning_prod"] * 0.5, 0.05)  # move up to half of RP weight
        weights["returning_prod"] -= shift
        weights["elo"] += shift

    # ── Start from SP+ as the backbone ──────────────────────────────────
    if sp_df is not None and not sp_df.empty:
        base = sp_df[["team", "rating"]].rename(columns={"rating": "sp_plus"}).copy()
    else:
        return pd.DataFrame()

    # ── Merge FPI ────────────────────────────────────────────────────────
    if fpi_df is not None and not fpi_df.empty:
        fpi_cols = [c for c in fpi_df.columns if "team" in c.lower() or "fpi" in c.lower()]
        fpi_clean = fpi_df[["team", "fpi"]].copy() if "fpi" in fpi_df.columns else None
        if fpi_clean is not None:
            base = base.merge(fpi_clean, on="team", how="left")
        else:
            base["fpi"] = np.nan
    else:
        base["fpi"] = np.nan

    # ── Merge Elo ────────────────────────────────────────────────────────
    if elo_df is not None and not elo_df.empty:
        elo_clean = elo_df[["team", "elo"]].copy() if "elo" in elo_df.columns else None
        if elo_clean is not None:
            base = base.merge(elo_clean, on="team", how="left")
        else:
            base["elo"] = 1500.0
    else:
        base["elo"] = 1500.0

    # ── Merge returning production ────────────────────────────────────────
    if returning_df is not None and not returning_df.empty:
        # Find the right source column (already renamed, or raw from API)
        src_col = None
        for candidate in ["returning_prod", "percentPPA", "usage"]:
            if candidate in returning_df.columns:
                src_col = candidate
                break
        if src_col:
            rp_clean = returning_df[["team", src_col]].rename(
                columns={src_col: "returning_prod"}).copy()
            base = base.merge(rp_clean, on="team", how="left")
        else:
            base["returning_prod"] = np.nan
    else:
        base["returning_prod"] = np.nan

    # ── Merge talent ──────────────────────────────────────────────────────
    if talent_df is not None and not talent_df.empty:
        talent_cols = [c for c in talent_df.columns if "talent" in c.lower()]
        if talent_cols:
            # CFBD returns 'school' in some years, 'team' in others
            name_col = "school" if "school" in talent_df.columns else "team"
            base = base.merge(
                talent_df[[name_col, talent_cols[0]]].rename(
                    columns={name_col: "team", talent_cols[0]: "talent"}),
                on="team", how="left"
            )
        elif "talent" in talent_df.columns:
            base = base.merge(talent_df[["team", "talent"]], on="team", how="left")
        else:
            base["talent"] = np.nan
    else:
        base["talent"] = np.nan

    # ── Merge opponent-adjusted EPA/PPA ───────────────────────────────────
    # PPA: offense_overall (higher = better), defense_overall (lower = better)
    # Net PPA = offense - (-defense) = offense + defense_for_opponent_perspective
    if epa_df is not None and not epa_df.empty:
        epa_cols = [c for c in epa_df.columns if "team" in c.lower()]
        name_col = epa_cols[0] if epa_cols else "team"
        if "offense_overall" in epa_df.columns and "defense_overall" in epa_df.columns:
            epa_clean = epa_df[[name_col, "offense_overall", "defense_overall"]].rename(
                columns={name_col: "team"}
            ).copy()
            # Net = offense - defense (defense_overall is negative = good defense)
            epa_clean["epa_net"] = epa_clean["offense_overall"] - epa_clean["defense_overall"]
            base = base.merge(epa_clean[["team", "offense_overall", "defense_overall", "epa_net"]],
                              on="team", how="left")
        elif "epa_per_play_off" in epa_df.columns and "epa_per_play_def" in epa_df.columns:
            epa_clean = epa_df[["team", "epa_per_play_off", "epa_per_play_def"]].copy()
            epa_clean["epa_net"] = epa_clean["epa_per_play_off"] - epa_clean["epa_per_play_def"]
            base = base.merge(epa_clean[["team", "epa_per_play_off", "epa_per_play_def", "epa_net"]],
                              on="team", how="left")
        else:
            base["epa_net"] = np.nan
    else:
        base["epa_net"] = np.nan

    # ── Merge Sagarin ─────────────────────────────────────────────────────
    if sagarin_df is not None and not sagarin_df.empty:
        sag_col = "predictor" if "predictor" in sagarin_df.columns else \
                  "sagarin_rating" if "sagarin_rating" in sagarin_df.columns else None
        if sag_col and "team" in sagarin_df.columns:
            base = base.merge(
                sagarin_df[["team", sag_col]].rename(columns={sag_col: "sagarin"}),
                on="team", how="left"
            )
        else:
            base["sagarin"] = np.nan
    else:
        base["sagarin"] = np.nan

    # ── Merge defensive havoc rate ────────────────────────────────────────
    if havoc_df is not None and not havoc_df.empty and "havoc_total" in havoc_df.columns:
        base = base.merge(havoc_df[["team", "havoc_total"]], on="team", how="left")
    else:
        base["havoc_total"] = np.nan

    # ── Normalize each metric to SP+ scale ───────────────────────────────
    base["sp_plus_norm"]    = base["sp_plus"].fillna(base["sp_plus"].mean())
    base["fpi_norm"]        = base["fpi"].fillna(base["fpi"].mean()) if base["fpi"].notna().any() \
                              else base["sp_plus_norm"]
    base["elo_norm"]        = _normalize_elo(base["elo"].fillna(1500))
    base["returning_norm"]  = _normalize_returning(base["returning_prod"].fillna(
                              base["returning_prod"].mean() if base["returning_prod"].notna().any() else 0.55))
    base["talent_norm"]     = _normalize_talent(base["talent"].fillna(
                              base["talent"].mean() if base["talent"].notna().any() else 500))
    base["sagarin_norm"]    = _normalize_sagarin(base["sagarin"].fillna(
                              base["sagarin"].mean() if base["sagarin"].notna().any() else 75.0))
    # EPA net: convert to SP+ scale (1 SD ≈ 5 SP+ points)
    has_epa_data = base["epa_net"].notna().any()
    base["epa_norm"]        = z_score(base["epa_net"].fillna(0)) * 5 \
                              if has_epa_data else pd.Series(0.0, index=base.index)
    has_havoc_data = base["havoc_total"].notna().any()
    base["havoc_norm"]      = _normalize_havoc(base["havoc_total"].fillna(
                              base["havoc_total"].mean() if has_havoc_data else 0.0)) \
                              if has_havoc_data else pd.Series(0.0, index=base.index)

    # Phase in opponent-adjusted EPA/PPA as real per-play performance data
    # accumulates, carved out of Elo's weight (elo_w below = elo - epa_adj).
    # Elo only sees the final score, so a favorite that escapes with a bad
    # offensive performance against a weak opponent (e.g. a 1-point Hail Mary
    # win) still banks a near-full Elo win. Opponent-adjusted EPA reflects how
    # the game was actually played, so once EPA data actually exists for this
    # call, let it start pulling weight the moment week 1 results exist,
    # ramping 2.5 pts/week up to the full 10% carve-out by week 4+. Gated on
    # has_epa_data so callers that don't fetch EPA (e.g. backtest.py) don't
    # silently lose Elo weight to a no-op zero signal.
    if week and week >= 1 and has_epa_data:
        weights["epa_adj"] = min(RATING_WEIGHTS["elo"], 0.025 * week)

    # def_havoc is a static (non-zero) config weight, unlike epa_adj which
    # defaults to 0 and ramps in. So when no havoc data is available (no
    # havoc_df passed — preseason, or callers like backtest.py that don't
    # fetch it), give its weight back to sagarin (where it was carved from)
    # rather than silently multiplying a real weight against an all-zero
    # signal, which would just dilute the rest of the composite by 3%.
    if not has_havoc_data:
        weights["sagarin"] = weights.get("sagarin", 0.0) + weights.get("def_havoc", 0.0)
        weights["def_havoc"] = 0.0

    # ── Weighted composite ────────────────────────────────────────────────
    w = weights
    epa_w = w.get("epa_adj", 0.0)
    havoc_w = w.get("def_havoc", 0.0)
    sag_w = w.get("sagarin", 0.0)
    elo_w = max(0, w["elo"] - epa_w)
    base["composite"] = (
        w["sp_plus"]        * base["sp_plus_norm"]   +
        w["fpi"]            * base["fpi_norm"]        +
        sag_w               * base["sagarin_norm"]    +
        elo_w               * base["elo_norm"]        +
        w["returning_prod"] * base["returning_norm"]  +
        w["talent"]         * base["talent_norm"]     +
        havoc_w              * base["havoc_norm"]      +
        epa_w               * base["epa_norm"]
    )

    # ── Carry over SP+ offense/defense/special-teams ratings for display ──
    # NOTE: specialTeams.rating is NOT given a composite weight — SP+'s
    # overall "rating" is already exactly offense.rating - defense.rating +
    # specialTeams.rating (verified across 2026 preseason data), so it's
    # already fully priced into sp_plus_norm above. Carried over here only
    # so the dashboard can show the breakdown; weighting it too would
    # double-count it.
    if sp_df is not None and not sp_df.empty:
        off_def_cols = [c for c in sp_df.columns if c in ("offense.rating", "defense.rating", "specialTeams.rating")]
        if off_def_cols and "team" in sp_df.columns:
            sp_name_col = "team"
            od = sp_df[[sp_name_col] + off_def_cols].rename(columns={sp_name_col: "team"})
            base = base.merge(od, on="team", how="left")

    # ── Apply coaching adjustments ────────────────────────────────────────
    if apply_coaching:
        try:
            from model.coaching_factor import apply_coaching_adjustments
            base = apply_coaching_adjustments(base, year=season, week=week)
        except Exception:
            base["coaching_flag"] = None

    # ── Add rank ──────────────────────────────────────────────────────────
    base = base.sort_values("composite", ascending=False).reset_index(drop=True)
    base.index += 1
    base.index.name = "rank"
    base = base.reset_index()

    return base


def load_prebuilt_ratings(path):
    """Load the 2025 power ratings CSV you already built last year."""
    df = pd.read_csv(path)
    # Standardize column names
    rename_map = {
        "SP+ Rating": "sp_plus",
        "FPI Rating": "fpi",
        "Returning production": "returning_prod",
        "2024 ELO Ratings final": "elo",
        "2024 final talent ratings": "talent",
    }
    df = df.rename(columns=rename_map)
    if "team" not in df.columns and df.columns[0] != "team":
        df = df.rename(columns={df.columns[0]: "team"})
    return df
