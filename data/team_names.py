"""
team_names.py — canonical team name mappings.

CFBD and ESPN/SP+ sometimes use different spellings for the same school.
All data gets normalized to the CFBD schedule names before merging.
Add new entries here whenever a mismatch appears.
"""

# Maps schedule names → ratings (SP+) names, for cases where they differ.
# Only add entries where a real mismatch exists — don't map names that already match.
TEAM_NAME_MAP = {
    # CFBD schedule uses these; SP+ article uses the right-hand side
    "App State":        "Appalachian State",
    "Hawai'i":          "Hawaii",
    "Massachusetts":    "UMass",
    "UL Monroe":        "Louisiana-Monroe",
}

# Reverse map: CFBD name → canonical display name (for readability)
REVERSE_MAP = {v: k for k, v in TEAM_NAME_MAP.items()}


def normalize(name: str) -> str:
    """Convert any known variant to the CFBD canonical name."""
    return TEAM_NAME_MAP.get(name, name)


def normalize_series(series):
    """Apply normalization to a pandas Series of team names."""
    return series.map(lambda x: normalize(x) if isinstance(x, str) else x)


def normalize_df_teams(df, cols=("team",)):
    """Normalize team name columns in a DataFrame in-place."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = normalize_series(df[col])
    return df
