"""
synopsis_generator.py — AI-generated matchup previews using Claude.

For each game, generates a 3-4 sentence preview covering:
  - Key team strengths/mismatches (offense vs defense ratings)
  - Whether the model agrees or disagrees with Vegas and why
  - Anything notable (rivalry, neutral site, FCS, home field)

Results are cached to cache/synopses_week{N}.json so the API is only
called once per batch — not on every page load.
"""

import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
MODEL = "claude-sonnet-4-6"


def _cache_path(week):
    week_str = str(week) if week is not None else "all"
    return os.path.join(CACHE_DIR, f"synopses_week{week_str}.json")


def _load_cache(week):
    path = _cache_path(week)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(week, cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(week), "w") as f:
        json.dump(cache, f, indent=2)


def _game_key(home_team, away_team):
    return f"{away_team}_at_{home_team}"


def _build_prompt(game):
    home = game.get("home_team", game.get("homeTeam", ""))
    away = game.get("away_team", game.get("awayTeam", ""))
    neutral = game.get("neutral", False)
    home_comp = game.get("home_composite")
    away_comp = game.get("away_composite")
    h_off = game.get("home_off_rating")
    a_off = game.get("away_off_rating")
    h_def = game.get("home_def_rating")
    a_def = game.get("away_def_rating")
    model_spread = game.get("predicted_spread")
    vegas_spread = game.get("vegas_spread")
    model_total = game.get("predicted_total")
    vegas_total = game.get("vegas_total")
    edge_grade = game.get("edge_grade", "")
    fcs_note = game.get("fcs_note")
    conf_game = game.get("conferenceGame", False)
    week = game.get("week")

    # Format spread for readability
    def fmt_spread(v, home):
        if v is None:
            return "N/A"
        if v < 0:
            return f"{home} -{abs(v)}"
        elif v > 0:
            return f"{away} -{abs(v)}"
        else:
            return "Pick'em"

    lines = [
        f"Game: {away} at {'a neutral site' if neutral else home}",
    ]
    if week:
        lines.append(f"Week: {int(week)}")
    if conf_game:
        lines.append("Conference game: Yes")
    if home_comp is not None and away_comp is not None:
        lines.append(f"Composite power ratings — {home}: {home_comp:+.1f}, {away}: {away_comp:+.1f} (higher = stronger)")
    if h_off is not None:
        lines.append(f"SP+ offense ratings — {home}: {h_off:.1f}, {away}: {a_off:.1f} (higher = better offense)")
    if h_def is not None:
        lines.append(f"SP+ defense ratings — {home}: {h_def:.1f}, {away}: {a_def:.1f} (lower = better defense)")
    if model_spread is not None:
        lines.append(f"Model spread: {fmt_spread(model_spread, home)}")
    if vegas_spread is not None:
        lines.append(f"Vegas spread: {fmt_spread(vegas_spread, home)}")
    if model_total is not None:
        lines.append(f"Model total: {model_total}")
    if vegas_total is not None:
        lines.append(f"Vegas total: {vegas_total}")
    if edge_grade:
        lines.append(f"Model edge grade: {edge_grade}")
    if fcs_note:
        lines.append(f"Note: {fcs_note}")

    data_block = "\n".join(lines)

    return f"""You are a college football analyst writing a sharp, concise game preview for a betting-focused audience.

Here is the data for this matchup:
{data_block}

Write a 3-4 sentence preview that covers:
1. The key matchup dynamic — which team has the edge and why (use the ratings to support it)
2. What the model thinks vs. Vegas — does it agree, disagree, and on which side
3. One specific angle a bettor should watch (e.g., home field, defensive mismatch, high/low scoring lean)

Be direct and analytical. No filler phrases like "In a highly anticipated matchup..." or "Both teams will look to...".
Write as if you're a sharp analyst, not a hype piece. Keep it under 100 words."""


def generate_synopsis(game, week=None, force=False):
    """
    Generate (or retrieve from cache) a synopsis for a single game.

    game: dict with keys like home_team/homeTeam, away_team/awayTeam,
          home_composite, away_composite, predicted_spread, vegas_spread, etc.
    week: used for cache file naming
    force: if True, bypass cache and regenerate

    Returns synopsis string, or None if API key is missing.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    home = game.get("home_team", game.get("homeTeam", ""))
    away = game.get("away_team", game.get("awayTeam", ""))
    key = _game_key(home, away)

    cache = _load_cache(week)
    if not force and key in cache:
        return cache[key]

    try:
        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_prompt(game)

        with client.messages.stream(
            model=MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            final = stream.get_final_message()

        synopsis = next(
            (b.text for b in final.content if b.type == "text"), ""
        ).strip()

        cache[key] = synopsis
        _save_cache(week, cache)
        return synopsis

    except Exception as e:
        return f"Preview unavailable ({e})"


def generate_synopses_batch(games, week=None, force=False):
    """
    Generate synopses for a list of game dicts.
    Only calls the API for games not already in cache.

    Returns dict keyed by _game_key(home, away) → synopsis string.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {}

    cache = _load_cache(week)
    results = {}

    for game in games:
        home = game.get("home_team", game.get("homeTeam", ""))
        away = game.get("away_team", game.get("awayTeam", ""))
        key = _game_key(home, away)

        if not force and key in cache:
            results[key] = cache[key]
            continue

        try:
            client = anthropic.Anthropic(api_key=api_key)
            prompt = _build_prompt(game)

            with client.messages.stream(
                model=MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                final = stream.get_final_message()

            synopsis = next(
                (b.text for b in final.content if b.type == "text"), ""
            ).strip()

            cache[key] = synopsis
            results[key] = synopsis

        except Exception as e:
            results[key] = f"Preview unavailable ({e})"

    _save_cache(week, cache)
    return results
