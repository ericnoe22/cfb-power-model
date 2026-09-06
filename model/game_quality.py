"""
game_quality.py — flags games where the scoreboard and the box score disagree.

Elo and the final score only tell you who won. CFBD's game-level advanced
box scores (PPA, success rate) tell you who actually played better, and
traditional box scores tell you who won the turnover battle. A team can
escape with a win despite getting outplayed play-for-play, or despite
losing the turnover battle badly (turnover margin is one of the least
repeatable stats in football — closer to variance than skill) — this
module flags both, plus the mirror cases, so they show up as a signal
rather than an anecdote.

Deliberately NOT folded into the composite rating's weights: unlike
opponent-adjusted PPA (which measures a repeatable skill — efficient
play), turnover margin is mostly noise. Rewarding/penalizing a team's
rating for it would bake in luck as if it were signal. It belongs here,
as a diagnostic flag, not in config.RATING_WEIGHTS.
"""

import pandas as pd

# Below this, a per-play PPA margin is treated as "essentially even" rather
# than a real edge either way — avoids flagging razor-thin games as upsets.
NEUTRAL_MARGIN = 0.05

# A turnover margin at least this lopsided, in the game's outcome-defying
# direction, gets flagged (e.g. won despite giving the ball away 2+ more
# times than you took it away).
TURNOVER_MARGIN_THRESHOLD = 2


def compute_game_quality_flags(games_df, box_df, team_game_stats_df=None):
    """
    Merge completed games with their advanced box scores (and, optionally,
    traditional box score turnover counts) and flag each team-game as a
    scoreboard mismatch or not.

    games_df: from fetch_completed_games() — needs id, homeTeam, awayTeam,
              homePoints, awayPoints, week.
    box_df:   from fetch_advanced_box_scores() — one row per team per game,
              needs gameId, team, opponent, offense.ppa, defense.ppa,
              offense.totalPPA, defense.totalPPA (dot-notation columns from
              pd.json_normalize).
    team_game_stats_df: optional, from fetch_team_game_stats() — needs
              gameId, team, turnovers. When omitted, turnover_margin and
              turnover_flag are left as None/NaN.

    Returns a DataFrame with one row per team per game:
      gameId, week, team, opponent, won, score_for, score_against,
      net_ppa_margin (per-play), net_total_ppa_margin, off_success_rate,
      def_success_rate, off_passing_downs_success_rate,
      flag ("won_but_outplayed" / "lost_but_outplayed" / None),
      turnover_margin (takeaways - giveaways, None if unknown),
      turnover_flag ("won_despite_turnovers" / "lost_despite_turnovers" / None)
    """
    if games_df is None or games_df.empty or box_df is None or box_df.empty:
        return pd.DataFrame()

    required = {"offense.ppa", "defense.ppa", "gameId", "team", "opponent"}
    if not required.issubset(box_df.columns):
        return pd.DataFrame()

    games = games_df[["id", "week", "homeTeam", "awayTeam", "homePoints", "awayPoints"]].copy()

    turnovers_by_game_team = {}
    if (team_game_stats_df is not None and not team_game_stats_df.empty
            and "turnovers" in team_game_stats_df.columns):
        for _, r in team_game_stats_df.iterrows():
            turnovers_by_game_team[(r["gameId"], r["team"])] = r["turnovers"]

    rows = []
    for _, box in box_df.iterrows():
        game = games[games["id"] == box["gameId"]]
        if game.empty:
            continue
        game = game.iloc[0]

        if box["team"] == game["homeTeam"]:
            score_for, score_against = game["homePoints"], game["awayPoints"]
        elif box["team"] == game["awayTeam"]:
            score_for, score_against = game["awayPoints"], game["homePoints"]
        else:
            continue
        if pd.isna(score_for) or pd.isna(score_against):
            continue

        won = score_for > score_against
        net_ppa_margin = box["offense.ppa"] - box["defense.ppa"]
        net_total_margin = box.get("offense.totalPPA", float("nan")) - box.get("defense.totalPPA", float("nan"))

        flag = None
        if won and net_ppa_margin < -NEUTRAL_MARGIN:
            flag = "won_but_outplayed"
        elif not won and net_ppa_margin > NEUTRAL_MARGIN:
            flag = "lost_but_outplayed"

        turnover_margin = None
        turnover_flag = None
        own_to = turnovers_by_game_team.get((box["gameId"], box["team"]))
        opp_to = turnovers_by_game_team.get((box["gameId"], box["opponent"]))
        if own_to is not None and opp_to is not None and pd.notna(own_to) and pd.notna(opp_to):
            turnover_margin = int(opp_to - own_to)  # positive = takeaways > giveaways
            if won and turnover_margin <= -TURNOVER_MARGIN_THRESHOLD:
                turnover_flag = "won_despite_turnovers"
            elif not won and turnover_margin >= TURNOVER_MARGIN_THRESHOLD:
                turnover_flag = "lost_despite_turnovers"

        rows.append({
            "gameId": box["gameId"],
            "week": game["week"],
            "team": box["team"],
            "opponent": box["opponent"],
            "won": won,
            "score_for": score_for,
            "score_against": score_against,
            "net_ppa_margin": round(net_ppa_margin, 3),
            "net_total_ppa_margin": round(net_total_margin, 2) if pd.notna(net_total_margin) else None,
            "off_success_rate": box.get("offense.successRate"),
            "def_success_rate": box.get("defense.successRate"),
            "off_passing_downs_success_rate": box.get("offense.passingDowns.successRate"),
            "flag": flag,
            "turnover_margin": turnover_margin,
            "turnover_flag": turnover_flag,
        })

    return pd.DataFrame(rows)
