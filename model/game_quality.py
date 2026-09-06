"""
game_quality.py — flags games where the scoreboard and the box score disagree.

Elo and the final score only tell you who won. CFBD's game-level advanced
box scores (PPA, success rate) tell you who actually played better. A team
can escape with a win despite getting outplayed play-for-play (or the
reverse: lose a heartbreaker while clearly the better team) — this module
flags both cases so they show up as a signal, not just an anecdote.
"""

import pandas as pd

# Below this, a per-play PPA margin is treated as "essentially even" rather
# than a real edge either way — avoids flagging razor-thin games as upsets.
NEUTRAL_MARGIN = 0.05


def compute_game_quality_flags(games_df, box_df):
    """
    Merge completed games with their advanced box scores and flag each
    team-game as a scoreboard/efficiency mismatch or not.

    games_df: from fetch_completed_games() — needs id, homeTeam, awayTeam,
              homePoints, awayPoints, week.
    box_df:   from fetch_advanced_box_scores() — one row per team per game,
              needs gameId, team, opponent, offense.ppa, defense.ppa,
              offense.totalPPA, defense.totalPPA (dot-notation columns from
              pd.json_normalize).

    Returns a DataFrame with one row per team per game:
      gameId, week, team, opponent, won, score_for, score_against,
      net_ppa_margin (per-play), net_total_ppa_margin, off_success_rate,
      def_success_rate, flag ("won_but_outplayed" / "lost_but_outplayed" / None)
    """
    if games_df is None or games_df.empty or box_df is None or box_df.empty:
        return pd.DataFrame()

    required = {"offense.ppa", "defense.ppa", "gameId", "team", "opponent"}
    if not required.issubset(box_df.columns):
        return pd.DataFrame()

    games = games_df[["id", "week", "homeTeam", "awayTeam", "homePoints", "awayPoints"]].copy()

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
        })

    return pd.DataFrame(rows)
