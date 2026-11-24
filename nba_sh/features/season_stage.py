"""Etapa de temporada, standings y motivación (guía 5.5 y 5.5A.4).

Las etapas (juegos 1-20 / 21-45 / 46-70 / últimos 12) cambian radicalmente el
valor de cada factor: los spots son débiles en octubre y la "ventana dorada"
está en febrero-marzo. Los standings se calculan con un recorrido cronológico
y solo con juegos ya disputados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.features.long_frame import ORDER_COLUMN
from nba_sh.teams import conference

SEASON_GAMES = 82
STAGE_EARLY = "early"
STAGE_MID = "mid"
STAGE_GOLDEN = "golden"
STAGE_LATE = "late"
STAGES = (STAGE_EARLY, STAGE_MID, STAGE_GOLDEN, STAGE_LATE)

PLAY_IN_LOW_SEED = 12
PLAY_IN_HIGH_SEED = 7
TANKING_FIRST_SEED = 13
CLINCHED_TOP_SEED = 3
TANKING_RANK_SPAN = 3.0
NEUTRAL_WIN_PCT = 0.5

STAGE_COLUMNS = (
    "stage",
    "games_remaining",
    "wins_to_date",
    "losses_to_date",
    "win_pct_to_date",
    "conference_rank",
    "must_win_flag",
    "tanking_score",
    "clinched_no_stakes_flag",
)


def stage_for(team_game_number: int, *, config: Config = CONFIG) -> str:
    """Etapa de temporada a partir del número de juego del equipo."""
    stage = config.stage
    number = int(team_game_number)
    if number <= stage.early_last_game:
        return STAGE_EARLY
    if number <= stage.mid_last_game:
        return STAGE_MID
    if number <= min(stage.golden_last_game, SEASON_GAMES - stage.late_games_count):
        return STAGE_GOLDEN
    return STAGE_LATE


def stage_multipliers(stage: str, *, config: Config = CONFIG) -> dict[str, float]:
    """Multiplicadores de la etapa para cada familia de factores."""
    settings = config.stage
    return {
        "spot": float(settings.spot_multiplier[stage]),
        "momentum": float(settings.momentum_multiplier[stage]),
        "injury": float(settings.injury_multiplier[stage]),
        "motivation": float(settings.motivation_multiplier[stage]),
        "rest": float(settings.rest_multiplier[stage]),
    }


def _conference_ranks(standings: dict[str, tuple[int, int]]) -> dict[str, int]:
    """Ranking por conferencia según el win pct acumulado."""
    frame = pd.DataFrame(
        [
            {
                "team": team,
                "conference": conference(team),
                "win_pct": wins / (wins + losses) if wins + losses else NEUTRAL_WIN_PCT,
            }
            for team, (wins, losses) in standings.items()
        ]
    )
    frame["rank"] = frame.groupby("conference")["win_pct"].rank(
        ascending=False, method="min"
    )
    return {row["team"]: int(row["rank"]) for row in frame.to_dict(orient="records")}


def add_stage_features(
    long_frame: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Añade etapa, standings previos y flags de motivación.

    Args:
        long_frame: frame equipo-juego con ``team_game_number``.
        config: configuración del modelo.

    Returns:
        El frame con las columnas de ``STAGE_COLUMNS`` añadidas.
    """
    if long_frame.empty:
        empty = long_frame.copy()
        for column in STAGE_COLUMNS:
            empty[column] = pd.Series(dtype="object")
        return empty

    frame = long_frame.sort_values(["game_date", "game_id", "is_home"]).copy()
    frame["stage"] = [
        stage_for(number, config=config) for number in frame["team_game_number"]
    ]
    frame["games_remaining"] = (
        SEASON_GAMES - frame["team_game_number"] + 1
    ).clip(lower=0)

    teams = sorted(frame["team"].unique())
    standings: dict[str, tuple[int, int]] = dict.fromkeys(teams, (0, 0))
    current_season: str | None = None
    wins_column: list[int] = []
    losses_column: list[int] = []
    rank_column: list[int] = []

    for _, day_rows in frame.groupby(["season", "game_date"], sort=False):
        season = str(day_rows["season"].iloc[0])
        if current_season is not None and season != current_season:
            standings = dict.fromkeys(teams, (0, 0))
        current_season = season
        ranks = _conference_ranks(standings)
        records = day_rows.to_dict(orient="records")
        for record in records:
            team = str(record["team"])
            wins, losses = standings[team]
            wins_column.append(wins)
            losses_column.append(losses)
            rank_column.append(ranks[team])
        for record in records:
            team = str(record["team"])
            won = bool(record["won"])
            wins, losses = standings[team]
            standings[team] = (wins + int(won), losses + int(not won))

    frame["wins_to_date"] = wins_column
    frame["losses_to_date"] = losses_column
    frame["conference_rank"] = rank_column
    played = (frame["wins_to_date"] + frame["losses_to_date"]).to_numpy(dtype=float)
    frame["win_pct_to_date"] = np.where(
        played > 0,
        frame["wins_to_date"].to_numpy(dtype=float) / np.maximum(played, 1.0),
        NEUTRAL_WIN_PCT,
    )

    is_late = frame["stage"] == STAGE_LATE
    is_late_or_golden = frame["stage"].isin((STAGE_GOLDEN, STAGE_LATE))
    frame["must_win_flag"] = (
        is_late
        & (frame["conference_rank"] >= PLAY_IN_HIGH_SEED)
        & (frame["conference_rank"] <= PLAY_IN_LOW_SEED)
    )
    frame["tanking_score"] = np.where(
        is_late_or_golden & (frame["conference_rank"] >= TANKING_FIRST_SEED),
        np.clip(
            (frame["conference_rank"] - PLAY_IN_LOW_SEED) / TANKING_RANK_SPAN, 0.0, 1.0
        ),
        0.0,
    )
    frame["clinched_no_stakes_flag"] = is_late & (
        frame["conference_rank"] <= CLINCHED_TOP_SEED
    )
    return frame.sort_values(ORDER_COLUMN).reset_index(drop=True)


__all__ = [
    "SEASON_GAMES",
    "STAGES",
    "STAGE_COLUMNS",
    "STAGE_EARLY",
    "STAGE_GOLDEN",
    "STAGE_LATE",
    "STAGE_MID",
    "add_stage_features",
    "stage_for",
    "stage_multipliers",
]
