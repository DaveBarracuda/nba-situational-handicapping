"""Ingesta real de juegos y box scores vía ``nba_api`` (guía 3.1).

``nba_api`` es la fuente oficial de NBA Stats. Este módulo la convierte al
esquema canónico del proyecto y estima las posesiones con la fórmula estándar:

    Poss = FGA + 0.44*FTA - OREB + TOV

Las posesiones del partido son la media de ambos equipos. Los periodos se
infieren de los minutos del box score (240 min en 4 periodos, +25 por
prórroga), porque el endpoint de game finder no los devuelve.

La importación de ``nba_api`` es diferida: el resto del paquete funciona sin
red ni dependencias opcionales.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from nba_sh.schemas import (
    GAME_COLUMNS,
    INJURY_COLUMNS,
    ODDS_COLUMNS,
    REFEREE_COLUMNS,
    SEASON_TYPE_REGULAR,
    Dataset,
    empty_frame,
)
from nba_sh.teams import is_valid_team, normalize_team

LOGGER = logging.getLogger(__name__)

FREE_THROW_POSSESSION_FACTOR = 0.44
REGULATION_MINUTES = 240.0
OVERTIME_MINUTES = 25.0
REGULATION_PERIODS = 4
DEFAULT_TIMEOUT_SECONDS = 60
HOME_MATCHUP_TOKEN = " vs. "
NOMINAL_TIPOFF_HOURS_UTC = 24


def _require_nba_api():
    """Importa ``nba_api`` o explica cómo instalarlo."""
    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError as error:  # pragma: no cover - depende del entorno
        raise ImportError("nba_api no está instalado: pip install nba_api") from error
    return leaguegamefinder


def estimate_possessions(
    field_goal_attempts: float,
    free_throw_attempts: float,
    offensive_rebounds: float,
    turnovers: float,
) -> float:
    """Posesiones estimadas de un equipo en un partido."""
    return float(
        field_goal_attempts
        + FREE_THROW_POSSESSION_FACTOR * free_throw_attempts
        - offensive_rebounds
        + turnovers
    )


def infer_periods(team_minutes: float) -> int:
    """Infiere los periodos jugados a partir de los minutos del equipo."""
    minutes = float(team_minutes)
    if minutes <= REGULATION_MINUTES + 1.0:
        return REGULATION_PERIODS
    overtimes = int(round((minutes - REGULATION_MINUTES) / OVERTIME_MINUTES))
    return REGULATION_PERIODS + max(overtimes, 1)


def fetch_season_box_scores(
    season: str,
    *,
    season_type: str = "Regular Season",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Descarga los box scores de equipo de una temporada.

    Args:
        season: temporada en formato NBA Stats (``"2023-24"``).
        season_type: ``"Regular Season"``, ``"Playoffs"``, etc.
        timeout: timeout de la petición en segundos.

    Returns:
        DataFrame crudo tal como lo devuelve el endpoint.
    """
    leaguegamefinder = _require_nba_api()
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable=season_type,
        timeout=timeout,
    )
    frame = finder.get_data_frames()[0]
    LOGGER.info("nba_api: %d filas de box score para %s", len(frame), season)
    return frame


def box_scores_to_games(
    box_scores: pd.DataFrame, *, season_type: str = SEASON_TYPE_REGULAR
) -> pd.DataFrame:
    """Convierte box scores de equipo (2 filas por juego) al esquema canónico."""
    if box_scores.empty:
        return empty_frame(GAME_COLUMNS)

    frame = box_scores.copy()
    frame["is_home"] = frame["MATCHUP"].astype(str).str.contains(HOME_MATCHUP_TOKEN)
    frame["team"] = frame["TEAM_ABBREVIATION"].map(
        lambda value: normalize_team(value) if is_valid_team(value) else None
    )
    frame = frame.dropna(subset=["team"])
    frame["possessions_team"] = [
        estimate_possessions(fga, fta, oreb, tov)
        for fga, fta, oreb, tov in zip(
            frame["FGA"], frame["FTA"], frame["OREB"], frame["TOV"]
        , strict=True)
    ]
    frame["periods_team"] = [infer_periods(minutes) for minutes in frame["MIN"]]

    rows: list[dict[str, object]] = []
    for game_id, group in frame.groupby("GAME_ID"):
        home = group[group["is_home"]]
        away = group[~group["is_home"]]
        if len(home) != 1 or len(away) != 1:
            LOGGER.warning("juego %s con filas incompletas, se descarta", game_id)
            continue
        home_row, away_row = home.iloc[0], away.iloc[0]
        game_date = pd.Timestamp(home_row["GAME_DATE"]).normalize()
        rows.append(
            {
                "game_id": str(game_id),
                "season": str(home_row.get("SEASON_ID", ""))[-5:],
                "season_type": season_type,
                "game_date": game_date,
                "tipoff_utc": game_date
                + pd.Timedelta(hours=NOMINAL_TIPOFF_HOURS_UTC),
                "home_team": home_row["team"],
                "away_team": away_row["team"],
                "home_score": int(home_row["PTS"]),
                "away_score": int(away_row["PTS"]),
                "periods": int(max(home_row["periods_team"], away_row["periods_team"])),
                "possessions": float(
                    (home_row["possessions_team"] + away_row["possessions_team"]) / 2.0
                ),
                "crew_id": None,
            }
        )
    games = pd.DataFrame(rows, columns=list(GAME_COLUMNS))
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def fetch_dataset(
    seasons: Sequence[str],
    *,
    season_type: str = "Regular Season",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dataset:
    """Descarga varias temporadas y devuelve un :class:`Dataset`.

    Las cuotas, lesiones y árbitros se devuelven vacíos: cada uno tiene su
    propia fuente (ver :mod:`nba_sh.ingest.odds_source` y
    :mod:`nba_sh.ingest.injury_source`).
    """
    frames = [
        box_scores_to_games(
            fetch_season_box_scores(season, season_type=season_type, timeout=timeout)
        )
        for season in seasons
    ]
    games = (
        pd.concat(frames, ignore_index=True) if frames else empty_frame(GAME_COLUMNS)
    )
    return Dataset(
        games=games,
        odds=empty_frame(ODDS_COLUMNS),
        injuries=empty_frame(INJURY_COLUMNS),
        referees=empty_frame(REFEREE_COLUMNS),
    )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "FREE_THROW_POSSESSION_FACTOR",
    "box_scores_to_games",
    "estimate_possessions",
    "fetch_dataset",
    "fetch_season_box_scores",
    "infer_periods",
]
