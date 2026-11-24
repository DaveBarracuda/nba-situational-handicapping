"""Árbitros: factor subutilizado, solo para el mercado de totales (guía 5.6).

    crew_foul_rate = faltas personales por 48 min del equipo arbitral
    crew_fta_rate  = tiros libres intentados por partido del crew
    impacto en el total esperado = ±3-5 puntos entre crews extremos
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sh.config import CONFIG, Config

REFEREE_COLUMNS_OUT = ("crew_foul_rate", "crew_fta_rate", "referee_total_points")


def referee_total_points(
    crew_fta_rate: pd.Series, *, config: Config = CONFIG
) -> pd.Series:
    """Puntos de ajuste al total esperado por perfil arbitral."""
    settings = config.referee
    delta = pd.Series(crew_fta_rate).astype(float) - settings.league_avg_crew_fta
    points = delta * settings.points_per_fta_above_average
    return points.clip(-settings.max_total_points, settings.max_total_points)


def add_referee_features(
    games: pd.DataFrame, referees: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Añade el perfil del crew y su ajuste al total.

    Si no hay asignación arbitral (habitual hasta la mañana del partido), el
    ajuste es 0: el modelo no inventa un crew.
    """
    frame = games.copy()
    if referees is None or referees.empty or "crew_id" not in frame.columns:
        frame["crew_foul_rate"] = np.nan
        frame["crew_fta_rate"] = np.nan
        frame["referee_total_points"] = 0.0
        return frame

    merged = frame.merge(
        referees[["crew_id", "crew_foul_rate", "crew_fta_rate"]],
        on="crew_id",
        how="left",
    )
    merged["referee_total_points"] = referee_total_points(
        merged["crew_fta_rate"].fillna(config.referee.league_avg_crew_fta),
        config=config,
    ).to_numpy()
    return merged


__all__ = ["REFEREE_COLUMNS_OUT", "add_referee_features", "referee_total_points"]
