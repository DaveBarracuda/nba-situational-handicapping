"""Ritmo y estilo: perfil de tiro propio y permitido (guía 5.4).

    efg / efg_allowed      = calidad de tiro propia y la que concede
    three_rate             = tendencia de zona de tiro (3PA / FGA)
    ft_rate                = tiros libres por intento de campo
    paint_share            = peso de los puntos en la pintura
    fouls                  = faltas cometidas (proxy del perfil arbitral)

Todas con ``shift(1)`` y media móvil de 10 partidos: el juego de hoy nunca
entra en sus propias medias. Si no hay detalle de tiro (p.ej. en la liga
sintética), las columnas quedan a 0 y los modelos simplemente no las usan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.features.long_frame import restore_order, sorted_by_team
from nba_sh.features.windows import season_weighted_rolling

WINDOW = 10
MIN_PERIODS = 4
EFG_THREE_WEIGHT = 0.5
SHOOTING_METRICS = (
    "efg",
    "three_rate",
    "ft_rate",
    "paint_share",
    "fouls",
    "fast_break_points",
)
ALLOWED_METRICS = ("efg_allowed", "three_rate_allowed", "ft_rate_allowed")
SHOOTING_COLUMNS = tuple(
    f"{metric}_last_10" for metric in (*SHOOTING_METRICS, *ALLOWED_METRICS)
)


def team_shooting_profile(team_box: pd.DataFrame) -> pd.DataFrame:
    """Métricas de estilo por equipo-partido, a partir del box score."""
    if team_box is None or team_box.empty:
        return pd.DataFrame(columns=["game_id", "team", *SHOOTING_METRICS])
    frame = team_box.copy()
    attempts = pd.to_numeric(frame["field_goals_attempted"], errors="coerce").replace(
        0.0, np.nan
    )
    made = pd.to_numeric(frame["field_goals_made"], errors="coerce")
    threes = pd.to_numeric(
        frame["three_point_field_goals_made"], errors="coerce"
    ).fillna(0.0)
    three_attempts = pd.to_numeric(
        frame["three_point_field_goals_attempted"], errors="coerce"
    ).fillna(0.0)
    free_throws = pd.to_numeric(frame["free_throws_attempted"], errors="coerce").fillna(
        0.0
    )
    score = pd.to_numeric(frame["score"], errors="coerce").replace(0.0, np.nan)
    profile = pd.DataFrame(
        {
            "game_id": frame["game_id"].astype(str).to_numpy(),
            "team": frame["team"].to_numpy(),
            "efg": ((made + EFG_THREE_WEIGHT * threes) / attempts).to_numpy(),
            "three_rate": (three_attempts / attempts).to_numpy(),
            "ft_rate": (free_throws / attempts).to_numpy(),
            "paint_share": (
                pd.to_numeric(frame.get("points_in_paint"), errors="coerce") / score
            ).to_numpy(),
            "fouls": pd.to_numeric(frame.get("fouls"), errors="coerce").to_numpy(),
            "fast_break_points": pd.to_numeric(
                frame.get("fast_break_points"), errors="coerce"
            ).to_numpy(),
        }
    )
    return profile.dropna(subset=["efg"]).reset_index(drop=True)


def add_shooting_features(
    long_frame: pd.DataFrame, team_box: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Añade el perfil de tiro rolling (propio y permitido) al frame largo."""
    enriched = long_frame.copy()
    profile = team_shooting_profile(team_box)
    if profile.empty:
        for column in SHOOTING_COLUMNS:
            enriched[column] = 0.0
        return enriched

    enriched = enriched.merge(profile, on=["game_id", "team"], how="left")
    allowed = profile.rename(
        columns={
            "team": "opponent",
            "efg": "efg_allowed",
            "three_rate": "three_rate_allowed",
            "ft_rate": "ft_rate_allowed",
        }
    )[["game_id", "opponent", *ALLOWED_METRICS]]
    enriched = enriched.merge(allowed, on=["game_id", "opponent"], how="left")

    ordered = sorted_by_team(enriched)
    sources = [*SHOOTING_METRICS, *ALLOWED_METRICS]
    for source in sources:
        ordered[f"{source}_last_10"] = season_weighted_rolling(
            ordered,
            source,
            window=WINDOW,
            min_periods=MIN_PERIODS,
            carryover_weight=config.form.season_carryover_weight,
        )
    ordered = ordered.drop(columns=sources)
    for column in SHOOTING_COLUMNS:
        ordered[column] = ordered[column].fillna(0.0)
    return restore_order(ordered)


__all__ = [
    "ALLOWED_METRICS",
    "EFG_THREE_WEIGHT",
    "MIN_PERIODS",
    "SHOOTING_COLUMNS",
    "SHOOTING_METRICS",
    "WINDOW",
    "add_shooting_features",
    "team_shooting_profile",
]
