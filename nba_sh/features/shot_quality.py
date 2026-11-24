"""Calidad de tiro: dónde se tira, no si entra (guía 5.4).

    shot_quality   = puntos por intento que valen los tiros elegidos
    shot_making    = puntos por intento por encima de lo que valían
    rim_rate       = proporción de tiros al aro (≤4 pies)
    mid_rate       = proporción de tiros de media distancia (8-23,75 pies)

Las versiones ``_allowed`` son lo mismo visto desde la defensa: qué tiros
concede y cuánto acierta el rival en ellos. Todas con ``shift(1)`` y media
móvil de 10 partidos. Sin play-by-play (p.ej. en la liga sintética) quedan a 0
y los modelos simplemente no las usan.

La diferencia con :mod:`nba_sh.features.shooting`: ahí el eFG mezcla las dos
cosas; aquí están separadas, porque no se comportan igual.
"""

from __future__ import annotations

import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.features.long_frame import restore_order, sorted_by_team
from nba_sh.features.windows import season_weighted_rolling

WINDOW = 10
MIN_PERIODS = 4
QUALITY_METRICS = ("shot_quality", "shot_making", "rim_rate", "mid_rate")
ALLOWED_METRICS = ("shot_quality_allowed", "shot_making_allowed")
SHOT_QUALITY_COLUMNS = tuple(
    f"{metric}_last_10" for metric in (*QUALITY_METRICS, *ALLOWED_METRICS)
)


def add_shot_quality_features(
    long_frame: pd.DataFrame, shot_quality: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Añade la calidad de tiro rolling (propia y concedida) al frame largo."""
    enriched = long_frame.copy()
    if shot_quality is None or shot_quality.empty:
        for column in SHOT_QUALITY_COLUMNS:
            enriched[column] = 0.0
        return enriched

    own = shot_quality[["game_id", "team", *QUALITY_METRICS]].copy()
    own["game_id"] = own["game_id"].astype(str)
    enriched["game_id"] = enriched["game_id"].astype(str)
    enriched = enriched.merge(own, on=["game_id", "team"], how="left")

    allowed = own[["game_id", "team", "shot_quality", "shot_making"]].rename(
        columns={
            "team": "opponent",
            "shot_quality": "shot_quality_allowed",
            "shot_making": "shot_making_allowed",
        }
    )
    enriched = enriched.merge(allowed, on=["game_id", "opponent"], how="left")

    ordered = sorted_by_team(enriched)
    sources = [*QUALITY_METRICS, *ALLOWED_METRICS]
    for source in sources:
        ordered[f"{source}_last_10"] = season_weighted_rolling(
            ordered,
            source,
            window=WINDOW,
            min_periods=MIN_PERIODS,
            carryover_weight=config.form.season_carryover_weight,
        )
    ordered = ordered.drop(columns=sources)
    for column in SHOT_QUALITY_COLUMNS:
        ordered[column] = ordered[column].fillna(0.0)
    return restore_order(ordered)


__all__ = [
    "ALLOWED_METRICS",
    "MIN_PERIODS",
    "QUALITY_METRICS",
    "SHOT_QUALITY_COLUMNS",
    "WINDOW",
    "add_shot_quality_features",
]
