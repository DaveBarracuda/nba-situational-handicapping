"""Lesiones y disponibilidad (guía 5.3).

    injury_impact = Σ minutos_del_ausente * on_off_net * coeficiente
    pct_minutos_perdidos_7d
    late_scratch_flag = baja reportada < 3h antes del tip

Nunca "estrella fuera = -X puntos": el impacto se modela por **minutos
ponderados por net rating on/off**, y los estados dudosos se ponderan por su
probabilidad de jugar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sh.config import CONFIG, MINUTES_PER_GAME, Config
from nba_sh.features.long_frame import restore_order, sorted_by_team
from nba_sh.schemas import STATUS_DOUBTFUL, STATUS_OUT, STATUS_QUESTIONABLE

PLAYERS_ON_COURT = 5
TEAM_MINUTES_PER_GAME = MINUTES_PER_GAME * PLAYERS_ON_COURT
RECENT_GAMES_WINDOW = 4
INJURY_FEATURE_COLUMNS = (
    "injury_impact_points",
    "injury_n_out",
    "injury_minutes_out",
    "injury_pct_minutes_out",
    "late_scratch_flag",
    "injury_pct_minutes_out_7d",
    "injury_pace_impact",
)


def missing_weight(status: str, *, config: Config = CONFIG) -> float:
    """Probabilidad de que el jugador NO juegue, según su estado."""
    normalized = str(status).lower()
    if normalized == STATUS_OUT:
        return 1.0
    if normalized == STATUS_DOUBTFUL:
        return 1.0 - config.injury.doubtful_probability
    if normalized == STATUS_QUESTIONABLE:
        return 1.0 - config.injury.questionable_probability
    return 0.0


def aggregate_injuries(
    injuries: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Agrega las bajas a nivel (juego, equipo).

    Returns:
        DataFrame con ``game_id``, ``team``, ``impact_points``, minutos
        perdidos, número de bajas y flag de late scratch.
    """
    columns = [
        "game_id",
        "team",
        "impact_points",
        "n_out",
        "minutes_out",
        "late_scratch_flag",
    ]
    if injuries is None or injuries.empty:
        return pd.DataFrame(columns=columns)

    frame = injuries.copy()
    frame["weight"] = frame["status"].map(
        lambda value: missing_weight(value, config=config)
    )
    impact = (
        frame["minutes_pg"]
        * frame["on_off_net"]
        * config.injury.points_per_minute_net_rating
        * frame["weight"]
    )
    late = frame["hours_before_tip"] < config.injury.late_scratch_hours
    frame["impact_points"] = np.where(
        late, impact * config.injury.late_scratch_extra_multiplier, impact
    )
    frame["late_scratch_flag"] = late.astype(float)
    frame["minutes_out"] = frame["minutes_pg"] * frame["weight"]

    aggregated = frame.groupby(["game_id", "team"], as_index=False).agg(
        impact_points=("impact_points", "sum"),
        n_out=("player_id", "count"),
        minutes_out=("minutes_out", "sum"),
        late_scratch_flag=("late_scratch_flag", "max"),
    )
    aggregated["impact_points"] = aggregated["impact_points"].clip(
        upper=config.injury.impact_cap_points
    )
    return aggregated[columns]


def add_injury_features(
    long_frame: pd.DataFrame, injuries: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Añade features de lesiones al frame largo.

    El informe de lesiones del día es información **previa** al tip-off, así
    que la ventana reciente incluye el juego actual.
    """
    if long_frame.empty:
        empty = long_frame.copy()
        for column in INJURY_FEATURE_COLUMNS:
            empty[column] = pd.Series(dtype="float64")
        return empty

    aggregated = aggregate_injuries(injuries, config=config)
    frame = long_frame.merge(aggregated, on=["game_id", "team"], how="left")
    # Sin parte de lesiones (o sin bajas en ese juego) el impacto es 0, no NaN.
    # El merge puede dejar las columnas como object, así que se fuerza numérico.
    for source, target in (
        ("impact_points", "injury_impact_points"),
        ("n_out", "injury_n_out"),
        ("minutes_out", "injury_minutes_out"),
        ("late_scratch_flag", "late_scratch_flag"),
    ):
        frame[target] = pd.to_numeric(frame[source], errors="coerce").fillna(0.0)
    frame["injury_pct_minutes_out"] = frame["injury_minutes_out"] / TEAM_MINUTES_PER_GAME
    frame["injury_pace_impact"] = (
        frame["injury_pct_minutes_out"] * config.injury.pace_impact_points_per_unit
    )
    frame = frame.drop(columns=["impact_points", "n_out", "minutes_out"])

    ordered = sorted_by_team(frame)
    ordered["injury_pct_minutes_out_7d"] = ordered.groupby("team", sort=False)[
        "injury_pct_minutes_out"
    ].transform(lambda series: series.rolling(RECENT_GAMES_WINDOW, min_periods=1).mean())
    return restore_order(ordered)


__all__ = [
    "INJURY_FEATURE_COLUMNS",
    "RECENT_GAMES_WINDOW",
    "TEAM_MINUTES_PER_GAME",
    "add_injury_features",
    "aggregate_injuries",
    "missing_weight",
]
