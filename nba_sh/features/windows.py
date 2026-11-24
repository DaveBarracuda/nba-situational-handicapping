"""Ventanas móviles con peso por temporada (guía 5.2).

Las medias de forma (`*_last_10`) miran los 10 partidos anteriores del equipo,
y esos 10 pueden cruzar el cambio de temporada: en octubre describen la
plantilla del año pasado. El Elo ya trata ese salto con carryover explícito
(regresión a la media); estas medias no lo trataban.

La solución es un único peso, ``season_carryover_weight``:

    media = (suma_en_temporada + w · suma_anterior) /
            (n_en_temporada    + w · n_anterior)

Con ``w = 1`` se recupera exactamente la media móvil de siempre y con ``w = 0``
el reset duro (las primeras jornadas quedan sin valor, como un equipo nuevo).
Entre medias, los partidos del año pasado cuentan, pero menos.

El reset duro no es gratis: deja ``efg_last_10`` a 0 en las primeras jornadas,
y un eFG de 0 no es neutro, es absurdo. Por eso el peso se elige midiendo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASON_COLUMN = "season"
TEAM_COLUMN = "team"
AGGREGATE_MEAN = "mean"
AGGREGATE_SUM = "sum"


def _shifted_rolling(
    frame: pd.DataFrame, column: str, keys: list[str], window: int
) -> pd.DataFrame:
    """Suma y recuento móviles de los juegos **anteriores**, por grupo."""
    keyed = [frame[key] for key in keys]
    shifted = frame.groupby(keys, sort=False)[column].transform(
        lambda series: series.shift(1)
    )
    rolled = shifted.groupby(keyed, sort=False)
    # Sin partidos previos en el grupo, la suma móvil sale NaN y no 0. Si se
    # deja, envenena la mezcla: el estreno de temporada acababa sin valor en
    # vez de heredar la forma del año anterior. El recuento a 0 ya dice que no
    # hay evidencia; quien decide si hay bastante es ``min_periods``.
    return pd.DataFrame(
        {
            "total": rolled.transform(
                lambda series: series.rolling(window, min_periods=1).sum()
            ).fillna(0.0),
            "count": rolled.transform(
                lambda series: series.rolling(window, min_periods=1).count()
            ).fillna(0.0),
        },
        index=frame.index,
    )


def season_weighted_rolling(
    frame: pd.DataFrame,
    column: str,
    *,
    window: int,
    min_periods: int,
    carryover_weight: float,
    aggregate: str = AGGREGATE_MEAN,
) -> pd.Series:
    """Media (o suma) móvil de los juegos anteriores, con peso por temporada.

    Los partidos de temporadas anteriores entran multiplicados por
    ``carryover_weight``. Con 1.0 el resultado es idéntico a
    ``series.shift(1).rolling(window, min_periods).mean()`` agrupado por
    equipo, que es lo que hacía el proyecto antes de existir este módulo.

    Args:
        frame: frame largo ordenado por equipo y fecha, con ``team`` y
            ``season``.
        column: columna numérica a agregar.
        window: tamaño de la ventana en partidos.
        min_periods: partidos previos mínimos (sobre el total, como siempre).
        carryover_weight: peso de los partidos de temporadas anteriores.
        aggregate: ``"mean"`` o ``"sum"``.

    Returns:
        Serie alineada con ``frame``; NaN donde no hay evidencia suficiente.
    """
    if frame.empty:
        return pd.Series(dtype="float64", index=frame.index)

    overall = _shifted_rolling(frame, column, [TEAM_COLUMN], window)
    if carryover_weight == 1.0:
        in_season = overall
    else:
        in_season = _shifted_rolling(
            frame, column, [TEAM_COLUMN, SEASON_COLUMN], window
        )

    total = in_season["total"] + carryover_weight * (
        overall["total"] - in_season["total"]
    )
    enough = overall["count"] >= min_periods
    if aggregate == AGGREGATE_SUM:
        return total.where(enough)
    weighted_count = in_season["count"] + carryover_weight * (
        overall["count"] - in_season["count"]
    )
    return (total / weighted_count.replace(0.0, np.nan)).where(enough)


__all__ = [
    "AGGREGATE_MEAN",
    "AGGREGATE_SUM",
    "SEASON_COLUMN",
    "TEAM_COLUMN",
    "season_weighted_rolling",
]
