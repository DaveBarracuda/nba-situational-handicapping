"""Puntos base esperados por equipo (guía 6.3).

    Pts_base_home = Liga_avg_pts * (Ataque_home/Liga_avg) * (Defensa_away/Liga_avg)
                    * Factor_ritmo * Factor_casa
    Pts_base_away = ídem con (2 − Factor_casa)

Los ajustes situacionales NO entran en el factor multiplicativo: se aplican
después, de forma aditiva y repartida entre margen y total, para no
doble-contarlos con el HFA dinámico del Elo (guía 5.5A.3b).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nba_sh.config import CONFIG, POSSESSIONS_SCALE, Config
from nba_sh.ratings.elo import ELO_POINTS_PER_POINT

REQUIRED_COLUMNS = (
    "off_home_pre",
    "def_home_pre",
    "pace_home_pre",
    "off_away_pre",
    "def_away_pre",
    "pace_away_pre",
    "league_off_mean",
    "league_pace_mean",
)


@dataclass(frozen=True)
class BasePoints:
    """Puntos esperados de un partido, antes y después de lo situacional."""

    pts_base_home: float
    pts_base_away: float
    pts_exp_home: float
    pts_exp_away: float

    @property
    def margin_base(self) -> float:
        """Margen esperado solo por ratings."""
        return self.pts_base_home - self.pts_base_away

    @property
    def total_base(self) -> float:
        """Total esperado solo por ratings."""
        return self.pts_base_home + self.pts_base_away

    @property
    def margin_expected(self) -> float:
        """Margen esperado con ajustes situacionales."""
        return self.pts_exp_home - self.pts_exp_away

    @property
    def total_expected(self) -> float:
        """Total esperado con ajustes situacionales."""
        return self.pts_exp_home + self.pts_exp_away


def home_factor(*, config: Config = CONFIG) -> float:
    """Factor de casa derivado del HFA estático en puntos Elo.

    Se usa el HFA **estático**: la parte dinámica (B2B visitante, altitud) la
    aporta la capa situacional, de modo que nada se cuenta dos veces.
    """
    hfa_points = config.elo.hfa_elo / ELO_POINTS_PER_POINT
    league_avg_pts = (
        config.power.prior_off_rating * config.power.prior_pace / POSSESSIONS_SCALE
    )
    return float(1.0 + hfa_points / 2.0 / league_avg_pts)


def expected_base_points(
    features: pd.DataFrame,
    *,
    margin_adjustment: pd.Series | None = None,
    total_adjustment: pd.Series | None = None,
    config: Config = CONFIG,
) -> pd.DataFrame:
    """Calcula los puntos esperados por equipo para cada juego.

    Args:
        features: frame con los ratings de potencia previos al juego.
        margin_adjustment: puntos situacionales a favor del local (signo +).
        total_adjustment: puntos situacionales sobre el total del partido.
        config: configuración del modelo.

    Returns:
        DataFrame con ``pts_base_home/away``, ``pts_exp_home/away``,
        ``margin_base``, ``total_base``, ``margin_expected``, ``total_expected``
        y el ritmo esperado del partido.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"missing power rating columns: {missing}")

    league_off = features["league_off_mean"]
    league_pace = features["league_pace_mean"]
    league_avg_pts = league_off * league_pace / POSSESSIONS_SCALE

    expected_possessions = (features["pace_home_pre"] + features["pace_away_pre"]) / 2.0
    pace_factor = expected_possessions / league_pace
    factor_home = home_factor(config=config)

    pts_base_home = (
        league_avg_pts
        * (features["off_home_pre"] / league_off)
        * (features["def_away_pre"] / league_off)
        * pace_factor
        * factor_home
    )
    pts_base_away = (
        league_avg_pts
        * (features["off_away_pre"] / league_off)
        * (features["def_home_pre"] / league_off)
        * pace_factor
        * (2.0 - factor_home)
    )

    zeros = pd.Series(0.0, index=features.index)
    margin_adj = zeros if margin_adjustment is None else margin_adjustment.fillna(0.0)
    total_adj = zeros if total_adjustment is None else total_adjustment.fillna(0.0)

    pts_exp_home = pts_base_home + margin_adj / 2.0 + total_adj / 2.0
    pts_exp_away = pts_base_away - margin_adj / 2.0 + total_adj / 2.0

    return pd.DataFrame(
        {
            "expected_possessions": expected_possessions,
            "pts_base_home": pts_base_home,
            "pts_base_away": pts_base_away,
            "margin_base": pts_base_home - pts_base_away,
            "total_base": pts_base_home + pts_base_away,
            "pts_exp_home": pts_exp_home,
            "pts_exp_away": pts_exp_away,
            "margin_expected": pts_exp_home - pts_exp_away,
            "total_expected": pts_exp_home + pts_exp_away,
        },
        index=features.index,
    )


__all__ = ["BasePoints", "REQUIRED_COLUMNS", "expected_base_points", "home_factor"]
