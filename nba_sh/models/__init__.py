"""Capa 5b: modelos de mercado (guía secciones 7, 8 y 9).

- :mod:`nba_sh.models.simulate`: simulación Monte Carlo con marginales
  binomiales negativas y correlación entre los puntos de ambos equipos.
- :mod:`nba_sh.models.totals`: mercado de totales.
- :mod:`nba_sh.models.moneyline`: logit calibrado + prior Elo.
- :mod:`nba_sh.models.spread`: gaussiano/simulado + ridge regularizado.
- :mod:`nba_sh.models.bundle`: los tres modelos entrenados a la vez.
"""

from nba_sh.models.bundle import PREDICTION_COLUMNS, ModelBundle
from nba_sh.models.moneyline import MoneylineModel
from nba_sh.models.simulate import SimulationSummary, simulate_market_probabilities
from nba_sh.models.spread import SpreadModel
from nba_sh.models.totals import TotalsModel

__all__ = [
    "PREDICTION_COLUMNS",
    "ModelBundle",
    "MoneylineModel",
    "SimulationSummary",
    "SpreadModel",
    "TotalsModel",
    "simulate_market_probabilities",
]
