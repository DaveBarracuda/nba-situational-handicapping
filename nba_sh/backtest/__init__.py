"""Capa 7: backtest riguroso y reportes (guía sección 12).

- :mod:`nba_sh.backtest.walkforward`: walk-forward puro, día a día.
- :mod:`nba_sh.backtest.report`: ROI por mercado y subgrupo, equity curve,
  sensibilidad y validación de spots (guía 5.5A.3d).
"""

from nba_sh.backtest.report import backtest_report, subgroup_report, validate_spots
from nba_sh.backtest.walkforward import BacktestResult, run_walkforward

__all__ = [
    "BacktestResult",
    "backtest_report",
    "run_walkforward",
    "subgroup_report",
    "validate_spots",
]
