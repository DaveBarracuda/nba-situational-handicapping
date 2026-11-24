"""Capa 6: motor de apuestas (guía secciones 10 y 11).

- :mod:`nba_sh.betting.edge`: comparación modelo vs mercado y umbrales.
- :mod:`nba_sh.betting.kelly`: Kelly fraccionado y sizing por confianza.
- :mod:`nba_sh.betting.limits`: stake caps, exposición y stop-loss.
- :mod:`nba_sh.betting.engine`: árbol de decisión por juego y registro.
"""

from nba_sh.betting.edge import BET_COLUMNS, compute_candidates
from nba_sh.betting.engine import BettingEngine, EngineResult
from nba_sh.betting.kelly import fractional_kelly, kelly_fraction
from nba_sh.betting.limits import BankrollState, LimitDecision, apply_limits
from nba_sh.betting.settle import settle_bets

__all__ = [
    "BET_COLUMNS",
    "BankrollState",
    "BettingEngine",
    "EngineResult",
    "LimitDecision",
    "apply_limits",
    "compute_candidates",
    "fractional_kelly",
    "kelly_fraction",
    "settle_bets",
]
