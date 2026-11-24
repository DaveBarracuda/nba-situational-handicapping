"""Límites y protecciones de bankroll (guía 11.2).

    - Stake máximo por apuesta: 2% del bankroll
    - Exposición máxima por día: 6%
    - Exposición máxima por equipo: 3% (riesgo correlacionado)
    - Stop-loss diario: -4% -> no apostar más ese día
    - Descuento por mercados correlacionados del mismo juego
    - Degradación: si la CLV de 30 días < 0, stakes a la mitad

El estado del bankroll es inmutable: cada apuesta aceptada devuelve un estado
nuevo, lo que hace el backtest reproducible y auditable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from nba_sh.config import CONFIG, Config

MIN_STAKE = 1.0

APPROVED = ""
REASON_STOP_LOSS = "stop-loss diario alcanzado"
REASON_MAX_BETS = "limite de apuestas por dia"
REASON_DAILY_EXPOSURE = "exposicion diaria agotada"
REASON_TEAM_EXPOSURE = "exposicion por equipo agotada"
REASON_TINY_STAKE = "stake por debajo del minimo"


@dataclass(frozen=True)
class BankrollState:
    """Estado del bankroll durante una jornada."""

    bankroll: float
    day_staked: float = 0.0
    day_pnl: float = 0.0
    day_bets: int = 0
    team_exposure: Mapping[str, float] = field(default_factory=dict)
    games_with_bets: frozenset[str] = frozenset()
    stake_multiplier: float = 1.0

    @classmethod
    def start(cls, *, config: Config = CONFIG) -> BankrollState:
        """Estado inicial con el bankroll configurado."""
        return cls(bankroll=config.bankroll.starting_bankroll)

    def new_day(self) -> BankrollState:
        """Reinicia los contadores diarios conservando el bankroll."""
        return replace(
            self,
            day_staked=0.0,
            day_pnl=0.0,
            day_bets=0,
            team_exposure={},
            games_with_bets=frozenset(),
        )

    def with_bet(
        self, *, stake: float, teams: tuple[str, ...], game_id: str
    ) -> BankrollState:
        """Registra una apuesta aceptada (devuelve un estado nuevo)."""
        exposure = dict(self.team_exposure)
        for team in teams:
            exposure[team] = exposure.get(team, 0.0) + stake
        return replace(
            self,
            day_staked=self.day_staked + stake,
            day_bets=self.day_bets + 1,
            team_exposure=exposure,
            games_with_bets=self.games_with_bets | {game_id},
        )

    def with_settlement(self, pnl: float) -> BankrollState:
        """Aplica el resultado de una apuesta al bankroll y al P&L del día."""
        return replace(self, bankroll=self.bankroll + pnl, day_pnl=self.day_pnl + pnl)

    def with_stake_multiplier(self, multiplier: float) -> BankrollState:
        """Fija el multiplicador global de stakes (degradación por CLV)."""
        return replace(self, stake_multiplier=float(multiplier))


@dataclass(frozen=True)
class LimitDecision:
    """Resultado de aplicar los límites a un stake propuesto."""

    stake: float
    approved: bool
    reason: str

    @property
    def rejected(self) -> bool:
        """True si la apuesta no pasó los límites."""
        return not self.approved


def apply_limits(
    proposed_stake: float,
    *,
    state: BankrollState,
    teams: tuple[str, ...],
    game_id: str,
    config: Config = CONFIG,
) -> LimitDecision:
    """Aplica todos los límites de riesgo a un stake propuesto.

    Args:
        proposed_stake: stake sugerido por Kelly fraccionado.
        state: estado del bankroll de la jornada.
        teams: equipos implicados (para la exposición correlacionada).
        game_id: juego de la apuesta (para el descuento por correlación).
        config: configuración del modelo.

    Returns:
        :class:`LimitDecision` con el stake final y el motivo si se rechaza.
    """
    bankroll_config = config.bankroll
    bankroll = state.bankroll
    if state.day_pnl <= -bankroll_config.daily_stop_loss_pct * bankroll:
        return LimitDecision(0.0, False, REASON_STOP_LOSS)
    if state.day_bets >= bankroll_config.max_bets_per_day:
        return LimitDecision(0.0, False, REASON_MAX_BETS)

    stake = float(proposed_stake) * state.stake_multiplier
    if game_id in state.games_with_bets:
        stake *= bankroll_config.correlated_market_discount
    stake = min(stake, bankroll_config.max_stake_pct * bankroll)

    daily_room = bankroll_config.max_daily_exposure_pct * bankroll - state.day_staked
    if daily_room <= 0.0:
        return LimitDecision(0.0, False, REASON_DAILY_EXPOSURE)
    stake = min(stake, daily_room)

    team_room = min(
        bankroll_config.max_team_exposure_pct * bankroll
        - state.team_exposure.get(team, 0.0)
        for team in teams
    )
    if team_room <= 0.0:
        return LimitDecision(0.0, False, REASON_TEAM_EXPOSURE)
    stake = min(stake, team_room)

    if stake < MIN_STAKE:
        return LimitDecision(0.0, False, REASON_TINY_STAKE)
    return LimitDecision(round(stake, 2), True, APPROVED)


__all__ = [
    "APPROVED",
    "BankrollState",
    "LimitDecision",
    "MIN_STAKE",
    "REASON_DAILY_EXPOSURE",
    "REASON_MAX_BETS",
    "REASON_STOP_LOSS",
    "REASON_TEAM_EXPOSURE",
    "REASON_TINY_STAKE",
    "apply_limits",
]
