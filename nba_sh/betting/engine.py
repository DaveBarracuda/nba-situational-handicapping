"""Motor de apuestas: árbol de decisión por juego y registro (guía 10.2).

    1. ¿Pasa filtros de calidad? (data completa, línea disponible)
    2. Calcular edge por mercado.
    3. ¿Edge > umbral mínimo del mercado?
    4. ¿El movimiento de línea confirma? (señal de confianza)
    5. Calcular stake (Kelly fraccionado).
    6. Registrar apuesta con timestamp, línea tomada, modelo y versión.

Se registran también las oportunidades **rechazadas** y el motivo: es la única
forma de que el backtest no esté sesgado (error común #7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from nba_sh.betting.edge import BET_COLUMNS, compute_candidates
from nba_sh.betting.kelly import fractional_kelly
from nba_sh.betting.limits import BankrollState, apply_limits
from nba_sh.config import CONFIG, Config
from nba_sh.schemas import TIMING_PREGAME

LOGGER = logging.getLogger(__name__)

LOG_COLUMNS: tuple[str, ...] = (
    *BET_COLUMNS,
    "stake",
    "approved",
    "limit_reason",
    "logged_at",
    "timing",
    "model_version",
)


@dataclass(frozen=True)
class EngineResult:
    """Salida del motor para una jornada."""

    bets: pd.DataFrame
    evaluated: pd.DataFrame
    state: BankrollState

    @property
    def n_bets(self) -> int:
        """Apuestas aceptadas."""
        return int(len(self.bets))

    @property
    def total_stake(self) -> float:
        """Importe total arriesgado."""
        return float(self.bets["stake"].sum()) if not self.bets.empty else 0.0


@dataclass(frozen=True)
class BettingEngine:
    """Convierte predicciones + líneas en apuestas con stake."""

    config: Config = CONFIG
    timing: str = TIMING_PREGAME

    def _shrunk_probability(self, row: pd.Series) -> float:
        """Probabilidad para el sizing: edge encogido hacia el mercado.

        El umbral de valor se evalúa con la probabilidad del modelo, pero el
        stake se calcula con el edge encogido, porque parte del edge medido es
        error de estimación propio (guía 1 y 11.1).
        """
        model_probability = float(row["model_probability"])
        market_probability = float(row["market_fair_probability"])
        shrink = self.config.bankroll.edge_shrink_for_sizing
        return float(
            market_probability + shrink * (model_probability - market_probability)
        )

    def evaluate(
        self,
        predictions: pd.DataFrame,
        lines: pd.DataFrame,
        *,
        state: BankrollState,
        logged_at: pd.Timestamp | None = None,
    ) -> EngineResult:
        """Evalúa una jornada completa y devuelve las apuestas aprobadas.

        Args:
            predictions: predicciones del modelo para los juegos del día.
            lines: líneas de mercado indexadas por ``game_id``.
            state: estado del bankroll al empezar la jornada.
            logged_at: timestamp del registro (por defecto, ahora en UTC).

        Returns:
            :class:`EngineResult` con apuestas, oportunidades evaluadas y el
            nuevo estado del bankroll.
        """
        evaluated = compute_candidates(predictions, lines, config=self.config)
        if evaluated.empty:
            empty = pd.DataFrame(columns=list(LOG_COLUMNS))
            return EngineResult(bets=empty, evaluated=empty, state=state)

        timestamp = (
            pd.Timestamp.now(tz="UTC") if logged_at is None else pd.Timestamp(logged_at)
        )
        evaluated = evaluated.assign(
            stake=0.0,
            approved=False,
            limit_reason="",
            logged_at=timestamp,
            timing=self.timing,
            model_version=self.config.model_version,
        )

        ordered_index = evaluated.sort_values("expected_value", ascending=False).index
        current_state = state
        for index in ordered_index:
            row = evaluated.loc[index]
            if not bool(row["is_candidate"]):
                continue
            sizing_probability = self._shrunk_probability(row)
            proposed = fractional_kelly(
                sizing_probability,
                float(row["american"]),
                bankroll=current_state.bankroll,
                kind=str(row["kelly_bucket"]),
                confidence=str(row["confidence"]),
                config=self.config,
            )
            teams = (str(row["home_team"]), str(row["away_team"]))
            decision = apply_limits(
                proposed,
                state=current_state,
                teams=teams,
                game_id=str(row["game_id"]),
                config=self.config,
            )
            evaluated.loc[index, "stake"] = decision.stake
            evaluated.loc[index, "approved"] = decision.approved
            evaluated.loc[index, "limit_reason"] = decision.reason
            if decision.approved:
                current_state = current_state.with_bet(
                    stake=decision.stake, teams=teams, game_id=str(row["game_id"])
                )

        bets = evaluated[evaluated["approved"]].copy()
        LOGGER.info(
            "jornada evaluada: %d oportunidades, %d apuestas, stake %.2f",
            len(evaluated),
            len(bets),
            float(bets["stake"].sum()) if not bets.empty else 0.0,
        )
        return EngineResult(bets=bets, evaluated=evaluated, state=current_state)


__all__ = ["LOG_COLUMNS", "BettingEngine", "EngineResult"]
