"""Backtest walk-forward puro (guía 12.1 y 12.2).

    for season in temporadas:
        for game_date in fechas(season):
            features = compute_features(as_of=game_date)   # shift(1) aplicado
            models   = train_on(all_data_before(game_date))
            preds    = predict(models, features)
            for market in ["ml","spread","total"]:
                line = get_historical_line(game_date, market, timing="open_1h")
                if edge_detected(preds, line): simulate_bet(size=kelly(...))

Reglas no negociables respetadas:

* **Walk-forward puro**: el modelo de cada día se entrena solo con juegos
  anteriores; nunca hay split aleatorio.
* **Líneas reales y disponibles**: por defecto se usa la línea de T-60 min, no
  la de cierre (usar el cierre sería trampa).
* **Costos incluidos**: el vig va dentro de las cuotas; el stake sale del
  bankroll y el P&L se reinvierte.
* El feature store se construye una vez porque todas sus features son
  causales (solo usan información previa al tip-off, ver ``features/build.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_sh.betting.engine import BettingEngine
from nba_sh.betting.limits import BankrollState
from nba_sh.betting.settle import settle_bets
from nba_sh.config import CONFIG, Config
from nba_sh.features.build import FeatureStore
from nba_sh.metrics import clv_summary
from nba_sh.models.bundle import ModelBundle
from nba_sh.schemas import TIMING_CLOSE, TIMING_OPEN, TIMING_PREGAME, odds_at

LOGGER = logging.getLogger(__name__)
DEFAULT_RETRAIN_DAYS = 14
SEED_SPREAD = 10_000
OPENING_LINE_COLUMNS = {
    "ml_home": "open_ml_home",
    "spread_home": "open_spread_home",
    "total_line": "open_total_line",
}
CONTEXT_COLUMNS = (
    "situational_margin_adjustment",
    "situational_total_adjustment",
    "late_scratch_flag_home",
    "late_scratch_flag_away",
    "stage_home",
    "is_b2b_home",
    "is_b2b_away",
    "spot_lookahead_home",
    "spot_lookahead_away",
    "spot_letdown_home",
    "spot_letdown_away",
    "spot_sandwich_home",
    "spot_sandwich_away",
    "spot_revenge_home",
    "spot_revenge_away",
)


@dataclass(frozen=True)
class BacktestResult:
    """Resultado completo de un backtest."""

    bets: pd.DataFrame
    evaluated: pd.DataFrame
    predictions: pd.DataFrame
    equity: pd.DataFrame
    model_log: pd.DataFrame
    starting_bankroll: float

    @property
    def final_bankroll(self) -> float:
        """Bankroll al final del backtest."""
        if self.equity.empty:
            return self.starting_bankroll
        return float(self.equity["bankroll"].iloc[-1])

    @property
    def n_bets(self) -> int:
        """Apuestas simuladas."""
        return int(len(self.bets))


def _merge_context(
    predictions: pd.DataFrame, games: pd.DataFrame, opening: pd.DataFrame
) -> pd.DataFrame:
    """Añade contexto situacional y líneas de apertura a las predicciones."""
    available = [
        column for column in CONTEXT_COLUMNS if column in games.columns
    ]
    merged = predictions.merge(
        games[["game_id", *available]], on="game_id", how="left"
    )
    zeros = pd.Series(0.0, index=merged.index)
    late_home = (
        merged["late_scratch_flag_home"]
        if "late_scratch_flag_home" in merged.columns
        else zeros
    )
    late_away = (
        merged["late_scratch_flag_away"]
        if "late_scratch_flag_away" in merged.columns
        else zeros
    )
    merged["injury_driven"] = (
        late_home.fillna(0.0) + late_away.fillna(0.0)
    ) > 0.0
    if not opening.empty:
        renamed = opening[list(OPENING_LINE_COLUMNS)].rename(columns=OPENING_LINE_COLUMNS)
        merged = merged.merge(renamed, left_on="game_id", right_index=True, how="left")
    return merged


def _stake_multiplier(
    bet_frames: list[pd.DataFrame], day: pd.Timestamp, *, config: Config
) -> float:
    """Regla de degradación: si la CLV de 30 días es negativa, medio stake."""
    if not bet_frames:
        return 1.0
    recent = pd.concat(bet_frames, ignore_index=True)
    window_start = day - pd.Timedelta(days=config.bankroll.clv_degradation_window_days)
    recent = recent[recent["game_date"] >= window_start]
    if recent.empty:
        return 1.0
    mean_clv = clv_summary(recent)["mean_clv"]
    if np.isnan(mean_clv) or mean_clv >= config.monitoring.clv_alert_threshold:
        return 1.0
    return config.bankroll.clv_degradation_multiplier


def run_walkforward(
    feature_store: FeatureStore,
    odds: pd.DataFrame,
    *,
    start_date: pd.Timestamp | None = None,
    timing: str = TIMING_PREGAME,
    retrain_days: int = DEFAULT_RETRAIN_DAYS,
    config: Config = CONFIG,
) -> BacktestResult:
    """Ejecuta el backtest día a día sobre el feature store.

    Args:
        feature_store: features ya calculadas (causales).
        odds: tabla de cuotas con apertura, T-60 y cierre.
        start_date: primera fecha apostable; por defecto, cuando hay datos
            suficientes para entrenar (``min_train_games``).
        timing: timing de la línea usada para apostar.
        retrain_days: cada cuántos días se reentrena (guía 14.3).
        config: configuración del modelo.

    Returns:
        :class:`BacktestResult` con apuestas, oportunidades, predicciones,
        equity curve y registro de versiones del modelo.
    """
    games = feature_store.games
    if games.empty:
        empty = pd.DataFrame()
        return BacktestResult(
            bets=empty,
            evaluated=empty,
            predictions=empty,
            equity=empty,
            model_log=empty,
            starting_bankroll=config.bankroll.starting_bankroll,
        )

    betting_lines = odds_at(odds, timing)
    opening_lines = odds_at(odds, TIMING_OPEN)
    closing_lines = odds_at(odds, TIMING_CLOSE)

    dates = [pd.Timestamp(day) for day in sorted(games["game_date"].unique())]
    if start_date is None:
        cumulative = games.groupby("game_date").size().cumsum()
        eligible = cumulative[cumulative >= config.model.min_train_games]
        start_date = pd.Timestamp(eligible.index[0]) if not eligible.empty else dates[-1]
    start_date = pd.Timestamp(start_date)

    engine = BettingEngine(config=config, timing=timing)
    state = BankrollState(bankroll=config.bankroll.starting_bankroll)
    bundle: ModelBundle | None = None
    last_fit: pd.Timestamp | None = None

    bet_frames: list[pd.DataFrame] = []
    evaluated_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    equity_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []

    for day in dates:
        if day < start_date:
            continue
        history = games[games["game_date"] < day]
        if len(history) < config.model.min_train_games:
            continue
        if bundle is None or last_fit is None or (day - last_fit).days >= retrain_days:
            bundle = ModelBundle.fit(history, config=config)
            last_fit = day
            model_rows.append(
                {"game_date": day, "n_train": len(history), **bundle.totals.summary()}
            )

        today = games[games["game_date"] == day]
        lines_today = betting_lines.reindex(today["game_id"]).dropna(
            subset=["spread_home", "total_line"]
        )
        if lines_today.empty:
            continue
        playable = today[today["game_id"].isin(lines_today.index)]
        predictions = bundle.predict(
            playable,
            spread_home=lines_today.loc[playable["game_id"], "spread_home"].to_numpy(),
            total_line=lines_today.loc[playable["game_id"], "total_line"].to_numpy(),
            seed=config.sim.random_seed + int(day.value % SEED_SPREAD),
        )
        predictions = _merge_context(predictions, playable, opening_lines)
        prediction_frames.append(predictions)

        state = state.new_day().with_stake_multiplier(
            _stake_multiplier(bet_frames, day, config=config)
        )
        result = engine.evaluate(predictions, lines_today, state=state, logged_at=day)
        evaluated_frames.append(result.evaluated)
        settled = settle_bets(
            result.bets,
            playable,
            closing_odds=closing_lines,
            bet_odds=lines_today,
        )
        state = result.state
        if not settled.empty:
            state = state.with_settlement(float(settled["pnl"].sum()))
            bet_frames.append(settled)
        equity_rows.append(
            {
                "game_date": day,
                "bankroll": state.bankroll,
                "n_bets": len(settled),
                "stake": float(settled["stake"].sum()) if not settled.empty else 0.0,
                "pnl": float(settled["pnl"].sum()) if not settled.empty else 0.0,
            }
        )

    bets = (
        pd.concat(bet_frames, ignore_index=True)
        if bet_frames
        else pd.DataFrame(columns=["game_id", "stake", "pnl", "clv", "result"])
    )
    LOGGER.info(
        "backtest terminado: %d apuestas, bankroll final %.2f", len(bets), state.bankroll
    )
    return BacktestResult(
        bets=bets,
        evaluated=(
            pd.concat(evaluated_frames, ignore_index=True)
            if evaluated_frames
            else pd.DataFrame()
        ),
        predictions=(
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        ),
        equity=pd.DataFrame(equity_rows),
        model_log=pd.DataFrame(model_rows),
        starting_bankroll=config.bankroll.starting_bankroll,
    )


__all__ = [
    "CONTEXT_COLUMNS",
    "DEFAULT_RETRAIN_DAYS",
    "BacktestResult",
    "run_walkforward",
]
