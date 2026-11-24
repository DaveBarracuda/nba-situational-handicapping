"""Orquestación de punta a punta: una función por caso de uso.

Esta capa une ingesta -> validación -> features -> modelos -> motor de
apuestas -> reportes, para que la CLI y el scheduler sean envoltorios finos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from nba_sh.backtest.report import backtest_report
from nba_sh.backtest.walkforward import BacktestResult, run_walkforward
from nba_sh.betting.engine import BettingEngine
from nba_sh.betting.limits import BankrollState
from nba_sh.betting.settle import settle_bets
from nba_sh.config import CONFIG, Config
from nba_sh.features.build import FEATURE_COLUMNS, FeatureStore, build_features
from nba_sh.ingest.sample_data import SampleDataset, generate_sample_dataset
from nba_sh.metrics import market_metrics
from nba_sh.models.bundle import ModelBundle
from nba_sh.schemas import TIMING_CLOSE, TIMING_OPEN, TIMING_PREGAME, Dataset, odds_at
from nba_sh.validation import ValidationReport, validate_dataset

LOGGER = logging.getLogger(__name__)
DEFAULT_SEASONS = 3
DEFAULT_RETRAIN_DAYS = 14
CONTEXT_COLUMNS = (
    "situational_margin_adjustment",
    "situational_total_adjustment",
    "late_scratch_flag_home",
    "late_scratch_flag_away",
    "stage_home",
)
OPENING_RENAMES = {
    "ml_home": "open_ml_home",
    "spread_home": "open_spread_home",
    "total_line": "open_total_line",
}


@dataclass(frozen=True)
class PreparedData:
    """Dataset limpio + feature store + reportes de validación."""

    dataset: Dataset
    features: FeatureStore
    validation: tuple[ValidationReport, ...]

    @property
    def games(self) -> pd.DataFrame:
        """Tabla de juegos con features."""
        return self.features.games


@dataclass(frozen=True)
class DailySlate:
    """Resultado del pipeline diario para una fecha."""

    game_date: pd.Timestamp
    predictions: pd.DataFrame
    bets: pd.DataFrame
    evaluated: pd.DataFrame
    bundle: ModelBundle

    @property
    def n_bets(self) -> int:
        """Apuestas aprobadas."""
        return int(len(self.bets))


def load_sample(
    n_seasons: int = DEFAULT_SEASONS,
    *,
    seed: int | None = None,
    config: Config = CONFIG,
) -> SampleDataset:
    """Genera el dataset sintético de verificación."""
    return generate_sample_dataset(n_seasons, seed=seed, config=config)


def prepare(dataset: Dataset, *, config: Config = CONFIG) -> PreparedData:
    """Valida el dataset y construye el feature store."""
    reference_day = (
        pd.Timestamp(dataset.games["game_date"].max())
        if not dataset.games.empty
        else None
    )
    clean, reports = validate_dataset(dataset, today=reference_day, config=config)
    features = build_features(clean, config=config)
    LOGGER.info(
        "preparado: %d juegos válidos, %d columnas por juego",
        features.n_games,
        features.games.shape[1],
    )
    return PreparedData(dataset=clean, features=features, validation=reports)


def _add_context(
    predictions: pd.DataFrame, playable: pd.DataFrame, opening: pd.DataFrame
) -> pd.DataFrame:
    """Añade contexto situacional y líneas de apertura a las predicciones."""
    available = [column for column in CONTEXT_COLUMNS if column in playable.columns]
    merged = predictions.merge(
        playable[["game_id", *available]], on="game_id", how="left"
    )
    zeros = pd.Series(0.0, index=merged.index)
    late_home = merged.get("late_scratch_flag_home", zeros).fillna(0.0)
    late_away = merged.get("late_scratch_flag_away", zeros).fillna(0.0)
    merged["injury_driven"] = (late_home + late_away) > 0.0
    if not opening.empty:
        merged = merged.merge(
            opening[list(OPENING_RENAMES)].rename(columns=OPENING_RENAMES),
            left_on="game_id",
            right_index=True,
            how="left",
        )
    return merged


def run_daily(
    prepared: PreparedData,
    *,
    game_date: pd.Timestamp | None = None,
    timing: str = TIMING_PREGAME,
    bankroll: float | None = None,
    config: Config = CONFIG,
) -> DailySlate:
    """Pipeline de un día: entrena con el pasado y evalúa la jornada.

    Args:
        prepared: datos ya validados con features.
        game_date: fecha a evaluar; por defecto la última disponible.
        timing: timing de la línea que se usa para apostar.
        bankroll: bankroll actual; por defecto el de la configuración.
        config: configuración del modelo.

    Returns:
        :class:`DailySlate` con predicciones, apuestas y oportunidades.

    Raises:
        ValueError: si no hay histórico, juegos o líneas para esa fecha.
    """
    games = prepared.games
    if games.empty:
        raise ValueError("no hay juegos con features para evaluar")
    target = (
        pd.Timestamp(game_date)
        if game_date is not None
        else pd.Timestamp(games["game_date"].max())
    )

    history = games[games["game_date"] < target]
    if history.empty:
        raise ValueError(f"no hay histórico anterior a {target.date()}")
    today = games[games["game_date"] == target]
    if today.empty:
        raise ValueError(f"no hay juegos el {target.date()}")

    bundle = ModelBundle.fit(history, config=config)
    lines = (
        odds_at(prepared.dataset.odds, timing)
        .reindex(today["game_id"])
        .dropna(subset=["spread_home", "total_line"])
    )
    playable = today[today["game_id"].isin(lines.index)]
    if playable.empty:
        raise ValueError(f"no hay líneas de mercado ({timing}) el {target.date()}")

    predictions = bundle.predict(
        playable,
        spread_home=lines.loc[playable["game_id"], "spread_home"].to_numpy(),
        total_line=lines.loc[playable["game_id"], "total_line"].to_numpy(),
    )
    predictions = _add_context(
        predictions, playable, odds_at(prepared.dataset.odds, TIMING_OPEN)
    )

    engine = BettingEngine(config=config, timing=timing)
    state = BankrollState(
        bankroll=config.bankroll.starting_bankroll if bankroll is None else bankroll
    )
    result = engine.evaluate(predictions, lines, state=state, logged_at=target)
    bets = settle_bets(
        result.bets,
        playable,
        closing_odds=odds_at(prepared.dataset.odds, TIMING_CLOSE),
        bet_odds=lines,
    )
    return DailySlate(
        game_date=target,
        predictions=predictions,
        bets=bets,
        evaluated=result.evaluated,
        bundle=bundle,
    )


def run_backtest(
    prepared: PreparedData,
    *,
    timing: str = TIMING_PREGAME,
    retrain_days: int = DEFAULT_RETRAIN_DAYS,
    config: Config = CONFIG,
) -> tuple[BacktestResult, dict[str, object]]:
    """Ejecuta el backtest walk-forward y devuelve el resultado y su reporte."""
    result = run_walkforward(
        prepared.features,
        prepared.dataset.odds,
        timing=timing,
        retrain_days=retrain_days,
        config=config,
    )
    return result, backtest_report(result, prepared.games, config=config)


def model_quality(prepared: PreparedData, *, config: Config = CONFIG) -> dict[str, float]:
    """Métricas de modelo out-of-sample sobre la última temporada."""
    games = prepared.games
    seasons = sorted(games["season"].unique())
    if len(seasons) < 2:
        raise ValueError("se necesitan 2+ temporadas para evaluar fuera de muestra")
    train = games[games["season"].isin(seasons[:-1])]
    test = games[games["season"] == seasons[-1]]
    bundle = ModelBundle.fit(train, config=config)
    lines = odds_at(prepared.dataset.odds, TIMING_PREGAME).reindex(test["game_id"])
    predictions = bundle.predict(
        test,
        spread_home=lines["spread_home"].to_numpy(),
        total_line=lines["total_line"].to_numpy(),
    )
    metrics = market_metrics(predictions, test, config=config)
    metrics["n_features"] = float(len(FEATURE_COLUMNS))
    return metrics


__all__ = [
    "CONTEXT_COLUMNS",
    "DEFAULT_RETRAIN_DAYS",
    "DEFAULT_SEASONS",
    "DailySlate",
    "PreparedData",
    "load_sample",
    "model_quality",
    "prepare",
    "run_backtest",
    "run_daily",
]
