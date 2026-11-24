"""Orquestador de la liga sintética: N temporadas completas y deterministas.

Permite verificar el pipeline entero (features -> ratings -> modelos ->
motor de apuestas -> backtest) sin red y sin credenciales, como pide el
cierre de la guía ("datos de ejemplo integrados para verificación inmediata").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.ingest.sample_market import build_market
from nba_sh.ingest.sample_schedule import ScheduleConfig, build_schedule, season_label
from nba_sh.ingest.sample_season import (
    draw_referee_crews,
    draw_rosters,
    draw_strengths,
    simulate_season,
)
from nba_sh.schemas import Dataset

LOGGER = logging.getLogger(__name__)
DEFAULT_START_YEAR = 2022
DEFAULT_SEASONS = 3


@dataclass(frozen=True)
class SampleDataset:
    """Dataset sintético + tabla de verdad (solo para diagnóstico)."""

    dataset: Dataset
    truth: pd.DataFrame

    @property
    def n_games(self) -> int:
        """Juegos generados."""
        return self.dataset.n_games


def generate_sample_dataset(
    n_seasons: int = DEFAULT_SEASONS,
    *,
    seed: int | None = None,
    start_year: int = DEFAULT_START_YEAR,
    config: Config = CONFIG,
    schedule_config: ScheduleConfig | None = None,
) -> SampleDataset:
    """Genera ``n_seasons`` temporadas sintéticas encadenadas.

    Args:
        n_seasons: número de temporadas (la guía pide >= 3 para backtest).
        seed: semilla; por defecto la de la configuración.
        start_year: año de inicio de la primera temporada.
        config: configuración del modelo (pesos situacionales verdaderos).
        schedule_config: parámetros del calendario.

    Returns:
        :class:`SampleDataset` con juegos, cuotas, lesiones, árbitros y verdad.
    """
    if n_seasons < 1:
        raise ValueError("n_seasons must be >= 1")
    rng = np.random.default_rng(config.sim.random_seed if seed is None else seed)
    schedule_config = schedule_config or ScheduleConfig()
    crews = draw_referee_crews(rng, config=config)

    games_frames: list[pd.DataFrame] = []
    odds_frames: list[pd.DataFrame] = []
    injury_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []

    strengths = None
    for season_index in range(n_seasons):
        year = start_year + season_index
        strengths = draw_strengths(rng, previous=strengths)
        rosters = draw_rosters(rng)
        schedule = build_schedule(year, rng, config=schedule_config)
        simulated = simulate_season(
            schedule, strengths, crews, rng, rosters=rosters, config=config
        )
        games_frames.append(simulated.games)
        injury_frames.append(simulated.injuries)
        truth_frames.append(simulated.truth)
        odds_frames.append(build_market(simulated.truth, rng, config=config))
        LOGGER.info(
            "temporada %s generada: %d juegos",
            season_label(year),
            len(simulated.games),
        )

    dataset = Dataset(
        games=pd.concat(games_frames, ignore_index=True),
        odds=pd.concat(odds_frames, ignore_index=True),
        injuries=pd.concat(injury_frames, ignore_index=True),
        referees=crews,
    )
    truth = pd.concat(truth_frames, ignore_index=True)
    return SampleDataset(dataset=dataset, truth=truth)


__all__ = [
    "DEFAULT_SEASONS",
    "DEFAULT_START_YEAR",
    "SampleDataset",
    "generate_sample_dataset",
]
