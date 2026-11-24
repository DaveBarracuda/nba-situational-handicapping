"""Fixtures compartidas de la suite de tests.

El dataset sintético se genera una sola vez por sesión (es determinista) para
que los tests sean rápidos sin perder realismo: usan el mismo pipeline que
produce el modelo en producción.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nba_sh.config import CONFIG, Config
from nba_sh.ingest.sample_data import SampleDataset, generate_sample_dataset
from nba_sh.models.bundle import ModelBundle
from nba_sh.pipeline import PreparedData, prepare

TEST_SEED = 3
TEST_SIMULATIONS = 300
TRAIN_SPLIT_QUANTILE = 0.8


@pytest.fixture(scope="session")
def config() -> Config:
    """Configuración rápida para tests (pocas simulaciones)."""
    return CONFIG.with_sims(TEST_SIMULATIONS).with_seed(TEST_SEED)


@pytest.fixture(scope="session")
def sample(config: Config) -> SampleDataset:
    """Una temporada sintética completa."""
    return generate_sample_dataset(1, seed=TEST_SEED, config=config)


@pytest.fixture(scope="session")
def two_season_sample(config: Config) -> SampleDataset:
    """Dos temporadas sintéticas (para walk-forward y carryover)."""
    return generate_sample_dataset(2, seed=TEST_SEED, config=config)


@pytest.fixture(scope="session")
def prepared(sample: SampleDataset, config: Config) -> PreparedData:
    """Dataset validado con feature store."""
    return prepare(sample.dataset, config=config)


@pytest.fixture(scope="session")
def games(prepared: PreparedData) -> pd.DataFrame:
    """Tabla de juegos con features."""
    return prepared.games


@pytest.fixture(scope="session")
def team_games(prepared: PreparedData) -> pd.DataFrame:
    """Frame equipo-juego (formato largo)."""
    return prepared.features.team_games


@pytest.fixture(scope="session")
def split(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición temporal: parte antigua para entrenar, reciente para predecir."""
    cutoff = games["game_date"].quantile(TRAIN_SPLIT_QUANTILE)
    return games[games["game_date"] <= cutoff], games[games["game_date"] > cutoff]


@pytest.fixture(scope="session")
def bundle(split: tuple[pd.DataFrame, pd.DataFrame], config: Config) -> ModelBundle:
    """Modelos entrenados con la parte antigua de la temporada."""
    train, _ = split
    return ModelBundle.fit(train, config=config)
