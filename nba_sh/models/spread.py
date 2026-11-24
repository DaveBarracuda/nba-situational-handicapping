"""Modelo de SPREAD (guía sección 9).

Dos estimaciones del margen que se combinan:

    margen_estructural = puntos base (ratings) + ajustes situacionales
    margen_lineal      = ridge sobre las mismas features, target = margen real

    P_final(cubrir) = 0.6 * P_simulada + 0.4 * P_modelo_lineal

La probabilidad "gaussiana" de la guía se obtiene de la simulación Monte Carlo
(marcadores enteros), que además da la masa exacta de push en spreads enteros
(guía 9.3). La componente lineal usa la normal con la sigma residual del ridge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_sh.config import CONFIG, Config
from nba_sh.features.build import FEATURE_COLUMNS
from nba_sh.odds_utils import clip_probability

MIN_SIGMA = 8.0
MAX_SIGMA = 16.0


@dataclass(frozen=True)
class SpreadModel:
    """Margen esperado y probabilidad de cubrir."""

    pipeline: Pipeline | None
    margin_bias: float
    sigma_structural: float
    sigma_linear: float
    feature_columns: tuple[str, ...]
    n_train: int
    config: Config = CONFIG

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        *,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        config: Config = CONFIG,
    ) -> SpreadModel:
        """Ajusta el ridge, el sesgo del margen y las sigmas residuales."""
        columns = tuple(feature_columns)
        usable = train.dropna(subset=["actual_margin", "margin_expected"])
        if len(usable) < config.model.min_train_games:
            return cls(
                pipeline=None,
                margin_bias=0.0,
                sigma_structural=config.model.margin_sigma_prior,
                sigma_linear=config.model.margin_sigma_prior,
                feature_columns=columns,
                n_train=int(len(usable)),
                config=config,
            )

        bias = float((usable["actual_margin"] - usable["margin_expected"]).mean())
        structural_residuals = (
            usable["actual_margin"] - usable["margin_expected"] - bias
        ).to_numpy(dtype=float)

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=config.model.ridge_alpha)),
            ]
        )
        matrix = usable[list(columns)].to_numpy(dtype=float)
        target = usable["actual_margin"].to_numpy(dtype=float)
        pipeline.fit(matrix, target)
        linear_residuals = target - pipeline.predict(matrix)

        return cls(
            pipeline=pipeline,
            margin_bias=bias,
            sigma_structural=float(
                np.clip(np.std(structural_residuals), MIN_SIGMA, MAX_SIGMA)
            ),
            sigma_linear=float(np.clip(np.std(linear_residuals), MIN_SIGMA, MAX_SIGMA)),
            feature_columns=columns,
            n_train=int(len(usable)),
            config=config,
        )

    @property
    def is_fitted(self) -> bool:
        """True si el ridge pudo entrenarse."""
        return self.pipeline is not None

    def structural_margin(self, frame: pd.DataFrame) -> np.ndarray:
        """Margen esperado por ratings + situacional, con sesgo corregido."""
        return frame["margin_expected"].to_numpy(dtype=float) + self.margin_bias

    def linear_margin(self, frame: pd.DataFrame) -> np.ndarray:
        """Margen esperado por el modelo lineal regularizado."""
        if self.pipeline is None:
            return self.structural_margin(frame)
        return self.pipeline.predict(
            frame[list(self.feature_columns)].to_numpy(dtype=float)
        )

    def blended_margin(self, frame: pd.DataFrame) -> np.ndarray:
        """Margen final: mezcla estructural + lineal (guía 9.2)."""
        model = self.config.model
        return (
            model.spread_gaussian_weight * self.structural_margin(frame)
            + model.spread_linear_weight * self.linear_margin(frame)
        ) / (model.spread_gaussian_weight + model.spread_linear_weight)

    def linear_cover_probability(
        self, frame: pd.DataFrame, spread_home: np.ndarray
    ) -> np.ndarray:
        """P(cubrir) del modelo lineal: ``P(margen > -spread)``."""
        margin = self.linear_margin(frame)
        spread = np.asarray(spread_home, dtype=float)
        with np.errstate(invalid="ignore"):
            return 1.0 - norm.cdf((-spread - margin) / self.sigma_linear)

    def cover_probability(
        self,
        frame: pd.DataFrame,
        spread_home: np.ndarray,
        simulated_cover: np.ndarray,
    ) -> np.ndarray:
        """Mezcla la probabilidad simulada con la del modelo lineal."""
        model = self.config.model
        linear = self.linear_cover_probability(frame, spread_home)
        simulated = np.asarray(simulated_cover, dtype=float)
        blended = (
            model.spread_gaussian_weight * simulated
            + model.spread_linear_weight * linear
        ) / (model.spread_gaussian_weight + model.spread_linear_weight)
        return np.array(
            [np.nan if np.isnan(value) else clip_probability(value) for value in blended]
        )

    def summary(self) -> dict[str, float]:
        """Resumen para artefactos versionados."""
        return {
            "margin_bias": float(self.margin_bias),
            "sigma_structural": float(self.sigma_structural),
            "sigma_linear": float(self.sigma_linear),
            "n_train": float(self.n_train),
            "fitted": float(self.is_fitted),
        }


__all__ = ["MAX_SIGMA", "MIN_SIGMA", "SpreadModel"]
