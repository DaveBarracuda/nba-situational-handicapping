"""Modelo de TOTALES (guía sección 7).

Los puntos por equipo son conteos con sobre-dispersión, así que la marginal es
una **binomial negativa** con ``var = mu + mu^2/theta`` y ``theta`` estimado
por máxima verosimilitud sobre el histórico. El total del partido se obtiene
simulando ambos equipos de forma correlacionada (el ritmo es común).

Nota sobre ``theta``: la guía cita "theta ~= 35", pero con medias de ~113
puntos eso implicaría una desviación típica de ~22 puntos por equipo, cuando
la real ronda los 12. El modelo respeta la forma funcional de la guía y
estima el parámetro por MLE, que es lo que la propia guía indica hacer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_sh.config import CONFIG, Config
from nba_sh.features.build import TOTAL_FEATURE_COLUMNS
from nba_sh.models.simulate import negbin_parameters, negbin_sd

REQUIRED_COLUMNS = ("pts_exp_home", "pts_exp_away", "total_expected")
MIN_TRAIN_ROWS = 50
MIN_DISPERSION_SCALE = 0.60
MAX_DISPERSION_SCALE = 1.40
MIN_SCORE_CORRELATION = 0.0
MAX_SCORE_CORRELATION = 0.6


def negbin_negative_log_likelihood(
    theta: float, mu: np.ndarray, observed: np.ndarray
) -> float:
    """-log-verosimilitud binomial negativa para una dispersión dada."""
    if theta <= 0.0:
        return float("inf")
    n, p = negbin_parameters(mu, theta)
    logpmf = nbinom.logpmf(observed, n, p)
    if not np.all(np.isfinite(logpmf)):
        return float("inf")
    return float(-np.sum(logpmf))


def fit_negbin_theta(
    mu: np.ndarray, observed: np.ndarray, *, config: Config = CONFIG
) -> float:
    """Estima ``theta`` por MLE dentro de los límites configurados."""
    low, high = config.sim.negbin_theta_bounds
    mu_array = np.asarray(mu, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    valid = np.isfinite(mu_array) & np.isfinite(observed_array)
    if int(valid.sum()) < MIN_TRAIN_ROWS:
        return float(config.sim.negbin_theta_prior)
    result = minimize_scalar(
        lambda log_theta: negbin_negative_log_likelihood(
            float(np.exp(log_theta)), mu_array[valid], observed_array[valid]
        ),
        bounds=(float(np.log(low)), float(np.log(high))),
        method="bounded",
    )
    if not result.success:
        return float(config.sim.negbin_theta_prior)
    return float(np.exp(result.x))


def fit_dispersion_scale(
    mu: np.ndarray, observed: np.ndarray, theta: float
) -> float:
    """Ajuste de dispersión: iguala la varianza simulada a la observada.

    La binomial negativa no puede estar por debajo de Poisson, y los puntos
    de un equipo NBA están justo en ese límite (sd ~ 10-12 con media ~113).
    Este factor escala la desviación respecto a la media para que la
    simulación reproduzca la dispersión real (guía 7.1, "Poisson con ajuste
    de dispersión").
    """
    mu_array = np.asarray(mu, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    valid = np.isfinite(mu_array) & np.isfinite(observed_array)
    if int(valid.sum()) < MIN_TRAIN_ROWS:
        return 1.0
    observed_sd = float(np.std(observed_array[valid] - mu_array[valid]))
    model_sd = negbin_sd(float(np.mean(mu_array[valid])), theta)
    if model_sd <= 0.0:
        return 1.0
    return float(
        np.clip(observed_sd / model_sd, MIN_DISPERSION_SCALE, MAX_DISPERSION_SCALE)
    )


def fit_score_correlation(
    residual_home: np.ndarray, residual_away: np.ndarray, *, config: Config = CONFIG
) -> float:
    """Correlación entre los puntos de ambos equipos (ritmo común).

    La guía la sitúa en 0.15-0.25 como referencia; aquí se **estima** del
    histórico y se usa esa, porque de ella dependen a la vez la dispersión del
    margen y la del total.
    """
    home = np.asarray(residual_home, dtype=float)
    away = np.asarray(residual_away, dtype=float)
    valid = np.isfinite(home) & np.isfinite(away)
    if int(valid.sum()) < MIN_TRAIN_ROWS:
        return float(config.sim.score_correlation)
    matrix = np.corrcoef(home[valid], away[valid])
    if not np.isfinite(matrix[0, 1]):
        return float(config.sim.score_correlation)
    return float(
        np.clip(matrix[0, 1], MIN_SCORE_CORRELATION, MAX_SCORE_CORRELATION)
    )


@dataclass(frozen=True)
class TotalsModel:
    """Modelo de totales: nivel corregido + dispersión estimada."""

    theta: float
    total_bias: float
    n_train: int
    dispersion_scale: float = 1.0
    score_correlation: float = CONFIG.sim.score_correlation
    pipeline: Pipeline | None = None
    feature_columns: tuple[str, ...] = TOTAL_FEATURE_COLUMNS
    config: Config = CONFIG

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        *,
        feature_columns: Sequence[str] = TOTAL_FEATURE_COLUMNS,
        config: Config = CONFIG,
    ) -> TotalsModel:
        """Ajusta nivel, dispersión y la vía aprendida sobre datos pasados."""
        missing = [column for column in REQUIRED_COLUMNS if column not in train.columns]
        if missing:
            raise ValueError(f"totals model requires columns: {missing}")
        usable = train.dropna(subset=["actual_total", "total_expected"])
        if len(usable) < MIN_TRAIN_ROWS:
            return cls(
                theta=float(config.sim.negbin_theta_prior),
                total_bias=0.0,
                n_train=int(len(usable)),
                config=config,
            )
        bias = float((usable["actual_total"] - usable["total_expected"]).mean())
        mu = np.concatenate(
            [
                usable["pts_exp_home"].to_numpy(dtype=float) + bias / 2.0,
                usable["pts_exp_away"].to_numpy(dtype=float) + bias / 2.0,
            ]
        )
        observed = np.concatenate(
            [
                usable["home_score"].to_numpy(dtype=float),
                usable["away_score"].to_numpy(dtype=float),
            ]
        )
        theta = fit_negbin_theta(mu, observed, config=config)
        half = bias / 2.0
        residual_home = (
            usable["home_score"].to_numpy(dtype=float)
            - usable["pts_exp_home"].to_numpy(dtype=float)
            - half
        )
        residual_away = (
            usable["away_score"].to_numpy(dtype=float)
            - usable["pts_exp_away"].to_numpy(dtype=float)
            - half
        )
        columns = tuple(
            column for column in feature_columns if column in usable.columns
        )
        pipeline: Pipeline | None = None
        if columns and len(usable) >= config.model.min_train_games:
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=config.model.ridge_alpha)),
                ]
            )
            pipeline.fit(
                usable[list(columns)].to_numpy(dtype=float),
                usable["actual_total"].to_numpy(dtype=float),
            )
        return cls(
            theta=theta,
            total_bias=bias,
            n_train=int(len(usable)),
            dispersion_scale=fit_dispersion_scale(mu, observed, theta),
            score_correlation=fit_score_correlation(
                residual_home, residual_away, config=config
            ),
            pipeline=pipeline,
            feature_columns=columns,
            config=config,
        )

    @property
    def is_fitted(self) -> bool:
        """True si la vía aprendida pudo entrenarse."""
        return self.pipeline is not None

    def structural_total(self, frame: pd.DataFrame) -> pd.Series:
        """Total por ritmo x eficiencia, con el sesgo de nivel corregido."""
        return frame["total_expected"] + self.total_bias

    def linear_total(self, frame: pd.DataFrame) -> pd.Series:
        """Total del modelo lineal regularizado (vía aprendida)."""
        if self.pipeline is None:
            return self.structural_total(frame)
        predicted = self.pipeline.predict(
            frame[list(self.feature_columns)].to_numpy(dtype=float)
        )
        return pd.Series(predicted, index=frame.index)

    def expected_total(self, frame: pd.DataFrame) -> pd.Series:
        """Total esperado: mezcla de la vía estructural y la aprendida."""
        model = self.config.model
        total_weight = model.total_structural_weight + model.total_linear_weight
        return (
            model.total_structural_weight * self.structural_total(frame)
            + model.total_linear_weight * self.linear_total(frame)
        ) / total_weight

    def expected_points(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Puntos esperados por equipo, repartiendo el total del modelo.

        El reparto usa el margen estructural para que el total simulado sea
        exactamente el del modelo de totales y el margen el del de spread.
        """
        total = self.expected_total(frame)
        margin = frame["margin_expected"]
        return (total + margin) / 2.0, (total - margin) / 2.0

    def implied_team_sd(self) -> float:
        """Desviación típica implícita de los puntos de un equipo."""
        mean_points = float(self.config.power.prior_off_rating)
        return negbin_sd(mean_points, self.theta) * self.dispersion_scale

    def summary(self) -> dict[str, float]:
        """Parámetros del modelo (para versionar artefactos)."""
        return {
            "theta": float(self.theta),
            "total_bias": float(self.total_bias),
            "dispersion_scale": float(self.dispersion_scale),
            "score_correlation": float(self.score_correlation),
            "linear_fitted": float(self.is_fitted),
            "n_features": float(len(self.feature_columns)),
            "n_train": float(self.n_train),
            "implied_team_sd": self.implied_team_sd(),
        }


__all__ = [
    "MAX_DISPERSION_SCALE",
    "MAX_SCORE_CORRELATION",
    "MIN_DISPERSION_SCALE",
    "MIN_SCORE_CORRELATION",
    "MIN_TRAIN_ROWS",
    "REQUIRED_COLUMNS",
    "TotalsModel",
    "fit_dispersion_scale",
    "fit_negbin_theta",
    "fit_score_correlation",
    "negbin_negative_log_likelihood",
]
