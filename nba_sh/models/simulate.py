"""Simulación Monte Carlo del partido (guía 7.3).

    Repetir N veces:
      pts_home = sim(Ataque_home vs Defensa_away, ritmo, ruido)
      pts_away = sim(Ataque_away vs Defensa_home, ritmo, ruido)
      correlacionar pts_home y pts_away con r ~ 0.15-0.25 (mismo ritmo)
      guardar total y margen

Detalles de implementación:

* Las marginales son **binomiales negativas** y la correlación se impone con
  una cópula gaussiana, de modo que cada marginal conserva su media.
* La inversa de la binomial negativa se evalúa por tabla de CDF sobre el
  soporte plausible de puntos (40-200) + ``searchsorted``: es exacta y ~50x
  más rápida que ``nbinom.ppf`` elemento a elemento.
* ``dispersion_scale`` es el "ajuste de dispersión" que menciona la guía
  (7.1): escala la desviación respecto a la media para que la varianza
  simulada coincida con la observada, incluso cuando los marcadores están
  *infra*-dispersos respecto a Poisson (que es el caso real de la NBA).
* Como los marcadores simulados son enteros, la masa de empate en spreads o
  totales enteros (push) sale directamente de la simulación (guía 9.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import nbinom, norm

from nba_sh.config import CONFIG, Config

MIN_MU = 60.0
MAX_MU = 160.0
SCORE_SUPPORT_MIN = 40
SCORE_SUPPORT_MAX = 200
SUMMARY_FIELDS = (
    "p_home_win",
    "p_home_cover",
    "p_push_spread",
    "p_over",
    "p_push_total",
    "mean_margin",
    "sd_margin",
    "mean_total",
    "sd_total",
)
_SCORE_SUPPORT = np.arange(SCORE_SUPPORT_MIN, SCORE_SUPPORT_MAX + 1)


@dataclass(frozen=True)
class SimulationSummary:
    """Probabilidades de mercado derivadas de la simulación."""

    p_home_win: np.ndarray
    p_home_cover: np.ndarray
    p_push_spread: np.ndarray
    p_over: np.ndarray
    p_push_total: np.ndarray
    mean_margin: np.ndarray
    sd_margin: np.ndarray
    mean_total: np.ndarray
    sd_total: np.ndarray

    def __len__(self) -> int:
        """Número de juegos simulados."""
        return int(self.p_home_win.size)


def negbin_parameters(mu: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Parámetros ``(n, p)`` de la binomial negativa con media ``mu``.

    Con esta parametrización, ``var = mu + mu^2/theta`` (apéndice A).
    """
    mu_clipped = np.clip(np.asarray(mu, dtype=float), MIN_MU, MAX_MU)
    n = np.full(mu_clipped.shape, float(theta))
    p = float(theta) / (float(theta) + mu_clipped)
    return n, p


def negbin_sd(mu: float, theta: float) -> float:
    """Desviación típica de la binomial negativa con media ``mu``."""
    mean = float(np.clip(mu, MIN_MU, MAX_MU))
    return float(np.sqrt(mean + mean**2 / float(theta)))


def _inverse_cdf_samples(
    mu: float, theta: float, uniforms: np.ndarray
) -> np.ndarray:
    """Muestras exactas de la binomial negativa por tabla de CDF."""
    n, p = negbin_parameters(np.array([mu]), theta)
    cdf = nbinom.cdf(_SCORE_SUPPORT, n[0], p[0])
    indexes = np.searchsorted(cdf, uniforms, side="left")
    return _SCORE_SUPPORT[np.clip(indexes, 0, _SCORE_SUPPORT.size - 1)].astype(float)


def simulate_scores(
    mu_home: float,
    mu_away: float,
    *,
    theta: float,
    correlation: float,
    n_sims: int,
    rng: np.random.Generator,
    dispersion_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simula ``n_sims`` marcadores correlacionados de un partido."""
    base = rng.standard_normal(size=(n_sims, 2))
    second = correlation * base[:, 0] + np.sqrt(1.0 - correlation**2) * base[:, 1]
    raw_home = _inverse_cdf_samples(mu_home, theta, norm.cdf(base[:, 0]))
    raw_away = _inverse_cdf_samples(mu_away, theta, norm.cdf(second))
    if dispersion_scale == 1.0:
        return raw_home, raw_away
    scaled_home = np.rint(mu_home + dispersion_scale * (raw_home - mu_home))
    scaled_away = np.rint(mu_away + dispersion_scale * (raw_away - mu_away))
    floor = float(SCORE_SUPPORT_MIN)
    return np.maximum(scaled_home, floor), np.maximum(scaled_away, floor)


def simulate_market_probabilities(
    mu_home: np.ndarray,
    mu_away: np.ndarray,
    *,
    spread_home: np.ndarray | None = None,
    total_line: np.ndarray | None = None,
    theta: float,
    dispersion_scale: float = 1.0,
    correlation: float | None = None,
    config: Config = CONFIG,
    seed: int | None = None,
) -> SimulationSummary:
    """Simula todos los juegos y devuelve las probabilidades de mercado.

    Args:
        mu_home: puntos esperados del local, por juego.
        mu_away: puntos esperados del visitante, por juego.
        spread_home: línea de spread del local (negativa si es favorito).
        total_line: línea de total.
        theta: dispersión de la binomial negativa.
        dispersion_scale: ajuste de dispersión estimado por el modelo.
        correlation: correlación entre los puntos de ambos equipos; si es
            ``None`` se usa la de la configuración.
        config: configuración (número de simulaciones y correlación).
        seed: semilla; por defecto la de la configuración.

    Returns:
        :class:`SimulationSummary` con arrays alineados con los juegos.
    """
    mu_home = np.asarray(mu_home, dtype=float)
    mu_away = np.asarray(mu_away, dtype=float)
    if mu_home.shape != mu_away.shape:
        raise ValueError("mu_home and mu_away must have the same shape")
    size = mu_home.size
    spreads = (
        np.full(size, np.nan)
        if spread_home is None
        else np.asarray(spread_home, dtype=float)
    )
    totals = (
        np.full(size, np.nan)
        if total_line is None
        else np.asarray(total_line, dtype=float)
    )

    rng = np.random.default_rng(config.sim.random_seed if seed is None else seed)
    score_correlation = (
        config.sim.score_correlation if correlation is None else float(correlation)
    )
    results = {name: np.zeros(size, dtype=float) for name in SUMMARY_FIELDS}

    for index in range(size):
        scores_home, scores_away = simulate_scores(
            float(mu_home[index]),
            float(mu_away[index]),
            theta=theta,
            correlation=score_correlation,
            n_sims=config.sim.n_sims,
            rng=rng,
            dispersion_scale=dispersion_scale,
        )
        margins = scores_home - scores_away
        simulated_totals = scores_home + scores_away
        results["p_home_win"][index] = float(np.mean(margins > 0))
        results["mean_margin"][index] = float(np.mean(margins))
        results["sd_margin"][index] = float(np.std(margins))
        results["mean_total"][index] = float(np.mean(simulated_totals))
        results["sd_total"][index] = float(np.std(simulated_totals))

        spread = spreads[index]
        if np.isnan(spread):
            results["p_home_cover"][index] = np.nan
            results["p_push_spread"][index] = np.nan
        else:
            results["p_home_cover"][index] = float(np.mean(margins + spread > 0))
            results["p_push_spread"][index] = float(np.mean(margins + spread == 0))

        total = totals[index]
        if np.isnan(total):
            results["p_over"][index] = np.nan
            results["p_push_total"][index] = np.nan
        else:
            results["p_over"][index] = float(np.mean(simulated_totals > total))
            results["p_push_total"][index] = float(np.mean(simulated_totals == total))

    return SimulationSummary(**results)


__all__ = [
    "MAX_MU",
    "MIN_MU",
    "SCORE_SUPPORT_MAX",
    "SCORE_SUPPORT_MIN",
    "SUMMARY_FIELDS",
    "SimulationSummary",
    "negbin_parameters",
    "negbin_sd",
    "simulate_market_probabilities",
    "simulate_scores",
]
