"""Los tres modelos de mercado entrenados y evaluados a la vez.

Una sola simulación Monte Carlo por juego alimenta totales, spread y la
probabilidad de victoria, lo que mantiene los tres mercados coherentes entre
sí (si el modelo dice que el local gana por 6, su ML y su total no pueden
contradecirlo).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nba_sh.config import CONFIG, Config
from nba_sh.features.build import FEATURE_COLUMNS
from nba_sh.models.moneyline import MoneylineModel
from nba_sh.models.simulate import simulate_market_probabilities
from nba_sh.models.spread import SpreadModel
from nba_sh.models.totals import TotalsModel
from nba_sh.odds_utils import clip_probability, probability_to_american, round_to_half

# Calibración final de la probabilidad de ML (guía 8.2). Se ajusta sobre el
# tramo más reciente del histórico y con menos simulaciones: bastan para
# estimar dos parámetros de Platt y mantienen el reentrenamiento rápido.
CALIBRATION_MAX_ROWS = 600
CALIBRATION_MIN_ROWS = 150
CALIBRATION_SIMULATIONS = 2_000

PREDICTION_COLUMNS: tuple[str, ...] = (
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "model_margin",
    "model_total",
    "model_spread_line",
    "model_total_line",
    "model_ml_home",
    "model_ml_away",
    "p_home_ml",
    "p_home_ml_submodel",
    "p_home_cover",
    "p_push_spread",
    "p_over",
    "p_push_total",
    "sim_p_home_win",
    "sd_margin",
    "sd_total",
)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """Log-odds acotadas, la escala natural para calibrar con Platt."""
    bounded = np.array(
        [clip_probability(value) for value in np.asarray(probabilities, dtype=float)]
    )
    return np.log(bounded / (1.0 - bounded))


@dataclass(frozen=True)
class ModelBundle:
    """Modelos de ML, spread y totales entrenados sobre la misma ventana."""

    moneyline: MoneylineModel
    spread: SpreadModel
    totals: TotalsModel
    version: str
    win_calibrator: LogisticRegression | None = None
    config: Config = CONFIG

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        *,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        config: Config = CONFIG,
        version: str | None = None,
    ) -> ModelBundle:
        """Entrena los tres modelos con datos estrictamente pasados.

        Tras ajustarlos, recalibra la probabilidad **final** de ML sobre el
        tramo más reciente: la mezcla de tres estimadores queda comprimida
        hacia 0.5 y sin este paso el ECE no cumple el objetivo de la guía.
        """
        bundle = cls(
            moneyline=MoneylineModel.fit(
                train, feature_columns=feature_columns, config=config
            ),
            spread=SpreadModel.fit(
                train, feature_columns=feature_columns, config=config
            ),
            totals=TotalsModel.fit(train, config=config),
            version=version or config.model_version,
            config=config,
        )
        return replace(bundle, win_calibrator=bundle._fit_win_calibrator(train))

    def _fit_win_calibrator(self, train: pd.DataFrame) -> LogisticRegression | None:
        """Ajusta un Platt sobre la probabilidad mezclada (validación temporal)."""
        if not self.is_fitted:
            return None
        usable = train.dropna(subset=["home_win"]).sort_values(["game_date", "game_id"])
        split = int(len(usable) * (1.0 - self.config.model.calibration_fraction))
        holdout = usable.iloc[split:].tail(CALIBRATION_MAX_ROWS)
        if len(holdout) < CALIBRATION_MIN_ROWS:
            return None
        calibration_sims = min(CALIBRATION_SIMULATIONS, self.config.sim.n_sims)
        probabilities = self._blended_win_probability(
            holdout, config=self.config.with_sims(calibration_sims)
        )
        calibrator = LogisticRegression(max_iter=1_000)
        calibrator.fit(
            _logit(probabilities).reshape(-1, 1),
            holdout["home_win"].to_numpy(dtype=float),
        )
        return calibrator

    @property
    def is_fitted(self) -> bool:
        """True si los modelos aprendidos tienen datos suficientes."""
        return self.moneyline.is_fitted and self.spread.is_fitted

    def expected_points(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Puntos esperados por equipo.

        Reparte el total del modelo de totales usando el margen estructural del
        modelo de spread, de modo que la simulación sea coherente con los tres
        mercados por construcción: la suma es el total del modelo y la
        diferencia, su margen.
        """
        total = self.totals.expected_total(frame).to_numpy(dtype=float)
        margin = self.spread.structural_margin(frame)
        return (total + margin) / 2.0, (total - margin) / 2.0

    def _blended_win_probability(
        self, frame: pd.DataFrame, *, config: Config | None = None
    ) -> np.ndarray:
        """Probabilidad mezclada (sin calibrar) de un frame cualquiera."""
        simulation_config = self.config if config is None else config
        mu_home, mu_away = self.expected_points(frame)
        simulation = simulate_market_probabilities(
            mu_home,
            mu_away,
            theta=self.totals.theta,
            dispersion_scale=self.totals.dispersion_scale,
            correlation=self.totals.score_correlation,
            config=simulation_config,
        )
        return self._blend_win_probability(
            self.moneyline.predict_proba(frame), simulation.p_home_win
        )

    def _calibrated_win_probability(self, blended: np.ndarray) -> np.ndarray:
        """Aplica la calibración final, si existe."""
        if self.win_calibrator is None:
            return blended
        calibrated = self.win_calibrator.predict_proba(
            _logit(blended).reshape(-1, 1)
        )[:, 1]
        return np.array([clip_probability(value) for value in calibrated])

    def _blend_win_probability(
        self, submodel: np.ndarray, simulated: np.ndarray
    ) -> np.ndarray:
        """Mezcla la probabilidad del submodelo de ML con la simulada.

        El submodelo (logit calibrado + prior Elo) aporta el matiz situacional;
        la simulación aporta coherencia con el spread y el total del mismo
        partido, que es lo que evita precios contradictorios.
        """
        model = self.config.model
        submodel_weight = model.logit_weight + model.elo_prior_weight
        total_weight = submodel_weight + model.simulation_weight
        blended = (
            submodel_weight * np.asarray(submodel, dtype=float)
            + model.simulation_weight * np.asarray(simulated, dtype=float)
        ) / total_weight
        return np.array([clip_probability(value) for value in blended])

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        spread_home: Sequence[float] | None = None,
        total_line: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Predice los tres mercados para los juegos de ``frame``.

        Args:
            frame: filas del feature store (una por juego).
            spread_home: línea de spread a evaluar (del mercado).
            total_line: línea de total a evaluar (del mercado).
            seed: semilla de la simulación.

        Returns:
            DataFrame con ``PREDICTION_COLUMNS``.
        """
        if frame.empty:
            return pd.DataFrame(columns=list(PREDICTION_COLUMNS))

        mu_home, mu_away = self.expected_points(frame)
        spreads = (
            np.full(len(frame), np.nan)
            if spread_home is None
            else np.asarray(spread_home, dtype=float)
        )
        totals = (
            np.full(len(frame), np.nan)
            if total_line is None
            else np.asarray(total_line, dtype=float)
        )
        simulation = simulate_market_probabilities(
            mu_home,
            mu_away,
            spread_home=spreads,
            total_line=totals,
            theta=self.totals.theta,
            dispersion_scale=self.totals.dispersion_scale,
            correlation=self.totals.score_correlation,
            config=self.config,
            seed=seed,
        )

        model_margin = self.spread.blended_margin(frame)
        model_total = self.totals.expected_total(frame).to_numpy(dtype=float)
        submodel_probability = self.moneyline.predict_proba(frame)
        probability_home = self._calibrated_win_probability(
            self._blend_win_probability(submodel_probability, simulation.p_home_win)
        )
        cover = self.spread.cover_probability(frame, spreads, simulation.p_home_cover)

        return pd.DataFrame(
            {
                "game_id": frame["game_id"].to_numpy(),
                "game_date": frame["game_date"].to_numpy(),
                "home_team": frame["home_team"].to_numpy(),
                "away_team": frame["away_team"].to_numpy(),
                "model_margin": model_margin,
                "model_total": model_total,
                "model_spread_line": [round_to_half(-value) for value in model_margin],
                "model_total_line": [round_to_half(value) for value in model_total],
                "model_ml_home": [
                    probability_to_american(value) for value in probability_home
                ],
                "model_ml_away": [
                    probability_to_american(1.0 - value) for value in probability_home
                ],
                "p_home_ml": probability_home,
                "p_home_ml_submodel": submodel_probability,
                "p_home_cover": cover,
                "p_push_spread": simulation.p_push_spread,
                "p_over": simulation.p_over,
                "p_push_total": simulation.p_push_total,
                "sim_p_home_win": simulation.p_home_win,
                "sd_margin": simulation.sd_margin,
                "sd_total": simulation.sd_total,
            }
        )

    def summary(self) -> dict[str, object]:
        """Resumen de los tres modelos para artefactos versionados."""
        return {
            "version": self.version,
            "win_probability_calibrated": self.win_calibrator is not None,
            "moneyline": self.moneyline.summary(),
            "spread": self.spread.summary(),
            "totals": self.totals.summary(),
        }


__all__ = [
    "CALIBRATION_MAX_ROWS",
    "CALIBRATION_MIN_ROWS",
    "CALIBRATION_SIMULATIONS",
    "PREDICTION_COLUMNS",
    "ModelBundle",
]
