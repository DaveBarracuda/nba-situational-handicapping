"""Modelo de MONEYLINE (guía sección 8).

Dos enfoques complementarios, combinados:

1. **Elo dinámico** -> probabilidad directa, buen prior y robusto.
2. **Regresión logística** sobre las features situacionales -> captura lo que
   el Elo no ve (fatiga, spots, lesiones, motivación).

La calibración es obligatoria (guía 8.2): el logit se ajusta con la parte
antigua del histórico y se calibra con el tramo más reciente (validación
temporal, nunca aleatoria), con isotónica o Platt según configuración.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_sh.config import CONFIG, Config
from nba_sh.features.build import FEATURE_COLUMNS
from nba_sh.odds_utils import clip_probability

CALIBRATION_ISOTONIC = "isotonic"
CALIBRATION_PLATT = "platt"
CALIBRATION_AUTO = "auto"
MIN_CALIBRATION_ROWS = 120
# Por debajo de este tamaño la isotónica degenera en una escalera de pocos
# valores, así que "auto" se pasa a Platt (2 parámetros, suave y estable).
ISOTONIC_MIN_ROWS = 600
ISOTONIC_FLOOR = 0.01
ISOTONIC_CEILING = 0.99


def resolve_calibration_method(method: str, n_rows: int) -> str:
    """Decide el método de calibración efectivo.

    Args:
        method: ``"auto"``, ``"isotonic"`` o ``"platt"``.
        n_rows: filas disponibles para calibrar.

    Returns:
        ``"isotonic"`` o ``"platt"``.

    Raises:
        ValueError: si el método no se reconoce.
    """
    if method == CALIBRATION_AUTO:
        return (
            CALIBRATION_ISOTONIC if n_rows >= ISOTONIC_MIN_ROWS else CALIBRATION_PLATT
        )
    if method in (CALIBRATION_ISOTONIC, CALIBRATION_PLATT):
        return method
    raise ValueError(f"unknown calibration method: {method!r}")


@dataclass(frozen=True)
class MoneylineModel:
    """Logit calibrado mezclado con el prior Elo."""

    pipeline: Pipeline | None
    calibrator: IsotonicRegression | LogisticRegression | None
    feature_columns: tuple[str, ...]
    n_train: int
    calibration_used: str = CALIBRATION_PLATT
    config: Config = CONFIG

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        *,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        config: Config = CONFIG,
    ) -> MoneylineModel:
        """Entrena el logit y su calibrador con validación temporal."""
        columns = tuple(feature_columns)
        usable = train.dropna(subset=["home_win"]).sort_values(["game_date", "game_id"])
        if len(usable) < config.model.min_train_games:
            return cls(
                pipeline=None,
                calibrator=None,
                feature_columns=columns,
                n_train=int(len(usable)),
                config=config,
            )

        split = int(len(usable) * (1.0 - config.model.calibration_fraction))
        fit_part = usable.iloc[:split]
        calibration_part = usable.iloc[split:]
        if len(calibration_part) < MIN_CALIBRATION_ROWS:
            fit_part, calibration_part = usable, usable.iloc[0:0]

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=config.model.logit_c, max_iter=2_000, solver="lbfgs"
                    ),
                ),
            ]
        )
        pipeline.fit(
            fit_part[list(columns)].to_numpy(dtype=float),
            fit_part["home_win"].to_numpy(dtype=float),
        )

        calibrator: IsotonicRegression | LogisticRegression | None = None
        method = resolve_calibration_method(
            config.model.calibration_method, len(calibration_part)
        )
        if not calibration_part.empty:
            raw = pipeline.predict_proba(
                calibration_part[list(columns)].to_numpy(dtype=float)
            )[:, 1]
            target = calibration_part["home_win"].to_numpy(dtype=float)
            if method == CALIBRATION_ISOTONIC:
                calibrator = IsotonicRegression(
                    y_min=ISOTONIC_FLOOR, y_max=ISOTONIC_CEILING, out_of_bounds="clip"
                )
                calibrator.fit(raw, target)
            else:
                calibrator = LogisticRegression(max_iter=1_000)
                calibrator.fit(raw.reshape(-1, 1), target)

        return cls(
            pipeline=pipeline,
            calibrator=calibrator,
            feature_columns=columns,
            n_train=int(len(usable)),
            calibration_used=method,
            config=config,
        )

    @property
    def is_fitted(self) -> bool:
        """True si hubo datos suficientes para entrenar el logit."""
        return self.pipeline is not None

    def calibrated_probability(self, frame: pd.DataFrame) -> np.ndarray:
        """Probabilidad del logit tras calibración."""
        if self.pipeline is None:
            return frame["p_elo_home"].to_numpy(dtype=float)
        raw = self.pipeline.predict_proba(
            frame[list(self.feature_columns)].to_numpy(dtype=float)
        )[:, 1]
        if self.calibrator is None:
            return raw
        if isinstance(self.calibrator, IsotonicRegression):
            return self.calibrator.predict(raw)
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Probabilidad final de victoria local (logit calibrado + Elo)."""
        elo_probability = frame["p_elo_home"].to_numpy(dtype=float)
        if not self.is_fitted:
            return np.array([clip_probability(value) for value in elo_probability])
        model = self.config.model
        blended = (
            model.logit_weight * self.calibrated_probability(frame)
            + model.elo_prior_weight * elo_probability
        ) / (model.logit_weight + model.elo_prior_weight)
        return np.array([clip_probability(value) for value in blended])

    def summary(self) -> dict[str, float | str]:
        """Resumen para artefactos versionados."""
        return {
            "n_train": float(self.n_train),
            "fitted": float(self.is_fitted),
            "calibration": self.calibration_used,
            "logit_weight": float(self.config.model.logit_weight),
            "elo_weight": float(self.config.model.elo_prior_weight),
        }


__all__ = [
    "CALIBRATION_AUTO",
    "CALIBRATION_ISOTONIC",
    "CALIBRATION_PLATT",
    "ISOTONIC_MIN_ROWS",
    "MIN_CALIBRATION_ROWS",
    "MoneylineModel",
    "resolve_calibration_method",
]
