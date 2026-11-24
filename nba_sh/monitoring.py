"""Monitoreo en producción (guía 14.2).

    - Data drift:   PSI > 0.1 = alerta
    - Model drift:  log loss rolling 15 días vs. línea base
    - Market drift: tu edge medio vs. hace 90 días
    - Alertas:      late scratch de jugador con on/off > +3 en un juego ya
                    evaluado -> re-correr el modelo en < 5 min
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.metrics import log_loss

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"
PSI_EPSILON = 1e-6
DEFAULT_EDGE_WINDOW_DAYS = 90
NAN = float("nan")


@dataclass(frozen=True)
class Alert:
    """Alerta de monitoreo."""

    kind: str
    severity: str
    message: str
    value: float

    def __str__(self) -> str:
        """Representación legible para logs y reportes."""
        return f"[{self.severity}] {self.kind}: {self.message} ({self.value:.4f})"


def population_stability_index(
    expected: Sequence[float], actual: Sequence[float], *, bins: int = 10
) -> float:
    """PSI entre la distribución de entrenamiento y la de producción.

    > 0.10 indica desplazamiento relevante (guía 14.2).
    """
    expected_array = np.asarray(expected, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    expected_array = expected_array[np.isfinite(expected_array)]
    actual_array = actual_array[np.isfinite(actual_array)]
    if expected_array.size == 0 or actual_array.size == 0:
        return NAN
    quantiles = np.unique(np.quantile(expected_array, np.linspace(0.0, 1.0, bins + 1)))
    if quantiles.size < 3:
        return 0.0
    expected_share = (
        np.histogram(expected_array, bins=quantiles)[0] / expected_array.size
    )
    actual_share = np.histogram(actual_array, bins=quantiles)[0] / actual_array.size
    expected_share = np.clip(expected_share, PSI_EPSILON, None)
    actual_share = np.clip(actual_share, PSI_EPSILON, None)
    return float(
        np.sum((actual_share - expected_share) * np.log(actual_share / expected_share))
    )


def feature_drift_report(
    train: pd.DataFrame,
    live: pd.DataFrame,
    columns: Sequence[str],
    *,
    config: Config = CONFIG,
) -> pd.DataFrame:
    """PSI por feature entre entrenamiento y producción."""
    rows: list[dict[str, object]] = []
    for column in columns:
        if column not in train.columns or column not in live.columns:
            continue
        psi = population_stability_index(
            train[column].to_numpy(dtype=float),
            live[column].to_numpy(dtype=float),
            bins=config.monitoring.psi_bins,
        )
        rows.append(
            {
                "feature": column,
                "psi": psi,
                "alert": bool(
                    np.isfinite(psi) and psi > config.monitoring.psi_alert_threshold
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "psi", "alert"])
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def rolling_log_loss(
    predictions: pd.DataFrame, *, window_days: int, as_of: pd.Timestamp | None = None
) -> float:
    """Log loss de la ventana reciente (model drift)."""
    if predictions.empty or "home_win" not in predictions.columns:
        return NAN
    reference = (
        pd.Timestamp(predictions["game_date"].max())
        if as_of is None
        else pd.Timestamp(as_of)
    )
    window = predictions[
        predictions["game_date"] > reference - pd.Timedelta(days=window_days)
    ].dropna(subset=["home_win", "p_home_ml"])
    if window.empty:
        return NAN
    return log_loss(window["home_win"].to_numpy(), window["p_home_ml"].to_numpy())


def edge_drift(
    bets: pd.DataFrame, *, window_days: int = DEFAULT_EDGE_WINDOW_DAYS
) -> float:
    """Diferencia entre el edge medio reciente y el de la ventana anterior."""
    if bets.empty or "prob_edge" not in bets.columns:
        return NAN
    reference = pd.Timestamp(bets["game_date"].max())
    recent = bets[bets["game_date"] > reference - pd.Timedelta(days=window_days)]
    previous = bets[
        (bets["game_date"] <= reference - pd.Timedelta(days=window_days))
        & (bets["game_date"] > reference - pd.Timedelta(days=2 * window_days))
    ]
    if recent.empty or previous.empty:
        return NAN
    return float(recent["prob_edge"].mean() - previous["prob_edge"].mean())


def late_scratch_alerts(
    injuries: pd.DataFrame, *, config: Config = CONFIG
) -> list[Alert]:
    """Alerta por bajas tardías de jugadores con on/off alto (guía 14.2)."""
    if injuries.empty:
        return []
    late = injuries[
        (injuries["hours_before_tip"] < config.injury.late_scratch_hours)
        & (injuries["on_off_net"] >= config.monitoring.late_scratch_on_off_alert)
    ]
    return [
        Alert(
            kind="late_scratch",
            severity=SEVERITY_CRITICAL,
            message=(
                f"{row['player_id']} ({row['team']}) fuera a "
                f"{row['hours_before_tip']:.1f}h del tip; re-correr el modelo"
            ),
            value=float(row["on_off_net"]),
        )
        for row in late.to_dict(orient="records")
    ]


def monitoring_report(
    *,
    train: pd.DataFrame,
    live: pd.DataFrame,
    feature_columns: Sequence[str],
    predictions: pd.DataFrame,
    bets: pd.DataFrame,
    injuries: pd.DataFrame | None = None,
    baseline_log_loss: float | None = None,
    config: Config = CONFIG,
) -> dict[str, object]:
    """Reporte de monitoreo con sus alertas."""
    drift = feature_drift_report(train, live, feature_columns, config=config)
    window_log_loss = rolling_log_loss(
        predictions, window_days=config.monitoring.logloss_window_days
    )
    alerts: list[Alert] = [
        Alert(
            kind="data_drift",
            severity=SEVERITY_WARNING,
            message=f"PSI alto en {row['feature']}",
            value=float(row["psi"]),
        )
        for row in drift[drift["alert"]].to_dict(orient="records")
    ]
    if (
        baseline_log_loss is not None
        and np.isfinite(window_log_loss)
        and window_log_loss - baseline_log_loss > config.monitoring.logloss_alert_delta
    ):
        alerts.append(
            Alert(
                kind="model_drift",
                severity=SEVERITY_WARNING,
                message="log loss rolling peor que la línea base",
                value=float(window_log_loss - baseline_log_loss),
            )
        )
    if not bets.empty and "clv" in bets.columns and bets["clv"].notna().any():
        mean_clv = float(bets["clv"].dropna().mean())
        if mean_clv < config.monitoring.clv_alert_threshold:
            alerts.append(
                Alert(
                    kind="clv_degradation",
                    severity=SEVERITY_CRITICAL,
                    message="CLV media negativa: reducir stakes 50% y auditar",
                    value=mean_clv,
                )
            )
    if injuries is not None:
        alerts.extend(late_scratch_alerts(injuries, config=config))

    return {
        "rolling_log_loss": window_log_loss,
        "edge_drift": edge_drift(bets),
        "drift": drift.to_dict(orient="records"),
        "alerts": [str(alert) for alert in alerts],
        "n_alerts": len(alerts),
    }


__all__ = [
    "Alert",
    "DEFAULT_EDGE_WINDOW_DAYS",
    "SEVERITY_CRITICAL",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "edge_drift",
    "feature_drift_report",
    "late_scratch_alerts",
    "monitoring_report",
    "population_stability_index",
    "rolling_log_loss",
]
