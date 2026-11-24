"""Métricas de evaluación (guía sección 13).

| Métrica      | Mercado         | Objetivo mínimo            |
|--------------|-----------------|----------------------------|
| Log loss     | ML              | < 0.66                     |
| Brier        | ML              | < 0.215                    |
| MAE          | Spread/Totales  | < 10.5 / < 14 pts          |
| ECE          | Todos           | < 0.02                     |
| CLV          | Todos           | > +1.5% consistente        |
| ROI          | Todos           | > 0 con IC 95% positivo    |
| Drawdown máx | Bankroll        | < 30%                      |
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.odds_utils import clip_probability

DEFAULT_CALIBRATION_BINS = 10
CONFIDENCE_Z = 1.96
PERCENT = 100.0
NAN = float("nan")


def log_loss(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    """Log loss binaria."""
    y = np.asarray(outcomes, dtype=float)
    p = np.array(
        [clip_probability(value) for value in np.asarray(probabilities, dtype=float)]
    )
    if y.size == 0:
        return NAN
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    """Brier score (error cuadrático medio de la probabilidad)."""
    y = np.asarray(outcomes, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.size == 0:
        return NAN
    return float(np.mean((p - y) ** 2))


def calibration_table(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> pd.DataFrame:
    """Tabla de calibración por deciles de probabilidad (guía 8.2)."""
    frame = pd.DataFrame(
        {
            "outcome": np.asarray(outcomes, dtype=float),
            "probability": np.asarray(probabilities, dtype=float),
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["bin", "n", "mean_probability", "hit_rate", "gap"])
    frame["bin"] = pd.cut(frame["probability"], bins=np.linspace(0.0, 1.0, bins + 1))
    grouped = frame.groupby("bin", observed=True).agg(
        n=("outcome", "size"),
        mean_probability=("probability", "mean"),
        hit_rate=("outcome", "mean"),
    )
    grouped["gap"] = (grouped["hit_rate"] - grouped["mean_probability"]).abs()
    return grouped.reset_index()


def expected_calibration_error(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> float:
    """ECE: error de calibración ponderado por tamaño de bin."""
    table = calibration_table(outcomes, probabilities, bins=bins)
    if table.empty:
        return NAN
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"]).sum())


def mean_absolute_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    """MAE en puntos."""
    predicted_array = np.asarray(predicted, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    valid = np.isfinite(predicted_array) & np.isfinite(actual_array)
    if not valid.any():
        return NAN
    return float(np.mean(np.abs(predicted_array[valid] - actual_array[valid])))


@dataclass(frozen=True)
class RoiSummary:
    """ROI con intervalo de confianza (guía 13)."""

    n_bets: int
    total_stake: float
    total_pnl: float
    roi: float
    roi_ci_low: float
    roi_ci_high: float

    @property
    def is_significant(self) -> bool:
        """True si el IC 95% del ROI no incluye 0."""
        return self.roi_ci_low > 0.0 or self.roi_ci_high < 0.0


def roi_summary(bets: pd.DataFrame) -> RoiSummary:
    """ROI sobre el importe arriesgado, con IC 95% normal."""
    if bets.empty or "pnl" not in bets.columns:
        return RoiSummary(0, 0.0, 0.0, NAN, NAN, NAN)
    stakes = bets["stake"].to_numpy(dtype=float)
    pnl = bets["pnl"].to_numpy(dtype=float)
    total_stake = float(stakes.sum())
    total_pnl = float(pnl.sum())
    if total_stake <= 0.0:
        return RoiSummary(len(bets), 0.0, total_pnl, NAN, NAN, NAN)
    unit_returns = pnl / np.where(stakes > 0.0, stakes, 1.0)
    roi = total_pnl / total_stake
    standard_error = (
        float(np.std(unit_returns, ddof=1) / np.sqrt(unit_returns.size))
        if unit_returns.size > 1
        else NAN
    )
    return RoiSummary(
        n_bets=int(len(bets)),
        total_stake=total_stake,
        total_pnl=total_pnl,
        roi=float(roi),
        roi_ci_low=float(roi - CONFIDENCE_Z * standard_error),
        roi_ci_high=float(roi + CONFIDENCE_Z * standard_error),
    )


def equity_curve(bets: pd.DataFrame, *, starting_bankroll: float) -> pd.DataFrame:
    """Curva de capital acumulada por fecha."""
    if bets.empty:
        return pd.DataFrame(columns=["game_date", "pnl", "bankroll", "drawdown"])
    daily = (
        bets.groupby("game_date", as_index=False)["pnl"]
        .sum()
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    daily["bankroll"] = starting_bankroll + daily["pnl"].cumsum()
    running_max = daily["bankroll"].cummax()
    daily["drawdown"] = (daily["bankroll"] - running_max) / running_max
    return daily


def max_drawdown(bets: pd.DataFrame, *, starting_bankroll: float) -> float:
    """Drawdown máximo (fracción positiva, 0.30 = 30%)."""
    curve = equity_curve(bets, starting_bankroll=starting_bankroll)
    if curve.empty:
        return 0.0
    return float(-curve["drawdown"].min())


def clv_summary(bets: pd.DataFrame) -> dict[str, float]:
    """Resumen de CLV: media, mediana y % de apuestas que batieron el cierre."""
    empty = {"mean_clv": NAN, "median_clv": NAN, "beat_close_rate": NAN}
    if bets.empty or "clv" not in bets.columns:
        return empty
    values = bets["clv"].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return empty
    return {
        "mean_clv": float(values.mean()),
        "median_clv": float(np.median(values)),
        "beat_close_rate": float(np.mean(values > 0.0)),
    }


def market_metrics(
    predictions: pd.DataFrame, games: pd.DataFrame, *, config: Config = CONFIG
) -> dict[str, float]:
    """Métricas de modelo por mercado contra los objetivos de la guía."""
    merged = predictions.merge(
        games[["game_id", "home_win", "actual_margin", "actual_total"]],
        on="game_id",
        how="inner",
    ).dropna(subset=["home_win"])
    if merged.empty:
        return {}
    targets = config.targets
    metrics = {
        "n_games": float(len(merged)),
        "log_loss": log_loss(merged["home_win"], merged["p_home_ml"]),
        "brier": brier_score(merged["home_win"], merged["p_home_ml"]),
        "ece": expected_calibration_error(merged["home_win"], merged["p_home_ml"]),
        "mae_margin": mean_absolute_error(
            merged["model_margin"], merged["actual_margin"]
        ),
        "mae_total": mean_absolute_error(merged["model_total"], merged["actual_total"]),
    }
    metrics["log_loss_ok"] = float(metrics["log_loss"] < targets.max_log_loss)
    metrics["brier_ok"] = float(metrics["brier"] < targets.max_brier)
    metrics["ece_ok"] = float(metrics["ece"] < targets.max_ece)
    metrics["mae_margin_ok"] = float(metrics["mae_margin"] < targets.max_mae_spread)
    metrics["mae_total_ok"] = float(metrics["mae_total"] < targets.max_mae_total)
    return metrics


def betting_metrics(
    bets: pd.DataFrame, *, starting_bankroll: float, config: Config = CONFIG
) -> dict[str, float]:
    """Métricas de rendimiento de las apuestas."""
    summary = roi_summary(bets)
    drawdown = max_drawdown(bets, starting_bankroll=starting_bankroll)
    metrics = {
        "n_bets": float(summary.n_bets),
        "total_stake": summary.total_stake,
        "total_pnl": summary.total_pnl,
        "roi": summary.roi,
        "roi_ci_low": summary.roi_ci_low,
        "roi_ci_high": summary.roi_ci_high,
        "roi_significant": float(summary.is_significant),
        "max_drawdown_pct": drawdown * PERCENT,
        "drawdown_ok": float(drawdown * PERCENT < config.targets.max_drawdown_pct),
        "enough_bets_for_conclusions": float(
            summary.n_bets >= config.targets.min_bets_for_conclusions
        ),
    }
    metrics.update(clv_summary(bets))
    if not bets.empty and "result" in bets.columns:
        decided = bets[bets["result"].isin(("win", "loss"))]
        metrics["win_rate"] = (
            float((decided["result"] == "win").mean()) if not decided.empty else NAN
        )
    return metrics


__all__ = [
    "CONFIDENCE_Z",
    "DEFAULT_CALIBRATION_BINS",
    "RoiSummary",
    "betting_metrics",
    "brier_score",
    "calibration_table",
    "clv_summary",
    "equity_curve",
    "expected_calibration_error",
    "log_loss",
    "market_metrics",
    "max_drawdown",
    "mean_absolute_error",
    "roi_summary",
]
