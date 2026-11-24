"""Reportes del backtest (guía 12.3 y 5.5A.3d).

    - ROI por mercado y por tipo de apuesta (B2B, revenge, etapa...)
    - Equity curve con drawdown máximo
    - CLV promedio
    - Sensibilidad: ¿sobrevive el modelo si el edge se reduce 30%?
    - Validación de spots: un spot que no mueve > 0.7 pts con > 200 juegos
      de muestra se descarta (muchos spots del folklore mueren aquí).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sh.backtest.walkforward import BacktestResult
from nba_sh.config import CONFIG, Config
from nba_sh.metrics import betting_metrics, equity_curve, market_metrics, roi_summary
from nba_sh.schemas import MARKET_ML, MARKET_SPREAD, MARKET_TOTAL

EDGE_HAIRCUT = 0.30
SPOT_NAMES = ("lookahead", "letdown", "sandwich", "revenge")
SUBGROUP_COLUMNS = ("group", "value", "n_bets", "stake", "pnl", "roi", "mean_clv")
NAN = float("nan")


def _roi_row(group: str, value: object, bets: pd.DataFrame) -> dict[str, object]:
    """Fila de ROI para un subgrupo."""
    summary = roi_summary(bets)
    mean_clv = NAN
    if "clv" in bets.columns:
        values = bets["clv"].dropna()
        mean_clv = float(values.mean()) if not values.empty else NAN
    return {
        "group": group,
        "value": value,
        "n_bets": summary.n_bets,
        "stake": summary.total_stake,
        "pnl": summary.total_pnl,
        "roi": summary.roi,
        "mean_clv": mean_clv,
    }


def subgroup_report(bets: pd.DataFrame) -> pd.DataFrame:
    """ROI por mercado, lado, confianza, etapa y spots (guía 12.3)."""
    if bets.empty:
        return pd.DataFrame(columns=list(SUBGROUP_COLUMNS))

    rows: list[dict[str, object]] = []
    for column in ("market", "side", "confidence", "stage_home", "timing"):
        if column not in bets.columns:
            continue
        for value, subset in bets.groupby(column, dropna=False):
            rows.append(_roi_row(column, value, subset))

    for flag in ("is_b2b_away", "is_b2b_home", "line_move_confirms", "injury_driven"):
        if flag not in bets.columns:
            continue
        for value, subset in bets.groupby(bets[flag].fillna(False).astype(bool)):
            rows.append(_roi_row(flag, bool(value), subset))

    for spot in SPOT_NAMES:
        columns = [
            name
            for name in (f"spot_{spot}_home", f"spot_{spot}_away")
            if name in bets.columns
        ]
        if not columns:
            continue
        mask = bets[columns].fillna(False).astype(bool).any(axis=1)
        if mask.any():
            rows.append(_roi_row("spot", spot, bets[mask]))
    return pd.DataFrame(rows, columns=list(SUBGROUP_COLUMNS))


def sensitivity_report(
    bets: pd.DataFrame, *, haircut: float = EDGE_HAIRCUT, config: Config = CONFIG
) -> dict[str, float]:
    """ROI si el edge estimado se reduce un ``haircut`` (guía 12.3).

    Las apuestas cuyo edge recortado ya no superaría el umbral de su mercado
    se descartan; el resto conserva su resultado real.
    """
    if bets.empty or "prob_edge" not in bets.columns:
        return {
            "roi_after_haircut": NAN,
            "n_bets_after_haircut": 0.0,
            "pnl_after_haircut": 0.0,
        }
    thresholds = {
        MARKET_ML: config.market.ml_min_prob_edge,
        MARKET_SPREAD: config.market.spread_min_prob_edge,
        MARKET_TOTAL: config.market.total_min_prob_edge,
    }
    shrunk_edge = bets["prob_edge"].to_numpy(dtype=float) * (1.0 - haircut)
    required = bets["market"].map(thresholds).fillna(0.0).to_numpy(dtype=float)
    survivors = bets[shrunk_edge >= required]
    summary = roi_summary(survivors)
    return {
        "roi_after_haircut": summary.roi,
        "n_bets_after_haircut": float(summary.n_bets),
        "pnl_after_haircut": summary.total_pnl,
    }


def validate_spots(games: pd.DataFrame, *, config: Config = CONFIG) -> pd.DataFrame:
    """Valida cada spot por subgrupo histórico (regla (d) de la guía 5.5A.3).

    El efecto se mide sobre el residuo ``margen_real - margen_base`` (lo que
    los ratings NO explican), comparando los juegos con el spot contra los que
    no lo tienen en ninguno de los dos equipos.

    Returns:
        DataFrame con el efecto estimado en puntos, el tamaño de muestra y si
        el spot se mantiene (``keep``).
    """
    required = {"actual_margin", "margin_base"}
    if games.empty or not required.issubset(games.columns):
        return pd.DataFrame(
            columns=["spot", "effect_points", "n_games", "keep", "reason"]
        )

    residual = (games["actual_margin"] - games["margin_base"]).to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for spot in SPOT_NAMES:
        home_column, away_column = f"spot_{spot}_home", f"spot_{spot}_away"
        if home_column not in games.columns or away_column not in games.columns:
            continue
        home_flag = games[home_column].fillna(False).to_numpy(dtype=bool)
        away_flag = games[away_column].fillna(False).to_numpy(dtype=bool)
        neutral = ~home_flag & ~away_flag
        if not neutral.any():
            continue
        baseline = float(np.nanmean(residual[neutral]))

        effects: list[float] = []
        sample = 0
        home_only = home_flag & ~away_flag
        away_only = away_flag & ~home_flag
        if home_only.any():
            effects.append(float(np.nanmean(residual[home_only])) - baseline)
            sample += int(home_only.sum())
        if away_only.any():
            effects.append(baseline - float(np.nanmean(residual[away_only])))
            sample += int(away_only.sum())
        if not effects:
            continue

        effect = float(np.mean(effects))
        enough_sample = sample >= config.spots.min_sample_games
        strong_enough = abs(effect) >= config.spots.min_effect_points
        reason = ""
        if not enough_sample:
            reason = f"muestra insuficiente ({sample} < {config.spots.min_sample_games})"
        elif not strong_enough:
            reason = f"efecto {effect:+.2f} pts < {config.spots.min_effect_points} pts"
        rows.append(
            {
                "spot": spot,
                "effect_points": effect,
                "n_games": sample,
                "keep": bool(enough_sample and strong_enough),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def backtest_report(
    result: BacktestResult, games: pd.DataFrame, *, config: Config = CONFIG
) -> dict[str, object]:
    """Reporte completo del backtest, listo para versionar como artefacto."""
    report: dict[str, object] = {
        "model_version": config.model_version,
        "starting_bankroll": result.starting_bankroll,
        "final_bankroll": result.final_bankroll,
        "n_bets": result.n_bets,
        "n_opportunities": int(len(result.evaluated)),
        "betting": betting_metrics(
            result.bets, starting_bankroll=result.starting_bankroll, config=config
        ),
        "market": market_metrics(result.predictions, games, config=config),
        "sensitivity": sensitivity_report(result.bets, config=config),
        "by_market": subgroup_report(result.bets).to_dict(orient="records"),
        "spot_validation": validate_spots(games, config=config).to_dict(orient="records"),
    }
    curve = equity_curve(result.bets, starting_bankroll=result.starting_bankroll)
    report["equity_points"] = int(len(curve))
    if not result.evaluated.empty and "rejection_reason" in result.evaluated.columns:
        report["rejections"] = (
            result.evaluated["rejection_reason"].value_counts().to_dict()
        )
    return report


__all__ = [
    "EDGE_HAIRCUT",
    "SPOT_NAMES",
    "SUBGROUP_COLUMNS",
    "backtest_report",
    "sensitivity_report",
    "subgroup_report",
    "validate_spots",
]
