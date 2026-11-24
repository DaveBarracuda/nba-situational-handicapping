"""Tests del backtest walk-forward, las métricas y el monitoreo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_sh.backtest.report import (
    backtest_report,
    sensitivity_report,
    subgroup_report,
    validate_spots,
)
from nba_sh.backtest.walkforward import BacktestResult, run_walkforward
from nba_sh.config import CONFIG, Config
from nba_sh.metrics import (
    betting_metrics,
    brier_score,
    calibration_table,
    clv_summary,
    equity_curve,
    expected_calibration_error,
    log_loss,
    market_metrics,
    max_drawdown,
    mean_absolute_error,
    roi_summary,
)
from nba_sh.monitoring import (
    SEVERITY_CRITICAL,
    feature_drift_report,
    late_scratch_alerts,
    monitoring_report,
    population_stability_index,
    rolling_log_loss,
)
from nba_sh.pipeline import PreparedData
from nba_sh.schemas import STATUS_OUT

PERFECT_PROBABILITY = 0.999
DRIFT_SHIFT = 5.0
MONITOR_SPLIT_QUANTILE = 0.7


@pytest.fixture(scope="module")
def backtest(prepared: PreparedData, config: Config) -> BacktestResult:
    """Backtest de una temporada (se reutiliza en varios tests)."""
    return run_walkforward(prepared.features, prepared.dataset.odds, config=config)


@pytest.mark.unit
def test_log_loss_and_brier_reward_accuracy() -> None:
    """Acertar con confianza puntúa mejor que dudar."""
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    sharp = np.array([0.9, 0.1, 0.85, 0.2])
    vague = np.array([0.5, 0.5, 0.5, 0.5])
    assert log_loss(outcomes, sharp) < log_loss(outcomes, vague)
    assert brier_score(outcomes, sharp) < brier_score(outcomes, vague)


@pytest.mark.unit
def test_log_loss_is_bounded_for_extreme_probabilities() -> None:
    """Las probabilidades se acotan: nunca hay log(0)."""
    assert np.isfinite(log_loss(np.array([1.0]), np.array([0.0])))
    assert log_loss(np.array([1.0]), np.array([PERFECT_PROBABILITY])) < 0.01


@pytest.mark.unit
def test_calibration_table_and_ece() -> None:
    """Un modelo calibrado tiene ECE pequeño."""
    rng = np.random.default_rng(7)
    probabilities = rng.uniform(0.1, 0.9, size=4_000)
    outcomes = (rng.uniform(size=probabilities.size) < probabilities).astype(float)
    table = calibration_table(outcomes, probabilities)
    assert not table.empty
    assert table["n"].sum() == probabilities.size
    assert expected_calibration_error(outcomes, probabilities) < 0.05


@pytest.mark.unit
def test_mean_absolute_error_ignores_missing_values() -> None:
    """El MAE ignora los huecos en lugar de propagar NaN."""
    predicted = np.array([1.0, 2.0, np.nan])
    actual = np.array([2.0, 4.0, 10.0])
    assert mean_absolute_error(predicted, actual) == pytest.approx(1.5)


@pytest.mark.unit
def test_roi_summary_and_confidence_interval() -> None:
    """El ROI se mide sobre el importe arriesgado, con IC 95%."""
    bets = pd.DataFrame(
        {
            "stake": [100.0, 100.0, 100.0, 100.0],
            "pnl": [90.0, -100.0, 95.0, 90.0],
            "game_date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ),
        }
    )
    summary = roi_summary(bets)
    assert summary.n_bets == 4
    assert summary.total_stake == 400.0
    assert summary.roi == pytest.approx(0.4375)
    assert summary.roi_ci_low < summary.roi < summary.roi_ci_high


@pytest.mark.unit
def test_equity_curve_and_drawdown() -> None:
    """La curva acumula por fecha y el drawdown mide el peor valle."""
    bets = pd.DataFrame(
        {
            "stake": [100.0, 100.0, 100.0],
            "pnl": [100.0, -400.0, 200.0],
            "game_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        }
    )
    curve = equity_curve(bets, starting_bankroll=1_000.0)
    assert list(curve["bankroll"]) == [1_100.0, 700.0, 900.0]
    assert max_drawdown(bets, starting_bankroll=1_000.0) == pytest.approx(
        400.0 / 1_100.0
    )


@pytest.mark.unit
def test_clv_summary_handles_missing_values() -> None:
    """La CLV resume solo lo observado."""
    summary = clv_summary(pd.DataFrame({"clv": [0.02, -0.01, np.nan]}))
    assert summary["mean_clv"] == pytest.approx(0.005)
    assert summary["beat_close_rate"] == pytest.approx(0.5)


@pytest.mark.unit
def test_empty_inputs_do_not_crash_the_metrics() -> None:
    """Sin apuestas, las métricas devuelven vacíos coherentes."""
    empty = pd.DataFrame(columns=["stake", "pnl", "game_date", "clv"])
    assert roi_summary(empty).n_bets == 0
    assert equity_curve(empty, starting_bankroll=100.0).empty
    assert max_drawdown(empty, starting_bankroll=100.0) == 0.0
    assert np.isnan(clv_summary(empty)["mean_clv"])


@pytest.mark.unit
def test_psi_is_zero_for_identical_distributions() -> None:
    """Sin drift, el PSI es ~0; con drift, supera el umbral de alerta."""
    rng = np.random.default_rng(5)
    base = rng.normal(size=5_000)
    assert population_stability_index(base, base.copy()) == pytest.approx(0.0, abs=1e-9)
    assert (
        population_stability_index(base, base + DRIFT_SHIFT)
        > CONFIG.monitoring.psi_alert_threshold
    )


@pytest.mark.unit
def test_feature_drift_report_flags_the_shifted_feature() -> None:
    """El reporte de drift marca la feature desplazada."""
    rng = np.random.default_rng(5)
    train = pd.DataFrame({"a": rng.normal(size=2_000), "b": rng.normal(size=2_000)})
    live = pd.DataFrame(
        {"a": rng.normal(size=2_000), "b": rng.normal(size=2_000) + DRIFT_SHIFT}
    )
    report = feature_drift_report(train, live, ("a", "b"), config=CONFIG)
    assert list(report[report["alert"]]["feature"]) == ["b"]


@pytest.mark.unit
def test_rolling_log_loss_uses_only_the_window() -> None:
    """El log loss rolling mira solo la ventana reciente."""
    predictions = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2023-01-01", "2023-03-01"]),
            "home_win": [1.0, 1.0],
            "p_home_ml": [0.1, 0.9],
        }
    )
    assert rolling_log_loss(predictions, window_days=15) < log_loss(
        np.array([1.0]), np.array([0.1])
    )


@pytest.mark.unit
def test_late_scratch_alerts_are_critical() -> None:
    """Una baja tardía de un jugador clave dispara alerta crítica."""
    injuries = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "team": "BOS",
                "player_id": "BOS-P01",
                "minutes_pg": 35.0,
                "on_off_net": 6.0,
                "status": STATUS_OUT,
                "hours_before_tip": 1.0,
            }
        ]
    )
    alerts = late_scratch_alerts(injuries, config=CONFIG)
    assert len(alerts) == 1
    assert alerts[0].severity == SEVERITY_CRITICAL
    assert "BOS-P01" in str(alerts[0])


@pytest.mark.integration
def test_walkforward_only_bets_after_the_training_window(
    backtest: BacktestResult, prepared: PreparedData, config: Config
) -> None:
    """No se apuesta hasta tener el histórico mínimo (walk-forward puro)."""
    assert backtest.n_bets > 0
    games = prepared.games
    first_bet_date = backtest.bets["game_date"].min()
    assert len(games[games["game_date"] < first_bet_date]) >= config.model.min_train_games


@pytest.mark.integration
def test_walkforward_bets_are_settled_and_consistent(backtest: BacktestResult) -> None:
    """Cada apuesta tiene stake, resultado y P&L coherentes."""
    bets = backtest.bets
    assert {"stake", "pnl", "result", "clv"}.issubset(bets.columns)
    assert (bets["stake"] > 0.0).all()
    assert bets["result"].isin(("win", "loss", "push")).all()
    losses = bets[bets["result"] == "loss"]
    assert np.allclose(losses["pnl"], -losses["stake"])
    pushes = bets[bets["result"] == "push"]
    assert (pushes["pnl"] == 0.0).all()


@pytest.mark.integration
def test_walkforward_equity_matches_the_settled_pnl(backtest: BacktestResult) -> None:
    """El bankroll final es el inicial más el P&L liquidado."""
    expected = backtest.starting_bankroll + backtest.bets["pnl"].sum()
    assert backtest.final_bankroll == pytest.approx(expected, abs=1e-6)
    assert backtest.equity["game_date"].is_monotonic_increasing


@pytest.mark.integration
def test_walkforward_logs_rejected_opportunities(backtest: BacktestResult) -> None:
    """Las apuestas no tomadas quedan registradas (error común #7)."""
    assert len(backtest.evaluated) > len(backtest.bets)
    assert backtest.evaluated["rejection_reason"].notna().any()
    assert not backtest.model_log.empty


@pytest.mark.integration
def test_backtest_report_has_every_section(
    backtest: BacktestResult, prepared: PreparedData, config: Config
) -> None:
    """El reporte cubre apuestas, modelo, sensibilidad, subgrupos y spots."""
    report = backtest_report(backtest, prepared.games, config=config)
    for section in ("betting", "market", "sensitivity", "by_market", "spot_validation"):
        assert section in report
    assert report["n_bets"] == backtest.n_bets
    assert report["market"]["n_games"] > 0


@pytest.mark.integration
def test_subgroup_and_sensitivity_reports(
    backtest: BacktestResult, config: Config
) -> None:
    """Los subgrupos cubren mercados y la sensibilidad recorta el edge."""
    subgroups = subgroup_report(backtest.bets)
    assert not subgroups.empty
    assert "market" in set(subgroups["group"])
    sensitivity = sensitivity_report(backtest.bets, config=config)
    assert sensitivity["n_bets_after_haircut"] <= backtest.n_bets


@pytest.mark.integration
def test_spot_validation_applies_the_sample_and_effect_rules(
    prepared: PreparedData, config: Config
) -> None:
    """Un spot se mantiene solo con muestra y efecto suficientes (regla d)."""
    table = validate_spots(prepared.games, config=config)
    assert not table.empty
    for row in table.to_dict(orient="records"):
        if row["keep"]:
            assert abs(row["effect_points"]) >= config.spots.min_effect_points
            assert row["n_games"] >= config.spots.min_sample_games
        else:
            assert row["reason"]


@pytest.mark.integration
def test_market_and_betting_metrics_on_the_backtest(
    backtest: BacktestResult, prepared: PreparedData, config: Config
) -> None:
    """Las métricas de la guía se calculan sobre el backtest completo."""
    market = market_metrics(backtest.predictions, prepared.games, config=config)
    assert market["log_loss"] < 1.0
    assert market["mae_margin"] < 15.0
    betting = betting_metrics(
        backtest.bets, starting_bankroll=backtest.starting_bankroll, config=config
    )
    assert betting["n_bets"] == backtest.n_bets
    assert betting["total_stake"] > 0.0


@pytest.mark.integration
def test_monitoring_report_runs_on_the_backtest(
    backtest: BacktestResult, prepared: PreparedData, config: Config
) -> None:
    """El monitoreo produce PSI, log loss rolling y lista de alertas."""
    games = prepared.games
    cutoff = games["game_date"].quantile(MONITOR_SPLIT_QUANTILE)
    report = monitoring_report(
        train=games[games["game_date"] <= cutoff],
        live=games[games["game_date"] > cutoff],
        feature_columns=("margin_base", "elo_margin", "rest_diff"),
        predictions=backtest.predictions.merge(
            games[["game_id", "home_win"]], on="game_id", how="left"
        ),
        bets=backtest.bets,
        injuries=prepared.dataset.injuries,
        baseline_log_loss=config.targets.max_log_loss,
        config=config,
    )
    assert "alerts" in report
    assert isinstance(report["n_alerts"], int)
    assert len(report["drift"]) == 3
