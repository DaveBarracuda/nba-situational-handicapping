"""Tests de los tres modelos de mercado (guía 7, 8 y 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from nba_sh.config import CONFIG, Config
from nba_sh.models.bundle import PREDICTION_COLUMNS, ModelBundle
from nba_sh.models.moneyline import (
    CALIBRATION_ISOTONIC,
    CALIBRATION_PLATT,
    resolve_calibration_method,
)
from nba_sh.models.simulate import (
    negbin_parameters,
    negbin_sd,
    simulate_market_probabilities,
    simulate_scores,
)
from nba_sh.models.totals import fit_dispersion_scale, fit_negbin_theta
from nba_sh.odds_utils import round_to_half

SIM_SEED = 11
MU_HOME = 115.0
MU_AWAY = 110.0
LARGE_SIMS = 20_000
COHERENCE_MIN_CORRELATION = 0.9


@pytest.mark.unit
def test_negbin_parameters_match_the_mean_and_variance() -> None:
    """La parametrización da ``var = mu + mu^2/theta``."""
    theta = 200.0
    n, p = negbin_parameters(np.array([MU_HOME]), theta)
    assert nbinom.mean(n[0], p[0]) == pytest.approx(MU_HOME)
    expected_variance = MU_HOME + MU_HOME**2 / theta
    assert nbinom.var(n[0], p[0]) == pytest.approx(expected_variance, rel=1e-6)
    assert negbin_sd(MU_HOME, theta) == pytest.approx(np.sqrt(expected_variance))


@pytest.mark.unit
def test_fit_negbin_theta_recovers_a_known_dispersion() -> None:
    """El MLE recupera la dispersión con la que se generó la muestra."""
    theta_true = 150.0
    rng = np.random.default_rng(SIM_SEED)
    mu = np.full(4_000, MU_HOME)
    n, p = negbin_parameters(mu, theta_true)
    observed = nbinom.rvs(n, p, random_state=rng)
    assert fit_negbin_theta(mu, observed, config=CONFIG) == pytest.approx(
        theta_true, rel=0.45
    )


@pytest.mark.unit
def test_fit_negbin_theta_falls_back_without_enough_data() -> None:
    """Con muy pocos datos se usa el prior, no un ajuste inventado."""
    fitted = fit_negbin_theta(
        np.array([110.0, 115.0]), np.array([110.0, 118.0]), config=CONFIG
    )
    assert fitted == pytest.approx(CONFIG.sim.negbin_theta_prior)


@pytest.mark.unit
def test_dispersion_scale_corrects_underdispersion() -> None:
    """Si los datos están menos dispersos que la NegBin, la escala baja de 1."""
    rng = np.random.default_rng(SIM_SEED)
    mu = np.full(1_000, MU_HOME)
    observed = mu + rng.normal(0.0, 6.0, size=mu.size)
    assert fit_dispersion_scale(mu, observed, 400.0) < 1.0


@pytest.mark.unit
def test_simulated_scores_match_the_requested_mean_and_correlation() -> None:
    """La simulación respeta medias y correlación objetivo."""
    rng = np.random.default_rng(SIM_SEED)
    home, away = simulate_scores(
        MU_HOME, MU_AWAY, theta=400.0, correlation=0.25, n_sims=LARGE_SIMS, rng=rng
    )
    assert home.mean() == pytest.approx(MU_HOME, abs=0.5)
    assert away.mean() == pytest.approx(MU_AWAY, abs=0.5)
    assert np.corrcoef(home, away)[0, 1] == pytest.approx(0.25, abs=0.05)


@pytest.mark.unit
def test_dispersion_scale_shrinks_the_simulated_spread() -> None:
    """La escala de dispersión reduce la desviación sin mover la media."""
    rng = np.random.default_rng(SIM_SEED)
    wide, _ = simulate_scores(
        MU_HOME, MU_AWAY, theta=400.0, correlation=0.2, n_sims=5_000, rng=rng
    )
    rng = np.random.default_rng(SIM_SEED)
    narrow, _ = simulate_scores(
        MU_HOME,
        MU_AWAY,
        theta=400.0,
        correlation=0.2,
        n_sims=5_000,
        rng=rng,
        dispersion_scale=0.7,
    )
    assert narrow.std() < wide.std()
    assert narrow.mean() == pytest.approx(wide.mean(), abs=0.6)


@pytest.mark.unit
def test_market_probabilities_are_coherent() -> None:
    """Las masas de probabilidad son válidas y las medias, las pedidas."""
    summary = simulate_market_probabilities(
        np.array([MU_HOME]),
        np.array([MU_AWAY]),
        spread_home=np.array([-4.5]),
        total_line=np.array([225.0]),
        theta=400.0,
        config=CONFIG.with_sims(5_000),
        seed=SIM_SEED,
    )
    assert len(summary) == 1
    assert 0.0 < summary.p_over[0] + summary.p_push_total[0] <= 1.0
    assert summary.mean_margin[0] == pytest.approx(MU_HOME - MU_AWAY, abs=1.0)
    assert summary.mean_total[0] == pytest.approx(MU_HOME + MU_AWAY, abs=1.5)


@pytest.mark.unit
def test_push_mass_only_exists_on_whole_number_lines() -> None:
    """Un spread de medio punto no puede empatar; uno entero sí (guía 9.3)."""
    config = CONFIG.with_sims(5_000)
    half = simulate_market_probabilities(
        np.array([MU_HOME]),
        np.array([MU_AWAY]),
        spread_home=np.array([-4.5]),
        total_line=np.array([224.5]),
        theta=400.0,
        config=config,
        seed=SIM_SEED,
    )
    whole = simulate_market_probabilities(
        np.array([MU_HOME]),
        np.array([MU_AWAY]),
        spread_home=np.array([-4.0]),
        total_line=np.array([224.0]),
        theta=400.0,
        config=config,
        seed=SIM_SEED,
    )
    assert half.p_push_spread[0] == 0.0
    assert half.p_push_total[0] == 0.0
    assert whole.p_push_spread[0] > 0.0
    assert whole.p_push_total[0] > 0.0


@pytest.mark.unit
def test_missing_lines_give_missing_probabilities() -> None:
    """Sin línea de mercado no se inventa una probabilidad."""
    summary = simulate_market_probabilities(
        np.array([MU_HOME]),
        np.array([MU_AWAY]),
        theta=400.0,
        config=CONFIG.with_sims(500),
        seed=SIM_SEED,
    )
    assert np.isnan(summary.p_over[0])
    assert np.isnan(summary.p_home_cover[0])
    assert not np.isnan(summary.p_home_win[0])


@pytest.mark.unit
def test_simulation_rejects_mismatched_inputs() -> None:
    """Las medias de local y visitante deben tener la misma forma."""
    with pytest.raises(ValueError):
        simulate_market_probabilities(
            np.array([110.0, 112.0]),
            np.array([108.0]),
            theta=400.0,
            config=CONFIG.with_sims(100),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "rows", "expected"),
    [
        ("auto", 100, CALIBRATION_PLATT),
        ("auto", 5_000, CALIBRATION_ISOTONIC),
        ("platt", 5_000, CALIBRATION_PLATT),
        ("isotonic", 10, CALIBRATION_ISOTONIC),
    ],
)
def test_calibration_method_resolution(method: str, rows: int, expected: str) -> None:
    """``auto`` usa Platt con muestras pequeñas e isotónica con grandes."""
    assert resolve_calibration_method(method, rows) == expected


@pytest.mark.unit
def test_unknown_calibration_method_raises() -> None:
    """Un método de calibración desconocido falla rápido."""
    with pytest.raises(ValueError):
        resolve_calibration_method("magic", 1_000)


@pytest.mark.integration
def test_totals_model_parameters_are_sane(bundle: ModelBundle) -> None:
    """Theta, sesgo, dispersión y correlación quedan en rango razonable."""
    totals = bundle.totals
    low, high = CONFIG.sim.negbin_theta_bounds
    assert low <= totals.theta <= high
    assert abs(totals.total_bias) < 15.0
    assert 0.6 <= totals.dispersion_scale <= 1.4
    assert 0.0 <= totals.score_correlation <= 0.6
    assert 8.0 < totals.implied_team_sd() < 16.0


@pytest.mark.integration
def test_spread_model_components_and_monotonicity(
    bundle: ModelBundle, split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """La probabilidad de cubrir baja cuando el local da más puntos."""
    _, test = split
    frame = test.head(50)
    assert np.isfinite(bundle.spread.structural_margin(frame)).all()
    assert np.isfinite(bundle.spread.linear_margin(frame)).all()
    assert np.isfinite(bundle.spread.blended_margin(frame)).all()
    assert 8.0 <= bundle.spread.sigma_structural <= 16.0

    easy = bundle.spread.linear_cover_probability(frame, np.full(len(frame), 10.0))
    hard = bundle.spread.linear_cover_probability(frame, np.full(len(frame), -10.0))
    assert (easy > hard).all()


@pytest.mark.integration
def test_moneyline_probabilities_are_valid(
    bundle: ModelBundle, split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """El submodelo de ML devuelve probabilidades en (0, 1)."""
    _, test = split
    probabilities = bundle.moneyline.predict_proba(test)
    assert bundle.moneyline.is_fitted
    assert ((probabilities > 0.0) & (probabilities < 1.0)).all()


@pytest.mark.integration
def test_bundle_predictions_are_complete_and_coherent(
    bundle: ModelBundle, split: tuple[pd.DataFrame, pd.DataFrame], config: Config
) -> None:
    """Las tres salidas existen y no se contradicen entre sí."""
    _, test = split
    predictions = bundle.predict(
        test,
        spread_home=np.full(len(test), -3.5),
        total_line=np.full(len(test), 224.5),
    )
    assert set(PREDICTION_COLUMNS).issubset(predictions.columns)
    assert len(predictions) == len(test)
    assert predictions["p_home_ml"].between(0.0, 1.0).all()
    assert predictions["p_home_cover"].between(0.0, 1.0).all()
    assert predictions["p_over"].between(0.0, 1.0).all()

    correlation = np.corrcoef(predictions["p_home_ml"], predictions["model_margin"])[
        0, 1
    ]
    assert correlation > COHERENCE_MIN_CORRELATION

    for value in predictions["model_spread_line"]:
        assert value == pytest.approx(round_to_half(value))
    assert predictions["model_total"].between(180.0, 280.0).all()
    assert bundle.is_fitted
    assert config.model.simulation_weight > 0.0


@pytest.mark.integration
def test_bundle_calibrates_the_final_win_probability(bundle: ModelBundle) -> None:
    """La probabilidad final de ML se recalibra, no solo el logit (guía 8.2)."""
    assert bundle.win_calibrator is not None
    assert bundle.summary()["win_probability_calibrated"] is True


@pytest.mark.integration
def test_bundle_handles_an_empty_frame(bundle: ModelBundle) -> None:
    """Un día sin juegos no rompe la predicción."""
    empty = bundle.predict(pd.DataFrame())
    assert empty.empty
    assert list(empty.columns) == list(PREDICTION_COLUMNS)
