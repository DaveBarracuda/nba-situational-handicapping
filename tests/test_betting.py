"""Tests del motor de apuestas: valor, Kelly, límites y liquidación."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_sh.betting.edge import (
    BET_COLUMNS,
    REASON_NO_LINE,
    SIDE_AWAY,
    SIDE_HOME,
    SIDE_OVER,
    SIDE_UNDER,
    compute_candidates,
)
from nba_sh.betting.engine import BettingEngine
from nba_sh.betting.kelly import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    KELLY_MAIN,
    KELLY_SITUATIONAL,
    fractional_kelly,
    kelly_fraction,
)
from nba_sh.betting.limits import (
    REASON_MAX_BETS,
    REASON_STOP_LOSS,
    REASON_TEAM_EXPOSURE,
    REASON_TINY_STAKE,
    BankrollState,
    apply_limits,
)
from nba_sh.betting.settle import RESULT_LOSS, RESULT_PUSH, RESULT_WIN, settle_bets
from nba_sh.config import CONFIG, Config
from nba_sh.schemas import MARKET_ML, MARKET_SPREAD, MARKET_TOTAL

BANKROLL = 10_000.0
GAME_ID = "g1"
WIN_PNL_AT_MINUS_110 = 90.909


def _prediction_row(**overrides: object) -> pd.DataFrame:
    """Predicción mínima de un juego para el motor."""
    row = {
        "game_id": GAME_ID,
        "game_date": pd.Timestamp("2023-02-01"),
        "home_team": "BOS",
        "away_team": "MIA",
        "model_margin": 6.0,
        "model_total": 230.0,
        "p_home_ml": 0.70,
        "p_home_cover": 0.60,
        "p_push_spread": 0.0,
        "p_over": 0.62,
        "p_push_total": 0.0,
        "situational_margin_adjustment": 1.0,
        "situational_total_adjustment": 1.0,
        "injury_driven": False,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _lines(**overrides: object) -> pd.DataFrame:
    """Línea de mercado mínima, indexada por ``game_id``."""
    row = {
        "game_id": GAME_ID,
        "ml_home": -150.0,
        "ml_away": 130.0,
        "spread_home": -2.5,
        "spread_home_price": -110.0,
        "spread_away_price": -110.0,
        "total_line": 224.5,
        "over_price": -110.0,
        "under_price": -110.0,
    }
    row.update(overrides)
    return pd.DataFrame([row]).set_index("game_id")


@pytest.mark.unit
def test_kelly_fraction_known_value() -> None:
    """f* = (p*b - q)/b con b=1 y p=0.55 da 0.10."""
    assert kelly_fraction(0.55, 100.0) == pytest.approx(0.10)


@pytest.mark.unit
def test_kelly_is_zero_without_edge() -> None:
    """Sin ventaja no hay stake: Kelly nunca propone apostar en contra."""
    assert kelly_fraction(0.50, -110.0) == 0.0
    assert kelly_fraction(0.40, 100.0) == 0.0


@pytest.mark.unit
def test_fractional_kelly_applies_bucket_and_confidence() -> None:
    """Las fracciones de la guía (0.25 principal, 0.10 situacional) se aplican."""
    main_high = fractional_kelly(
        0.60, 100.0, bankroll=BANKROLL, kind=KELLY_MAIN, confidence=CONFIDENCE_HIGH
    )
    situational_low = fractional_kelly(
        0.60,
        100.0,
        bankroll=BANKROLL,
        kind=KELLY_SITUATIONAL,
        confidence=CONFIDENCE_LOW,
    )
    assert main_high == pytest.approx(
        BANKROLL * 0.20 * CONFIG.bankroll.kelly_fraction_main
    )
    assert situational_low < main_high


@pytest.mark.unit
def test_stake_is_capped_at_two_percent() -> None:
    """Ninguna apuesta pasa del 2% del bankroll (guía 11.2)."""
    decision = apply_limits(
        BANKROLL,
        state=BankrollState(bankroll=BANKROLL),
        teams=("BOS", "MIA"),
        game_id=GAME_ID,
        config=CONFIG,
    )
    assert decision.approved
    assert decision.stake == pytest.approx(BANKROLL * CONFIG.bankroll.max_stake_pct)


@pytest.mark.unit
def test_daily_exposure_limit_is_enforced() -> None:
    """La exposición diaria máxima es el 6% del bankroll."""
    state = BankrollState(
        bankroll=BANKROLL,
        day_staked=CONFIG.bankroll.max_daily_exposure_pct * BANKROLL,
    )
    decision = apply_limits(
        200.0, state=state, teams=("BOS", "MIA"), game_id=GAME_ID, config=CONFIG
    )
    assert decision.rejected


@pytest.mark.unit
def test_team_exposure_limit_is_enforced() -> None:
    """Un mismo equipo no puede superar el 3% de exposición."""
    state = BankrollState(
        bankroll=BANKROLL,
        team_exposure={"BOS": CONFIG.bankroll.max_team_exposure_pct * BANKROLL},
    )
    decision = apply_limits(
        100.0, state=state, teams=("BOS", "MIA"), game_id=GAME_ID, config=CONFIG
    )
    assert decision.rejected
    assert decision.reason == REASON_TEAM_EXPOSURE


@pytest.mark.unit
def test_daily_stop_loss_blocks_new_bets() -> None:
    """Con -4% en el día se deja de apostar (anti-tilt)."""
    state = BankrollState(
        bankroll=BANKROLL, day_pnl=-CONFIG.bankroll.daily_stop_loss_pct * BANKROLL
    )
    decision = apply_limits(
        100.0, state=state, teams=("BOS", "MIA"), game_id=GAME_ID, config=CONFIG
    )
    assert decision.rejected
    assert decision.reason == REASON_STOP_LOSS


@pytest.mark.unit
def test_max_bets_per_day_is_enforced() -> None:
    """El límite diario de apuestas se respeta."""
    state = BankrollState(bankroll=BANKROLL, day_bets=CONFIG.bankroll.max_bets_per_day)
    decision = apply_limits(
        100.0, state=state, teams=("BOS", "MIA"), game_id=GAME_ID, config=CONFIG
    )
    assert decision.reason == REASON_MAX_BETS


@pytest.mark.unit
def test_tiny_stakes_are_rejected() -> None:
    """Un stake insignificante no se registra."""
    decision = apply_limits(
        0.2,
        state=BankrollState(bankroll=BANKROLL),
        teams=("BOS", "MIA"),
        game_id=GAME_ID,
        config=CONFIG,
    )
    assert decision.reason == REASON_TINY_STAKE


@pytest.mark.unit
def test_correlated_markets_of_the_same_game_are_discounted() -> None:
    """La segunda apuesta del mismo juego entra con descuento (error #6)."""
    state = BankrollState(bankroll=BANKROLL)
    first = apply_limits(
        100.0, state=state, teams=("BOS", "MIA"), game_id=GAME_ID, config=CONFIG
    )
    state = state.with_bet(stake=first.stake, teams=("BOS", "MIA"), game_id=GAME_ID)
    second = apply_limits(
        100.0, state=state, teams=("BOS", "MIA"), game_id=GAME_ID, config=CONFIG
    )
    assert second.stake < first.stake


@pytest.mark.unit
def test_bankroll_state_is_immutable() -> None:
    """Registrar apuestas devuelve estados nuevos, no muta el anterior."""
    state = BankrollState(bankroll=BANKROLL)
    updated = state.with_bet(stake=100.0, teams=("BOS",), game_id=GAME_ID)
    assert state.day_staked == 0.0
    assert updated.day_staked == 100.0
    assert updated.with_settlement(50.0).bankroll == BANKROLL + 50.0
    assert updated.new_day().day_bets == 0


@pytest.mark.unit
def test_candidates_cover_every_market_and_side() -> None:
    """Se evalúan las seis salidas del partido, apostables o no."""
    candidates = compute_candidates(_prediction_row(), _lines(), config=CONFIG)
    assert list(candidates.columns) == list(BET_COLUMNS)
    assert len(candidates) == 6
    assert set(candidates["market"]) == {MARKET_ML, MARKET_SPREAD, MARKET_TOTAL}
    assert set(candidates["side"]) == {SIDE_HOME, SIDE_AWAY, SIDE_OVER, SIDE_UNDER}


@pytest.mark.unit
def test_candidates_respect_the_market_thresholds() -> None:
    """Toda oportunidad marcada supera el umbral de su mercado."""
    candidates = compute_candidates(_prediction_row(), _lines(), config=CONFIG)
    accepted = candidates[candidates["is_candidate"]]
    thresholds = {
        MARKET_ML: CONFIG.market.ml_min_prob_edge,
        MARKET_SPREAD: CONFIG.market.spread_min_prob_edge,
        MARKET_TOTAL: CONFIG.market.total_min_prob_edge,
    }
    for row in accepted.to_dict(orient="records"):
        assert row["prob_edge"] >= thresholds[row["market"]]
    spreads = accepted[accepted["market"] == MARKET_SPREAD]
    assert (spreads["points_edge"] >= CONFIG.market.spread_min_points).all()
    totals = accepted[accepted["market"] == MARKET_TOTAL]
    assert (totals["points_edge"] >= CONFIG.market.total_min_points).all()


@pytest.mark.unit
def test_no_edge_means_no_candidate() -> None:
    """Si el modelo coincide con el mercado, no se apuesta nada."""
    predictions = _prediction_row(
        model_margin=2.5,
        model_total=224.5,
        p_home_ml=0.60,
        p_home_cover=0.5,
        p_over=0.5,
    )
    candidates = compute_candidates(predictions, _lines(), config=CONFIG)
    assert not candidates["is_candidate"].any()


@pytest.mark.unit
def test_missing_line_is_logged_as_unplayable() -> None:
    """Un juego sin línea queda registrado como no apostable."""
    candidates = compute_candidates(
        _prediction_row(), pd.DataFrame(columns=["spread_home"]), config=CONFIG
    )
    assert len(candidates) == 1
    assert candidates.loc[0, "rejection_reason"] == REASON_NO_LINE
    assert not bool(candidates.loc[0, "is_candidate"])


@pytest.mark.unit
def test_illiquid_moneyline_odds_are_filtered() -> None:
    """Fuera del rango -400/+400 no se apuesta ML (guía 8.3)."""
    candidates = compute_candidates(
        _prediction_row(p_home_ml=0.95),
        _lines(ml_home=-900.0, ml_away=600.0),
        config=CONFIG,
    )
    moneylines = candidates[candidates["market"] == MARKET_ML]
    assert not moneylines["is_candidate"].any()


@pytest.mark.unit
def test_engine_sizes_and_logs_every_opportunity(config: Config) -> None:
    """El motor registra todo y solo aprueba lo que pasa límites."""
    engine = BettingEngine(config=config)
    result = engine.evaluate(
        _prediction_row(), _lines(), state=BankrollState(bankroll=BANKROLL)
    )
    assert len(result.evaluated) == 6
    assert result.n_bets >= 1
    assert (result.bets["stake"] > 0.0).all()
    assert result.total_stake <= config.bankroll.max_daily_exposure_pct * BANKROLL + 1e-6
    assert result.state.day_bets == result.n_bets


@pytest.mark.unit
def test_engine_sizes_on_the_shrunk_edge(config: Config) -> None:
    """El stake se calcula sobre el edge encogido hacia el mercado."""
    engine = BettingEngine(config=config)
    shrunk = engine._shrunk_probability(  # noqa: SLF001 - verificación del guardrail
        pd.Series({"model_probability": 0.70, "market_fair_probability": 0.50})
    )
    assert shrunk == pytest.approx(0.60)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("market", "side", "line", "margin", "total", "expected"),
    [
        (MARKET_ML, SIDE_HOME, -150.0, 5.0, 220.0, RESULT_WIN),
        (MARKET_ML, SIDE_AWAY, 130.0, 5.0, 220.0, RESULT_LOSS),
        (MARKET_SPREAD, SIDE_HOME, -2.5, 5.0, 220.0, RESULT_WIN),
        (MARKET_SPREAD, SIDE_HOME, -6.0, 5.0, 220.0, RESULT_LOSS),
        (MARKET_SPREAD, SIDE_HOME, -5.0, 5.0, 220.0, RESULT_PUSH),
        (MARKET_SPREAD, SIDE_AWAY, 2.5, -5.0, 220.0, RESULT_WIN),
        (MARKET_TOTAL, SIDE_OVER, 218.5, 5.0, 220.0, RESULT_WIN),
        (MARKET_TOTAL, SIDE_UNDER, 218.5, 5.0, 220.0, RESULT_LOSS),
        (MARKET_TOTAL, SIDE_OVER, 220.0, 5.0, 220.0, RESULT_PUSH),
    ],
)
def test_settlement_resolves_every_market(
    market: str, side: str, line: float, margin: float, total: float, expected: str
) -> None:
    """Ganada, perdida y push se resuelven bien en los tres mercados."""
    bets = pd.DataFrame(
        [
            {
                "game_id": GAME_ID,
                "market": market,
                "side": side,
                "line": line,
                "american": -110.0,
                "stake": 100.0,
            }
        ]
    )
    games = pd.DataFrame(
        [{"game_id": GAME_ID, "actual_margin": margin, "actual_total": total}]
    )
    settled = settle_bets(bets, games)
    assert settled.loc[0, "result"] == expected
    if expected == RESULT_WIN:
        assert settled.loc[0, "pnl"] == pytest.approx(WIN_PNL_AT_MINUS_110, abs=1e-2)
    elif expected == RESULT_LOSS:
        assert settled.loc[0, "pnl"] == pytest.approx(-100.0)
    else:
        assert settled.loc[0, "pnl"] == 0.0


@pytest.mark.unit
def test_clv_is_positive_when_the_line_moves_our_way() -> None:
    """Tomar -2.5 y ver cerrar en -5.5 es CLV positiva en puntos y en prob."""
    bets = pd.DataFrame(
        [
            {
                "game_id": GAME_ID,
                "market": MARKET_SPREAD,
                "side": SIDE_HOME,
                "line": -2.5,
                "american": -110.0,
                "stake": 100.0,
            }
        ]
    )
    games = pd.DataFrame(
        [{"game_id": GAME_ID, "actual_margin": 4.0, "actual_total": 220.0}]
    )
    settled = settle_bets(
        bets,
        games,
        closing_odds=_lines(spread_home=-5.5),
        bet_odds=_lines(spread_home=-2.5),
    )
    assert settled.loc[0, "clv_points"] == pytest.approx(3.0)
    assert settled.loc[0, "clv"] > 0.0


@pytest.mark.unit
def test_clv_is_negative_when_the_line_moves_against_us() -> None:
    """Si el mercado se va en contra, la CLV lo refleja."""
    bets = pd.DataFrame(
        [
            {
                "game_id": GAME_ID,
                "market": MARKET_TOTAL,
                "side": SIDE_OVER,
                "line": 224.5,
                "american": -110.0,
                "stake": 100.0,
            }
        ]
    )
    games = pd.DataFrame(
        [{"game_id": GAME_ID, "actual_margin": 4.0, "actual_total": 220.0}]
    )
    settled = settle_bets(
        bets,
        games,
        closing_odds=_lines(total_line=220.5),
        bet_odds=_lines(total_line=224.5),
    )
    assert settled.loc[0, "clv_points"] == pytest.approx(-4.0)
    assert settled.loc[0, "clv"] < 0.0


@pytest.mark.unit
def test_settlement_handles_unknown_games() -> None:
    """Un juego sin resultado no inventa P&L."""
    bets = pd.DataFrame(
        [
            {
                "game_id": "missing",
                "market": MARKET_ML,
                "side": SIDE_HOME,
                "line": -110.0,
                "american": -110.0,
                "stake": 100.0,
            }
        ]
    )
    settled = settle_bets(
        bets, pd.DataFrame(columns=["game_id", "actual_margin", "actual_total"])
    )
    assert settled.loc[0, "pnl"] == 0.0
    assert np.isnan(settled.loc[0, "clv"])
