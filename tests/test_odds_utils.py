"""Tests de la matemática de cuotas (guía 4, 8.3 y apéndice A)."""

from __future__ import annotations

import pytest

from nba_sh.odds_utils import (
    american_to_decimal,
    american_to_probability,
    closing_line_value,
    decimal_to_american,
    expected_value,
    fair_two_way,
    hold_percentage,
    probability_to_american,
    remove_vig_multiplicative,
    remove_vig_shin,
    round_to_half,
)

TOLERANCE = 1e-9


@pytest.mark.unit
@pytest.mark.parametrize(
    ("american", "decimal"),
    [(-110.0, 1.909090909090909), (100.0, 2.0), (150.0, 2.5), (-200.0, 1.5)],
)
def test_american_to_decimal_known_values(american: float, decimal: float) -> None:
    """Las conversiones coinciden con los valores de referencia."""
    assert american_to_decimal(american) == pytest.approx(decimal)


@pytest.mark.unit
@pytest.mark.parametrize("american", [-500.0, -110.0, 100.0, 275.0])
def test_american_decimal_roundtrip(american: float) -> None:
    """Americana -> decimal -> americana no pierde información."""
    assert decimal_to_american(american_to_decimal(american)) == pytest.approx(american)


@pytest.mark.unit
@pytest.mark.parametrize("invalid", [0.0, 50.0, -99.0])
def test_invalid_american_odds_raise(invalid: float) -> None:
    """Las cuotas en (-100, 100) no existen."""
    with pytest.raises(ValueError):
        american_to_decimal(invalid)


@pytest.mark.unit
@pytest.mark.parametrize("probability", [0.1, 0.35, 0.5, 0.78])
def test_probability_roundtrip(probability: float) -> None:
    """Probabilidad -> cuota fair -> probabilidad es la identidad."""
    american = probability_to_american(probability)
    assert american_to_probability(american) == pytest.approx(probability)


@pytest.mark.unit
def test_remove_vig_multiplicative_sums_to_one() -> None:
    """La normalización multiplicativa reparte el vig y suma 1."""
    fair = remove_vig_multiplicative([0.55, 0.50])
    assert sum(fair) == pytest.approx(1.0, abs=TOLERANCE)
    assert fair[0] > fair[1]


@pytest.mark.unit
def test_remove_vig_shin_corrects_the_longshot_bias() -> None:
    """Shin suma 1 y recorta al perdedor: el libro infla los longshots.

    Es el resultado estándar del método: respecto a la normalización
    multiplicativa, Shin sube la probabilidad del favorito y baja la del
    longshot, que es justo el sesgo favorito-perdedor que corrige.
    """
    implied = [0.70, 0.35]
    shin = remove_vig_shin(implied)
    multiplicative = remove_vig_multiplicative(implied)
    assert sum(shin) == pytest.approx(1.0, abs=1e-6)
    assert shin[0] >= multiplicative[0]
    assert shin[1] <= multiplicative[1]


@pytest.mark.unit
def test_fair_two_way_removes_the_hold() -> None:
    """Las probabilidades fair de un mercado -110/-110 son 0.5 y 0.5."""
    fair_home, fair_away = fair_two_way(-110.0, -110.0)
    assert fair_home == pytest.approx(0.5)
    assert fair_away == pytest.approx(0.5)
    assert hold_percentage([-110.0, -110.0]) == pytest.approx(0.0476, abs=1e-3)


@pytest.mark.unit
def test_expected_value_sign() -> None:
    """El valor esperado es positivo solo si la probabilidad supera la cuota."""
    assert expected_value(0.60, -110.0) > 0.0
    assert expected_value(0.50, -110.0) < 0.0


@pytest.mark.unit
def test_closing_line_value_positive_when_market_moves_to_us() -> None:
    """Tomar +150 y ver cerrar a -110 es CLV positiva."""
    clv = closing_line_value(150.0, -110.0, closing_opposite_american=-110.0)
    assert clv > 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"), [(5.3, 5.5), (5.1, 5.0), (-6.26, -6.5), (220.24, 220.0)]
)
def test_round_to_half(value: float, expected: float) -> None:
    """El redondeo a medio punto respeta las líneas de mercado."""
    assert round_to_half(value) == pytest.approx(expected)
