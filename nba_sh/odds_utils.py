"""Matemática de cuotas: conversiones, eliminación de vig y CLV.

Guía 4 (eliminación del vig), 8.3 (fair ML y detección de valor) y
apéndice A (fair prob, CLV).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

AMERICAN_BASE = 100.0
MIN_PROBABILITY = 1e-6
MAX_PROBABILITY = 1.0 - 1e-6
SHIN_TOLERANCE = 1e-10
SHIN_MAX_ITER = 100


def american_to_decimal(american: float) -> float:
    """Convierte cuota americana a decimal europea.

    Raises:
        ValueError: si ``american`` está en (-100, 100), rango inexistente.
    """
    value = float(american)
    if -AMERICAN_BASE < value < AMERICAN_BASE:
        raise ValueError(f"invalid american odds: {american!r}")
    if value > 0:
        return 1.0 + value / AMERICAN_BASE
    return 1.0 + AMERICAN_BASE / abs(value)


def decimal_to_american(decimal_odds: float) -> float:
    """Convierte cuota decimal a americana."""
    value = float(decimal_odds)
    if value <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds!r}")
    if value >= 2.0:
        return (value - 1.0) * AMERICAN_BASE
    return -AMERICAN_BASE / (value - 1.0)


def clip_probability(probability: float) -> float:
    """Acota una probabilidad al intervalo abierto (0, 1)."""
    return float(min(max(float(probability), MIN_PROBABILITY), MAX_PROBABILITY))


def american_to_probability(american: float) -> float:
    """Probabilidad implícita (con vig) de una cuota americana."""
    return 1.0 / american_to_decimal(american)


def probability_to_american(probability: float) -> float:
    """Cuota americana *fair* para una probabilidad dada."""
    return decimal_to_american(1.0 / clip_probability(probability))


def probability_to_decimal(probability: float) -> float:
    """Cuota decimal *fair* para una probabilidad dada."""
    return 1.0 / clip_probability(probability)


def remove_vig_multiplicative(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Normalización multiplicativa: ``p_i / Σp`` (guía 4).

    Es el método por defecto del sistema por simplicidad y estabilidad.
    """
    values = np.asarray([clip_probability(p) for p in probabilities], dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("probabilities must sum to a positive number")
    return tuple(float(v) for v in values / total)


def _shin_fair(values: np.ndarray, booksum: float, z: float) -> np.ndarray:
    """Probabilidades fair dadas las implícitas y el parámetro z de Shin."""
    if z <= 0.0:
        return values / booksum
    inner = z**2 + 4.0 * (1.0 - z) * (values**2) / booksum
    return (np.sqrt(np.clip(inner, 0.0, None)) - z) / (2.0 * (1.0 - z))


def _shin_two_way_z(values: np.ndarray, booksum: float) -> float:
    """z analítico (proporción de dinero informado) para dos salidas."""
    denominator = booksum - (values[0] - values[1]) ** 2 / booksum
    if denominator <= 0.0:
        return 0.0
    return float((booksum - 1.0) / denominator)


def remove_vig_shin(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Método de Shin: corrige el sesgo favorito-perdedor.

    Para mercados de dos salidas usa la solución analítica de z; en cualquier
    otro caso (o si el libro no tiene vig) cae al método multiplicativo.
    """
    values = np.asarray([clip_probability(p) for p in probabilities], dtype=float)
    booksum = float(values.sum())
    if values.size != 2 or booksum <= 1.0:
        return remove_vig_multiplicative(probabilities)
    z = min(max(_shin_two_way_z(values, booksum), 0.0), 0.5)
    fair = _shin_fair(values, booksum, z)
    return remove_vig_multiplicative(fair)


def fair_two_way(
    american_a: float, american_b: float, *, method: str = "multiplicative"
) -> tuple[float, float]:
    """Probabilidades fair de un mercado de dos salidas a partir de cuotas.

    Args:
        american_a: cuota americana de la salida A.
        american_b: cuota americana de la salida B.
        method: ``"multiplicative"`` (por defecto) o ``"shin"``.
    """
    implied = (american_to_probability(american_a), american_to_probability(american_b))
    if method == "shin":
        fair = remove_vig_shin(implied)
    elif method == "multiplicative":
        fair = remove_vig_multiplicative(implied)
    else:
        raise ValueError(f"unknown vig removal method: {method!r}")
    return fair[0], fair[1]


def hold_percentage(americans: Iterable[float]) -> float:
    """Vig (hold) del libro: ``Σ p_implícita − 1``."""
    total = sum(american_to_probability(a) for a in americans)
    return float(total - 1.0)


def expected_value(probability: float, american: float) -> float:
    """Valor esperado por unidad apostada: ``p * decimal − 1`` (guía 8.3)."""
    return float(clip_probability(probability) * american_to_decimal(american) - 1.0)


def no_vig_closing_probability(
    closing_american: float, closing_opposite_american: float
) -> float:
    """Probabilidad de cierre sin vig (benchmark de calidad del mercado)."""
    fair, _ = fair_two_way(closing_american, closing_opposite_american)
    return fair


def closing_line_value(
    taken_american: float,
    closing_american: float,
    *,
    closing_opposite_american: float | None = None,
) -> float:
    """CLV en puntos de probabilidad (guía 13).

    Positivo = el mercado cerró con una probabilidad mayor que la implícita de
    la cuota tomada, es decir, la línea se movió a favor de la apuesta.
    """
    taken_prob = american_to_probability(taken_american)
    if closing_opposite_american is None:
        closing_prob = american_to_probability(closing_american)
    else:
        closing_prob = no_vig_closing_probability(
            closing_american, closing_opposite_american
        )
    return float(closing_prob - taken_prob)


def round_to_half(value: float) -> float:
    """Redondea al 0.5 más cercano (líneas de spread/total)."""
    return float(np.round(float(value) * 2.0) / 2.0)


__all__ = [
    "AMERICAN_BASE",
    "american_to_decimal",
    "american_to_probability",
    "clip_probability",
    "closing_line_value",
    "decimal_to_american",
    "expected_value",
    "fair_two_way",
    "hold_percentage",
    "no_vig_closing_probability",
    "probability_to_american",
    "probability_to_decimal",
    "remove_vig_multiplicative",
    "remove_vig_shin",
    "round_to_half",
]
