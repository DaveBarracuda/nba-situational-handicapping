"""Reglas de peso de los ajustes situacionales (guía 5.5A.3).

Funciones puras, sin pandas: las usan tanto el feature store como el
generador de data sintética, de modo que la combinación de spots sea
exactamente la misma en todas partes.

Reglas implementadas:
    (a) multiplicidad: ``max(|spot|) + 0.4 * segundo_mayor`` por signo
    (b) momentum solo si |z| > 1.5σ
    (c) ajuste * 0.5 si el equipo está fatigado (B2B o 1 día de descanso)
    (e) cap total de |Σ ajustes situacionales| ≤ 4.0 pts
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

SECONDARY_SPOT_WEIGHT_DEFAULT = 0.4
TOTAL_CAP_DEFAULT = 4.0


def combine_same_sign(
    values: Sequence[float], *, secondary_weight: float = SECONDARY_SPOT_WEIGHT_DEFAULT
) -> float:
    """Combina spots del mismo signo sin sumarlos linealmente (regla a).

    Por cada signo: el mayor en magnitud entra completo, el segundo al
    ``secondary_weight``, el resto se ignora (no añaden información nueva).

    >>> round(combine_same_sign([-1.5, -1.0]), 3)
    -1.9
    """
    total = 0.0
    for sign in (1.0, -1.0):
        magnitudes = sorted(
            (abs(value) for value in values if value * sign > 0.0), reverse=True
        )
        if not magnitudes:
            continue
        combined = magnitudes[0]
        if len(magnitudes) > 1:
            combined += secondary_weight * magnitudes[1]
        total += sign * combined
    return float(total)


def apply_fatigue_multiplier(
    value: float, *, is_fatigued: bool, multiplier: float
) -> float:
    """Reduce un ajuste de spot si la fatiga ya castiga al equipo (regla c)."""
    return float(value * multiplier) if is_fatigued else float(value)


def momentum_points(
    momentum_z: float,
    *,
    sigma_gate: float,
    points_per_sigma: float,
    cap: float,
) -> float:
    """Puntos por momentum, solo si el momentum es extremo (regla b).

    Por debajo del gate devuelve 0.0: el rolling de los ratings ya absorbió
    esa información.
    """
    if abs(momentum_z) <= sigma_gate:
        return 0.0
    points = (abs(momentum_z) - sigma_gate) * points_per_sigma
    return float(min(points, cap)) * (1.0 if momentum_z > 0 else -1.0)


def cap_total(value: float, *, cap: float = TOTAL_CAP_DEFAULT) -> float:
    """Aplica el cap duro de ajustes situacionales (regla e)."""
    return float(max(min(value, cap), -cap))


def sum_capped(values: Iterable[float], *, cap: float = TOTAL_CAP_DEFAULT) -> float:
    """Suma ajustes y aplica el cap total."""
    return cap_total(float(sum(values)), cap=cap)


__all__ = [
    "SECONDARY_SPOT_WEIGHT_DEFAULT",
    "TOTAL_CAP_DEFAULT",
    "apply_fatigue_multiplier",
    "cap_total",
    "combine_same_sign",
    "momentum_points",
    "sum_capped",
]
