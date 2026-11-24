"""Kelly fraccionado y sizing por confianza (guía 11.1 y 11.3).

    f* = (p * b - q) / b        b = cuota decimal - 1, q = 1 - p
    Stake = bankroll * f* * fracción

Fracciones: 0.25 en mercados principales, 0.15 en derivados y 0.10 cuando el
edge viene solo de información situacional (scratch tardío).
"""

from __future__ import annotations

from nba_sh.config import CONFIG, Config
from nba_sh.odds_utils import american_to_decimal, clip_probability

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_LEVELS = (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW)

KELLY_MAIN = "main"
KELLY_DERIVED = "derived"
KELLY_SITUATIONAL = "situational"


def kelly_fraction(probability: float, american: float) -> float:
    """Fracción de Kelly completa (0 si el valor esperado es negativo)."""
    probability = clip_probability(probability)
    net_odds = american_to_decimal(american) - 1.0
    if net_odds <= 0.0:
        return 0.0
    fraction = (probability * net_odds - (1.0 - probability)) / net_odds
    return float(max(fraction, 0.0))


def kelly_multiplier(kind: str, *, config: Config = CONFIG) -> float:
    """Fracción de Kelly aplicable al tipo de mercado."""
    bankroll = config.bankroll
    if kind == KELLY_MAIN:
        return bankroll.kelly_fraction_main
    if kind == KELLY_DERIVED:
        return bankroll.kelly_fraction_derived
    if kind == KELLY_SITUATIONAL:
        return bankroll.kelly_fraction_situational
    raise ValueError(f"unknown kelly bucket: {kind!r}")


def confidence_multiplier(level: str, *, config: Config = CONFIG) -> float:
    """Multiplicador de stake por nivel de confianza (guía 11.3)."""
    bankroll = config.bankroll
    if level == CONFIDENCE_HIGH:
        return bankroll.confidence_multiplier_high
    if level == CONFIDENCE_MEDIUM:
        return bankroll.confidence_multiplier_medium
    if level == CONFIDENCE_LOW:
        return bankroll.confidence_multiplier_low
    raise ValueError(f"unknown confidence level: {level!r}")


def fractional_kelly(
    probability: float,
    american: float,
    *,
    bankroll: float,
    kind: str = KELLY_MAIN,
    confidence: str = CONFIDENCE_MEDIUM,
    config: Config = CONFIG,
) -> float:
    """Stake recomendado antes de aplicar límites de riesgo.

    Args:
        probability: probabilidad del modelo para la salida apostada.
        american: cuota americana disponible.
        bankroll: bankroll actual.
        kind: ``"main"``, ``"derived"`` o ``"situational"``.
        confidence: ``"high"``, ``"medium"`` o ``"low"``.
        config: configuración del modelo.

    Returns:
        Stake en unidades monetarias (0 si no hay ventaja).
    """
    fraction = kelly_fraction(probability, american)
    if fraction <= 0.0:
        return 0.0
    stake = (
        bankroll
        * fraction
        * kelly_multiplier(kind, config=config)
        * confidence_multiplier(confidence, config=config)
    )
    return float(max(stake, 0.0))


__all__ = [
    "CONFIDENCE_HIGH",
    "CONFIDENCE_LEVELS",
    "CONFIDENCE_LOW",
    "CONFIDENCE_MEDIUM",
    "KELLY_DERIVED",
    "KELLY_MAIN",
    "KELLY_SITUATIONAL",
    "confidence_multiplier",
    "fractional_kelly",
    "kelly_fraction",
    "kelly_multiplier",
]
