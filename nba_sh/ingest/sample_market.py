"""Mercado sintético: apertura, T-60 y cierre de ML, spread y totales.

El libro sintético conoce la fuerza de los equipos casi perfectamente, pero
**subvalora** los factores situacionales (spots, lesiones, árbitros) y arrastra
ruido que se reduce hacia el cierre. Eso reproduce dos hechos reales:

1. el edge del handicapping situacional existe pero es pequeño;
2. la línea de cierre es más afilada que la de apertura (de ahí que la CLV
   sea la métrica honesta, guía 13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from nba_sh.config import CONFIG, Config
from nba_sh.odds_utils import probability_to_american, round_to_half
from nba_sh.schemas import TIMING_CLOSE, TIMING_OPEN, TIMING_PREGAME

BOOK_NAME = "synthetic_book"
MARKET_MARGIN_SIGMA = 12.0
STANDARD_PRICE = -110.0
ML_HOLD = 0.045
AMERICAN_ROUNDING_STEP = 5.0
MIN_IMPLIED_PROBABILITY = 0.02
MAX_IMPLIED_PROBABILITY = 0.98

# Fracción del efecto verdadero que el mercado NO incorpora.
SPOT_UNDERPRICE = 0.55
INJURY_UNDERPRICE = 0.30
REFEREE_UNDERPRICE = 0.70

# Error del libro sintético respecto a la "verdad", en puntos. Se calibra para
# que el mercado sea comparable a un modelo bien entrenado (RMSE ~3 pts en
# margen) y para que el cierre sea más afilado que la apertura: así la CLV es
# informativa. Un libro real es bastante más afilado que esto; el demo usa un
# mercado algo más blando para que el motor de valor sea observable.
MARGIN_NOISE_SD = {TIMING_OPEN: 2.6, TIMING_PREGAME: 2.1, TIMING_CLOSE: 1.6}
TOTAL_NOISE_SD = {TIMING_OPEN: 4.3, TIMING_PREGAME: 3.5, TIMING_CLOSE: 2.7}


def _round_american(value: float) -> float:
    """Redondea una cuota americana al múltiplo de 5 válido más cercano."""
    rounded = float(np.round(value / AMERICAN_ROUNDING_STEP) * AMERICAN_ROUNDING_STEP)
    if -100.0 < rounded < 100.0:
        return 100.0 if rounded >= 0.0 else -100.0
    return rounded


def _moneyline_pair(margin: float) -> tuple[float, float]:
    """Cuotas americanas de ML con juice, a partir del margen de mercado."""
    probability = float(norm.cdf(margin / MARKET_MARGIN_SIGMA))
    implied_home = float(
        np.clip(
            probability * (1.0 + ML_HOLD),
            MIN_IMPLIED_PROBABILITY,
            MAX_IMPLIED_PROBABILITY,
        )
    )
    implied_away = float(
        np.clip(
            (1.0 - probability) * (1.0 + ML_HOLD),
            MIN_IMPLIED_PROBABILITY,
            MAX_IMPLIED_PROBABILITY,
        )
    )
    return (
        _round_american(probability_to_american(implied_home)),
        _round_american(probability_to_american(implied_away)),
    )


def build_market(
    truth: pd.DataFrame, rng: np.random.Generator, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Construye la tabla de cuotas para todos los juegos y timings.

    Args:
        truth: tabla de verdad de :func:`nba_sh.ingest.sample_season.simulate_season`.
        rng: generador determinista.
        config: se acepta por simetría de la API; el mercado no depende de ella.

    Returns:
        DataFrame con el esquema ``ODDS_COLUMNS``, tres filas por juego.
    """
    rules = config.validation
    max_spread = rules.max_abs_spread - 0.5
    min_total = rules.min_total_line + 0.5
    max_total = rules.max_total_line - 0.5
    rows: list[dict[str, object]] = []
    for record in truth.to_dict(orient="records"):
        margin_core = (
            float(record["true_margin"])
            - SPOT_UNDERPRICE * float(record["spot_margin"])
            - INJURY_UNDERPRICE * float(record["injury_margin"])
        )
        total_core = float(record["true_total"]) - REFEREE_UNDERPRICE * float(
            record["referee_total"]
        )
        for timing in (TIMING_OPEN, TIMING_PREGAME, TIMING_CLOSE):
            margin = margin_core + float(rng.normal(0.0, MARGIN_NOISE_SD[timing]))
            total = total_core + float(rng.normal(0.0, TOTAL_NOISE_SD[timing]))
            ml_home, ml_away = _moneyline_pair(margin)
            rows.append(
                {
                    "game_id": record["game_id"],
                    "book": BOOK_NAME,
                    "timing": timing,
                    "ml_home": ml_home,
                    "ml_away": ml_away,
                    "spread_home": float(
                        np.clip(round_to_half(-margin), -max_spread, max_spread)
                    ),
                    "spread_home_price": STANDARD_PRICE,
                    "spread_away_price": STANDARD_PRICE,
                    "total_line": float(
                        np.clip(round_to_half(total), min_total, max_total)
                    ),
                    "over_price": STANDARD_PRICE,
                    "under_price": STANDARD_PRICE,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "BOOK_NAME",
    "INJURY_UNDERPRICE",
    "MARKET_MARGIN_SIGMA",
    "ML_HOLD",
    "REFEREE_UNDERPRICE",
    "SPOT_UNDERPRICE",
    "STANDARD_PRICE",
    "build_market",
]
