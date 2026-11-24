"""NBA Situational Handicapping — modelo completo (ML, spread y totales).

Implementación del sistema descrito en ``guia_nba_situational_handicapping.md``.

Capas (ver sección 2 de la guía):
    1. Ingesta           -> :mod:`nba_sh.ingest`
    2. Almacén raw       -> :mod:`nba_sh.storage`
    3. Limpieza/validac. -> :mod:`nba_sh.validation`
    4. Feature store     -> :mod:`nba_sh.features`
    5. Modelos           -> :mod:`nba_sh.ratings`, :mod:`nba_sh.models`
    6. Motor de apuestas -> :mod:`nba_sh.betting`
    7. Backtest/monitoreo-> :mod:`nba_sh.backtest`, :mod:`nba_sh.monitoring`

Uso rápido::

    python -m nba_sh demo
    python -m nba_sh backtest --seasons 3
"""

from nba_sh.config import CONFIG, Config

__all__ = ["CONFIG", "Config", "__version__"]
__version__ = "1.0.0"
