"""Capa 1: ingesta de data.

- :mod:`nba_sh.ingest.sample_data`: liga sintética determinista (verificación
  inmediata del pipeline completo, sin red).
- :mod:`nba_sh.ingest.nba_api_source`: juegos y box scores reales vía ``nba_api``.
- :mod:`nba_sh.ingest.odds_source`: líneas reales vía The Odds API.
"""

from nba_sh.ingest.sample_data import generate_sample_dataset

__all__ = ["generate_sample_dataset"]
