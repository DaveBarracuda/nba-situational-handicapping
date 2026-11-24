"""Capa 4: feature store situacional (guía sección 5).

Regla de oro: **ninguna** feature usa información posterior al tip-off. Las
medias móviles llevan ``shift(1)``, los ratings son pre-juego y los spots se
calculan con el calendario y los resultados ya conocidos (guía 5.7).
"""

from nba_sh.features.build import FEATURE_COLUMNS, FeatureStore, build_features

__all__ = ["FEATURE_COLUMNS", "FeatureStore", "build_features"]
