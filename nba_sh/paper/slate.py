"""Predicción de una jornada **por jugar**.

`pipeline.run_daily` evalúa un día que ya está en el frame validado, con su
marcador: sirve para backtest, no para apostar esta noche. Aquí se hace lo
otro: coger el histórico limpio, pegarle las filas de la jornada sin marcador,
construir features y predecir.

Por qué no vale `prepare()` para esto: la validación de la guía (§4) exige
marcadores plausibles fila a fila, y hace bien — protege el histórico. Una
jornada sin jugar no tiene marcador, así que el histórico se valida como
siempre y el slate entra **después**, solo al feature store.

Eso es seguro porque las features son causales por construcción: la ventana de
cada fila mira ``shift(1)``, así que añadir filas futuras no puede cambiar las
features del pasado (hay un test que lo comprueba truncando el futuro). Lo que
sí exige es **una jornada de una sola fecha**: si un equipo apareciera dos
veces, su segunda fila heredaría los NaN de la primera.
"""

from __future__ import annotations

import logging

import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.features.build import build_features
from nba_sh.models import ModelBundle
from nba_sh.schemas import GAME_COLUMNS, Dataset

LOGGER = logging.getLogger(__name__)


def predict_slate(
    history: Dataset,
    slate: pd.DataFrame,
    *,
    spread_home: pd.Series | None = None,
    total_line: pd.Series | None = None,
    config: Config = CONFIG,
) -> pd.DataFrame:
    """Predice los tres mercados de una jornada por jugar.

    Args:
        history: dataset ya validado, sin la jornada.
        slate: juegos por jugar (``GAME_COLUMNS``, marcadores a NaN), todos
            de la misma fecha.
        spread_home: línea de spread del mercado, alineada con ``slate``.
        total_line: línea de total del mercado, alineada con ``slate``.
        config: configuración del modelo.

    Returns:
        Predicciones con una fila por juego del slate.

    Raises:
        ValueError: si el slate está vacío, mezcla fechas, repite un equipo o
            no hay histórico anterior.
    """
    if slate.empty:
        raise ValueError("la jornada está vacía: nada que predecir")
    dates = pd.to_datetime(slate["game_date"]).dt.normalize().unique()
    if len(dates) != 1:
        raise ValueError(f"la jornada mezcla {len(dates)} fechas; usa una por llamada")
    target = pd.Timestamp(dates[0])

    teams = pd.concat([slate["home_team"], slate["away_team"]])
    repeated = teams[teams.duplicated()].unique()
    if repeated.size:
        raise ValueError(f"equipos repetidos en la jornada: {sorted(repeated)}")

    past = history.games[pd.to_datetime(history.games["game_date"]) < target]
    if past.empty:
        raise ValueError(f"no hay histórico anterior a {target.date()}")

    # Los marcadores van a float en los dos lados antes de concatenar: en el
    # histórico son enteros y en el slate son NaN, y pandas avisa del dtype.
    aligned = slate.reindex(columns=list(GAME_COLUMNS)).astype(
        {"home_score": "float64", "away_score": "float64"}
    )
    extended = pd.concat(
        [past.astype({"home_score": "float64", "away_score": "float64"}), aligned],
        ignore_index=True,
    )
    features = build_features(history.with_games(extended), config=config)
    frame = features.games
    today = frame[pd.to_datetime(frame["game_date"]) == target]
    if today.empty:
        raise ValueError("la jornada no sobrevivió al feature store")

    trained = frame[pd.to_datetime(frame["game_date"]) < target]
    bundle = ModelBundle.fit(trained, config=config)
    LOGGER.info(
        "slate %s: %d juegos, modelo entrenado con %d",
        target.date(),
        len(today),
        len(trained),
    )
    return bundle.predict(today, spread_home=spread_home, total_line=total_line)


__all__ = ["predict_slate"]
