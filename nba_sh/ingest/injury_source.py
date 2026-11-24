"""Ingesta de lesiones (guía 3.3).

El NBA Injury Report oficial se publica a las 17:30 ET del día previo en PDF y
cambia de formato con frecuencia, así que aquí no se scrapea: el sistema lee un
fichero CSV/JSON que tú mantienes (manual o con tu propio scraper), lo valida y
lo normaliza. Eso mantiene el pipeline reproducible y auditable.

Esquema esperado (una fila por jugador y partido):

    game_id, team, player_id, minutes_pg, on_off_net, status, hours_before_tip

``status`` es ``out``, ``doubtful`` o ``questionable``; ``on_off_net`` es el
diferencial on/off del jugador por 100 posesiones (p.ej. de PBPStats).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from nba_sh.schemas import INJURY_COLUMNS, empty_frame
from nba_sh.validation import normalize_injuries

LOGGER = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".csv", ".json", ".parquet")


def load_injuries(path: str | Path) -> pd.DataFrame:
    """Carga y normaliza un fichero de lesiones.

    Args:
        path: ruta a un ``.csv``, ``.json`` o ``.parquet`` con el esquema
            documentado en este módulo.

    Returns:
        DataFrame normalizado con el esquema ``INJURY_COLUMNS``.

    Raises:
        FileNotFoundError: si el fichero no existe.
        ValueError: si faltan columnas obligatorias o el formato no se soporta.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"no existe el fichero de lesiones: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(file_path)
    elif suffix == ".json":
        frame = pd.read_json(file_path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(file_path)
    else:
        raise ValueError(
            f"formato no soportado: {suffix} (usa {', '.join(SUPPORTED_SUFFIXES)})"
        )

    missing = [column for column in INJURY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"faltan columnas en el fichero de lesiones: {missing}")
    normalized = normalize_injuries(frame[list(INJURY_COLUMNS)])
    LOGGER.info("lesiones cargadas: %d filas desde %s", len(normalized), file_path)
    return normalized


def empty_injuries() -> pd.DataFrame:
    """Tabla de lesiones vacía con el esquema correcto."""
    return empty_frame(INJURY_COLUMNS)


__all__ = ["SUPPORTED_SUFFIXES", "empty_injuries", "load_injuries"]
