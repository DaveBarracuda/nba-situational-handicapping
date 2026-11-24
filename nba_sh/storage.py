"""Capa 2: almacén raw inmutable + artefactos versionados.

Reglas de la guía (sección 2): *nunca* sobrescribir raw, y cada capa produce
outputs versionados y auditables. El almacén usa SQLite para consultas y
parquet para snapshots inmutables con timestamp.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba_sh.schemas import TABLE_NAMES, Dataset

LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")
RAW_SUBDIR = "raw"
ARTIFACT_SUBDIR = "artifacts"
SQLITE_FILENAME = "nba_sh.sqlite"
SNAPSHOT_TIME_FORMAT = "%Y%m%dT%H%M%SZ"


def _utc_stamp() -> str:
    """Timestamp UTC compacto para versionar snapshots."""
    return datetime.now(UTC).strftime(SNAPSHOT_TIME_FORMAT)


@dataclass(frozen=True)
class Storage:
    """Almacén local de data cruda y artefactos del modelo."""

    root: Path = DEFAULT_DATA_DIR

    @property
    def raw_dir(self) -> Path:
        """Directorio de snapshots inmutables."""
        return self.root / RAW_SUBDIR

    @property
    def artifact_dir(self) -> Path:
        """Directorio de artefactos versionados (modelos, reportes)."""
        return self.root / ARTIFACT_SUBDIR

    @property
    def sqlite_path(self) -> Path:
        """Ruta del fichero SQLite."""
        return self.root / SQLITE_FILENAME

    def ensure_dirs(self) -> None:
        """Crea la estructura de directorios si no existe."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Snapshots inmutables
    # ------------------------------------------------------------------
    def write_snapshot(self, dataset: Dataset, *, label: str = "ingest") -> Path:
        """Escribe un snapshot parquet inmutable y devuelve su directorio.

        Nunca sobrescribe: cada llamada crea un directorio nuevo con
        ``label`` y timestamp UTC.
        """
        self.ensure_dirs()
        snapshot_dir = self.raw_dir / f"{label}_{_utc_stamp()}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in dataset.tables().items():
            frame.to_parquet(snapshot_dir / f"{name}.parquet", index=False)
        LOGGER.info("snapshot escrito en %s (%d juegos)", snapshot_dir, dataset.n_games)
        return snapshot_dir

    def read_snapshot(self, snapshot_dir: Path) -> Dataset:
        """Lee un snapshot parquet previamente escrito."""
        frames = {}
        for name in TABLE_NAMES:
            path = Path(snapshot_dir) / f"{name}.parquet"
            # Los snapshots antiguos no incluyen todas las tablas.
            if path.exists():
                frames[name] = pd.read_parquet(path)
        return Dataset(**frames) if frames else Dataset.empty()

    def latest_snapshot(self, *, label: str = "ingest") -> Path | None:
        """Directorio del snapshot más reciente para un label, si existe."""
        if not self.raw_dir.exists():
            return None
        candidates = sorted(self.raw_dir.glob(f"{label}_*"))
        return candidates[-1] if candidates else None

    # ------------------------------------------------------------------
    # SQLite (consultas y pipeline diario)
    # ------------------------------------------------------------------
    def write_tables(self, dataset: Dataset, *, if_exists: str = "replace") -> None:
        """Materializa las tablas del dataset en SQLite.

        ``if_exists="append"`` conserva el histórico; ``"replace"`` reconstruye
        la vista derivada (el raw inmutable sigue en parquet).
        """
        self.ensure_dirs()
        with sqlite3.connect(self.sqlite_path) as connection:
            for name, frame in dataset.tables().items():
                serializable = frame.copy()
                for column in serializable.columns:
                    if pd.api.types.is_datetime64_any_dtype(serializable[column]):
                        serializable[column] = serializable[column].astype(str)
                serializable.to_sql(name, connection, if_exists=if_exists, index=False)
        LOGGER.info("tablas materializadas en %s", self.sqlite_path)

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        """Consulta parametrizada sobre el almacén SQLite."""
        with sqlite3.connect(self.sqlite_path) as connection:
            return pd.read_sql_query(sql, connection, params=params)

    # ------------------------------------------------------------------
    # Artefactos versionados
    # ------------------------------------------------------------------
    def write_artifact(
        self, payload: Mapping[str, Any], *, name: str, version: str
    ) -> Path:
        """Guarda un artefacto JSON versionado (métricas, reportes, config)."""
        self.ensure_dirs()
        path = self.artifact_dir / f"{name}_{version}_{_utc_stamp()}.json"
        path.write_text(
            json.dumps(dict(payload), indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info("artefacto escrito en %s", path)
        return path

    def write_table_artifact(
        self, frame: pd.DataFrame, *, name: str, version: str
    ) -> Path:
        """Guarda una tabla (apuestas, backtest) como CSV versionado."""
        self.ensure_dirs()
        path = self.artifact_dir / f"{name}_{version}_{_utc_stamp()}.csv"
        frame.to_csv(path, index=False)
        LOGGER.info("tabla escrita en %s (%d filas)", path, len(frame))
        return path


__all__ = [
    "ARTIFACT_SUBDIR",
    "DEFAULT_DATA_DIR",
    "RAW_SUBDIR",
    "SQLITE_FILENAME",
    "Storage",
]
