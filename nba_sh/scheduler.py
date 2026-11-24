"""Automatización diaria (guía 14.1).

Cron diario (hora ET):

    06:00 -> actualizar resultados y box scores de anoche
    06:15 -> recalcular ratings y features
    07:00 -> pull del injury report oficial + beat writers
    07:30 -> correr modelos y generar líneas propias
    08:00 -> comparar con mercado y generar apuestas candidatas
    T-90m -> re-verificar scratches, confirmar o aplanar stakes
    T-30m -> pull de líneas finales y ejecutar apuestas aprobadas
    post  -> registrar resultados, actualizar CLV y alertas

El plan es declarativo: :func:`build_scheduler` lo monta con APScheduler si
está instalado, pero el mismo plan puede ejecutarse con cron del sistema.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

MARKET_TIMEZONE = "America/New_York"

JOB_RESULTS = "update_results"
JOB_FEATURES = "rebuild_features"
JOB_INJURIES = "pull_injuries"
JOB_MODELS = "run_models"
JOB_CANDIDATES = "generate_candidates"
JOB_RECHECK = "recheck_scratches"
JOB_EXECUTE = "final_lines_and_execute"
JOB_SETTLE = "settle_and_monitor"


@dataclass(frozen=True)
class Job:
    """Tarea programada del pipeline diario."""

    name: str
    schedule_et: str
    description: str
    # Comando de la CLI que ejecuta la tarea, cuando existe uno. El plan deja
    # de ser documentación y pasa a ser un runbook copiable.
    command: str = ""

    @property
    def is_relative_to_tipoff(self) -> bool:
        """True si la hora es relativa al tip-off (T-90, T-30)."""
        return self.schedule_et.startswith("T-")

    @property
    def cron_fields(self) -> tuple[int, int] | None:
        """``(hora, minuto)`` si la tarea tiene hora fija, si no ``None``."""
        if self.is_relative_to_tipoff or ":" not in self.schedule_et:
            return None
        hour, minute = self.schedule_et.split(":")
        return int(hour), int(minute)


DAILY_PLAN: tuple[Job, ...] = (
    Job(JOB_RESULTS, "06:00", "Resultados y box scores de anoche"),
    Job(JOB_FEATURES, "06:15", "Recalcular ratings, spots y feature store"),
    Job(
        JOB_INJURIES,
        "07:00",
        "Injury report oficial + scratches tempranos",
        "nba_sh paper capture --season <temporada> --timing open",
    ),
    Job(JOB_MODELS, "07:30", "Correr los tres modelos y generar líneas propias"),
    Job(
        JOB_CANDIDATES,
        "08:00",
        "Comparar con mercado y listar candidatas",
        "nba_sh paper bet --season <temporada>",
    ),
    Job(JOB_RECHECK, "T-90m", "Re-verificar scratches y confirmar stakes"),
    Job(
        JOB_EXECUTE,
        "T-30m",
        "Líneas finales y ejecución de aprobadas",
        "nba_sh paper capture --season <temporada> --timing close",
    ),
    Job(
        JOB_SETTLE,
        "post",
        "Registrar resultados, CLV y alertas de drift",
        "nba_sh paper settle --season <temporada> && nba_sh paper report",
    ),
)


def describe_plan() -> str:
    """Plan diario en texto, listo para log o documentación."""
    lines = [f"Plan diario (hora {MARKET_TIMEZONE}):"]
    for job in DAILY_PLAN:
        lines.append(f"  {job.schedule_et:>6}  {job.name:<26} {job.description}")
        if job.command:
            lines.append(f"{'':>10}  $ {job.command}")
    return "\n".join(lines)


def build_scheduler(
    handlers: Mapping[str, Callable[[], None]], *, timezone: str = MARKET_TIMEZONE
):
    """Monta un ``BackgroundScheduler`` de APScheduler con el plan diario.

    Solo registra las tareas con hora fija; las relativas al tip-off (T-90,
    T-30) dependen del calendario del día y las programa el propio job de
    modelos cuando conoce las horas de los partidos.

    Args:
        handlers: mapa ``nombre_de_job -> callable`` sin argumentos.
        timezone: zona horaria del scheduler (ET por defecto).

    Returns:
        El scheduler ya configurado (hay que llamar a ``start()``).

    Raises:
        ImportError: si APScheduler no está instalado.
        KeyError: si falta un handler para una tarea con hora fija.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError as error:  # pragma: no cover - depende del entorno
        raise ImportError(
            "APScheduler no está instalado: pip install apscheduler"
        ) from error

    scheduler = BackgroundScheduler(timezone=timezone)
    for job in DAILY_PLAN:
        fields = job.cron_fields
        if fields is None:
            continue
        if job.name not in handlers:
            raise KeyError(f"falta el handler del job {job.name!r}")
        hour, minute = fields
        scheduler.add_job(
            handlers[job.name],
            CronTrigger(hour=hour, minute=minute, timezone=timezone),
            id=job.name,
            name=job.description,
            replace_existing=True,
        )
        LOGGER.info("job %s programado a las %02d:%02d ET", job.name, hour, minute)
    return scheduler


__all__ = [
    "DAILY_PLAN",
    "JOB_CANDIDATES",
    "JOB_EXECUTE",
    "JOB_FEATURES",
    "JOB_INJURIES",
    "JOB_MODELS",
    "JOB_RECHECK",
    "JOB_RESULTS",
    "JOB_SETTLE",
    "MARKET_TIMEZONE",
    "Job",
    "build_scheduler",
    "describe_plan",
]
