"""Interfaz de línea de comandos del modelo.

    python -m nba_sh demo                 # pipeline completo + slate de un día
    python -m nba_sh backtest --seasons 3 # walk-forward con reporte
    python -m nba_sh validate             # reporte de validación de data
    python -m nba_sh metrics              # métricas fuera de muestra
    python -m nba_sh monitor              # drift y alertas
    python -m nba_sh plan                 # plan diario de producción
    python -m nba_sh spots                # validación histórica de spots

Todas las salidas van por ``logging`` a stdout, de modo que el mismo comando
sirve en terminal y en un cron de producción.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from nba_sh import __version__
from nba_sh.backtest.report import validate_spots
from nba_sh.config import CONFIG, Config
from nba_sh.features.build import FEATURE_COLUMNS
from nba_sh.metrics import betting_metrics
from nba_sh.monitoring import monitoring_report
from nba_sh.paper.report import DEFAULT_CLV_TARGET
from nba_sh.pipeline import (
    DEFAULT_SEASONS,
    PreparedData,
    load_sample,
    model_quality,
    prepare,
    run_backtest,
    run_daily,
)
from nba_sh.scheduler import describe_plan
from nba_sh.schemas import TIMING_OPEN, TIMING_PREGAME, TIMINGS
from nba_sh.storage import Storage

LOGGER = logging.getLogger("nba_sh")
DEMO_SIMULATIONS = 20_000
BACKTEST_SIMULATIONS = 4_000
DEFAULT_RETRAIN_DAYS = 14
MAX_ALERTS_SHOWN = 20
MAX_ROWS_SHOWN = 10

SLATE_COLUMNS = (
    "game_id",
    "home_team",
    "away_team",
    "model_spread_line",
    "model_total_line",
    "p_home_ml",
)
BET_COLUMNS_SHOWN = (
    "game_id",
    "market",
    "side",
    "line",
    "american",
    "model_probability",
    "prob_edge",
    "points_edge",
    "confidence",
    "stake",
    "result",
    "pnl",
    "clv",
)
SUBGROUP_COLUMNS_SHOWN = ("group", "value", "n_bets", "stake", "pnl", "roi", "mean_clv")
SPOT_COLUMNS_SHOWN = ("spot", "effect_points", "n_games", "keep", "reason")


def configure_logging(verbose: bool = False) -> None:
    """Configura el logging a stdout."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(message)s"
    )
    LOGGER.setLevel(logging.INFO)


def _config_for(args: argparse.Namespace, *, default_sims: int) -> Config:
    """Configuración derivada de los flags comunes."""
    config = CONFIG.with_sims(getattr(args, "sims", None) or default_sims)
    if getattr(args, "seed", None) is not None:
        config = config.with_seed(args.seed)
    if getattr(args, "bankroll", None) is not None:
        config = replace(
            config, bankroll=replace(config.bankroll, starting_bankroll=args.bankroll)
        )
    return config


def _prepared(args: argparse.Namespace, config: Config) -> PreparedData:
    """Genera y prepara el dataset sintético según los flags."""
    sample = load_sample(args.seasons, seed=getattr(args, "seed", None), config=config)
    LOGGER.info(
        "dataset sintético: %d juegos, %d temporadas",
        sample.n_games,
        len(sample.dataset.seasons),
    )
    return prepare(sample.dataset, config=config)


def _log_table(frame: pd.DataFrame, columns: tuple[str, ...], *, title: str) -> None:
    """Imprime una tabla recortada a las columnas indicadas."""
    LOGGER.info("\n%s", title)
    if frame is None or frame.empty:
        LOGGER.info("  (vacío)")
        return
    available = [column for column in columns if column in frame.columns]
    LOGGER.info(
        "%s",
        frame[available].to_string(
            index=False, float_format=lambda value: f"{value:,.3f}"
        ),
    )


def command_demo(args: argparse.Namespace) -> int:
    """Pipeline completo sobre data sintética + slate del último día."""
    config = _config_for(args, default_sims=DEMO_SIMULATIONS)
    prepared = _prepared(args, config)
    for report in prepared.validation:
        LOGGER.info("validación -> %s %s", report.summary(), " ".join(report.messages))

    slate = run_daily(prepared, timing=args.timing, config=config)
    LOGGER.info(
        "\nJornada %s: %d juegos, %d apuestas aprobadas",
        slate.game_date.date(),
        len(slate.predictions),
        slate.n_bets,
    )
    LOGGER.info("modelo: %s", json.dumps(slate.bundle.summary(), default=str))
    _log_table(slate.predictions, SLATE_COLUMNS, title="Líneas propias del modelo")
    _log_table(slate.bets, BET_COLUMNS_SHOWN, title="Apuestas aprobadas")
    rejected = slate.evaluated[~slate.evaluated["is_candidate"].fillna(False)]
    LOGGER.info(
        "\nOportunidades descartadas: %d (%s)",
        len(rejected),
        dict(rejected["rejection_reason"].value_counts()),
    )
    if args.save:
        storage = Storage()
        storage.write_snapshot(prepared.dataset, label="sample")
        storage.write_table_artifact(
            slate.bets, name="bets", version=config.model_version
        )
        LOGGER.info("artefactos guardados en %s", storage.root.resolve())
    return 0


def command_backtest(args: argparse.Namespace) -> int:
    """Backtest walk-forward con reporte completo."""
    config = _config_for(args, default_sims=BACKTEST_SIMULATIONS)
    prepared = _prepared(args, config)
    result, report = run_backtest(
        prepared, timing=args.timing, retrain_days=args.retrain_days, config=config
    )
    LOGGER.info("\napuestas: %s", json.dumps(report["betting"], indent=2, default=str))
    LOGGER.info("\nmodelo: %s", json.dumps(report["market"], indent=2, default=str))
    LOGGER.info("sensibilidad: %s", json.dumps(report["sensitivity"], default=str))
    LOGGER.info(
        "\nbankroll %.2f -> %.2f en %d apuestas",
        result.starting_bankroll,
        result.final_bankroll,
        result.n_bets,
    )
    subgroups = pd.DataFrame(report["by_market"])
    if not subgroups.empty:
        _log_table(
            subgroups[subgroups["group"].isin(("market", "confidence", "stage_home"))],
            SUBGROUP_COLUMNS_SHOWN,
            title="ROI por subgrupo",
        )
    _log_table(
        pd.DataFrame(report["spot_validation"]),
        SPOT_COLUMNS_SHOWN,
        title="Validación de spots (regla d)",
    )
    if args.save:
        storage = Storage()
        storage.write_artifact(report, name="backtest", version=config.model_version)
        storage.write_table_artifact(
            result.bets, name="backtest_bets", version=config.model_version
        )
        LOGGER.info("artefactos guardados en %s", storage.root.resolve())
    return 0


def command_validate(args: argparse.Namespace) -> int:
    """Reporte de validación y limpieza de data."""
    config = _config_for(args, default_sims=BACKTEST_SIMULATIONS)
    prepared = _prepared(args, config)
    for report in prepared.validation:
        LOGGER.info("%s %s", report.summary(), " ".join(report.messages))
        if not report.failures.empty:
            _log_table(
                report.failures.head(MAX_ROWS_SHOWN),
                ("game_id", "failed_checks"),
                title=f"primeras filas rechazadas ({report.table})",
            )
    return 0


def command_metrics(args: argparse.Namespace) -> int:
    """Métricas de modelo fuera de muestra contra los objetivos de la guía."""
    config = _config_for(args, default_sims=BACKTEST_SIMULATIONS)
    prepared = _prepared(args, config)
    LOGGER.info(
        "\n%s", json.dumps(model_quality(prepared, config=config), indent=2, default=str)
    )
    return 0


def command_monitor(args: argparse.Namespace) -> int:
    """Reporte de drift y alertas de producción."""
    config = _config_for(args, default_sims=BACKTEST_SIMULATIONS)
    prepared = _prepared(args, config)
    games = prepared.games
    seasons = sorted(games["season"].unique())
    train = games[games["season"].isin(seasons[:-1])]
    live = games[games["season"] == seasons[-1]]
    result, _ = run_backtest(prepared, timing=args.timing, config=config)
    predictions = result.predictions.merge(
        games[["game_id", "home_win"]], on="game_id", how="left"
    )
    report = monitoring_report(
        train=train,
        live=live,
        feature_columns=FEATURE_COLUMNS,
        predictions=predictions,
        bets=result.bets,
        injuries=prepared.dataset.injuries,
        baseline_log_loss=config.targets.max_log_loss,
        config=config,
    )
    LOGGER.info("\nrolling log loss: %.4f", report["rolling_log_loss"])
    LOGGER.info("edge drift: %s", report["edge_drift"])
    _log_table(
        pd.DataFrame(report["drift"]).head(MAX_ROWS_SHOWN),
        ("feature", "psi", "alert"),
        title="PSI por feature (top 10)",
    )
    LOGGER.info("\nalertas (%d):", report["n_alerts"])
    for alert in report["alerts"][:MAX_ALERTS_SHOWN]:
        LOGGER.info("  %s", alert)
    LOGGER.info(
        "\napuestas: %s",
        json.dumps(
            betting_metrics(
                result.bets, starting_bankroll=result.starting_bankroll, config=config
            ),
            indent=2,
            default=str,
        ),
    )
    return 0


def command_spots(args: argparse.Namespace) -> int:
    """Valida los spots contra el histórico (guía 5.5A.3d)."""
    config = _config_for(args, default_sims=BACKTEST_SIMULATIONS)
    prepared = _prepared(args, config)
    _log_table(
        validate_spots(prepared.games, config=config),
        SPOT_COLUMNS_SHOWN,
        title="Validación de spots",
    )
    return 0


def command_plan(args: argparse.Namespace) -> int:
    """Muestra el plan diario de producción."""
    del args
    LOGGER.info("%s", describe_plan())
    return 0


def _add_common_arguments(subparser: argparse.ArgumentParser) -> None:
    """Flags compartidos por los comandos que generan data."""
    subparser.add_argument(
        "--seasons",
        type=int,
        default=DEFAULT_SEASONS,
        help="temporadas sintéticas a generar (por defecto 3)",
    )
    subparser.add_argument("--seed", type=int, default=None, help="semilla")
    subparser.add_argument(
        "--sims", type=int, default=None, help="simulaciones Monte Carlo por juego"
    )
    subparser.add_argument(
        "--timing",
        choices=TIMINGS,
        default=TIMING_PREGAME,
        help="momento de la línea usada para apostar",
    )
    subparser.add_argument(
        "--bankroll", type=float, default=None, help="bankroll inicial"
    )


def command_paper_capture(args: argparse.Namespace) -> int:
    """Guarda una captura de cuotas en el libro de paper-trading."""
    from nba_sh.paper.session import capture_odds

    result = capture_odds(
        timing=args.timing,
        season=args.season,
        day=pd.Timestamp(args.day) if args.day else None,
        payload_file=Path(args.payload_file) if args.payload_file else None,
        bookmaker=args.book,
    )
    LOGGER.info(
        "capturadas %d líneas de %d juegos (%s, fuente %s) -> %s",
        result.n_lines,
        result.n_games,
        result.timing,
        result.source,
        result.path,
    )
    return 0


def command_paper_bet(args: argparse.Namespace) -> int:
    """Corre el modelo y registra las apuestas de papel. No envía nada."""
    from nba_sh.paper.session import place_bets

    config = _config_for(args, default_sims=DEMO_SIMULATIONS)
    result = place_bets(
        season=args.season,
        day=pd.Timestamp(args.day) if args.day else None,
        timing=args.timing,
        config=config,
    )
    LOGGER.info(
        "jornada %s: %d apuestas de papel de %d oportunidades (bankroll %.2f)",
        result.game_date.date(),
        result.n_bets,
        result.n_evaluated,
        result.bankroll,
    )
    if result.rejections:
        LOGGER.info("descartes: %s", result.rejections)
    if result.path is not None:
        LOGGER.info("registradas en %s", result.path)
    return 0


def command_paper_settle(args: argparse.Namespace) -> int:
    """Liquida las apuestas de papel cuyos partidos ya terminaron."""
    from nba_sh.paper.session import settle

    result = settle(season=args.season, resettle=args.resettle)
    LOGGER.info(
        "liquidadas %d (%d con CLV), %d siguen pendientes",
        result.n_settled,
        result.n_with_clv,
        result.n_still_pending,
    )
    if result.path is not None:
        LOGGER.info("registradas en %s", result.path)
    return 0


def command_paper_reconstruct(args: argparse.Namespace) -> int:
    """Rescata el cierre de partidos ya jugados desde el pickcenter de ESPN."""
    from nba_sh.paper.session import reconstruct_close

    del args
    result = reconstruct_close()
    LOGGER.info(
        "cierre reconstruido: %d líneas de %d juegos (fuente %s) -> %s",
        result.n_lines,
        result.n_games,
        result.source,
        result.path,
    )
    LOGGER.info("corre `paper settle` otra vez para que calcule la CLV")
    return 0


def command_paper_record(args: argparse.Namespace) -> int:
    """Record del paper-trading: historial y acumulado."""
    from nba_sh.paper.ledger import PaperLedger
    from nba_sh.paper.record import bet_log, record_from_ledger

    record = record_from_ledger()
    log = bet_log(PaperLedger().settled())
    if not log.empty:
        _log_table(log, tuple(log.columns), title="Historial de apuestas de papel")
    LOGGER.info("=== Record acumulado ===")
    LOGGER.info(
        "%d liquidadas (%d-%d-%d) | %d pendientes",
        record.n_bets,
        record.wins,
        record.losses,
        record.pushes,
        record.pending,
    )
    if record.n_bets:
        LOGGER.info(
            "apostado %.2f | P&L %+.2f | ROI %+.1f%%",
            record.staked,
            record.pnl,
            100 * record.roi,
        )
        LOGGER.info(
            "CLV media %s (%d con CLV) | en puntos de línea %s",
            "n/d" if record.mean_clv != record.mean_clv else f"{record.mean_clv:+.4f}",
            record.n_with_clv,
            "n/d"
            if record.mean_clv_points != record.mean_clv_points
            else f"{record.mean_clv_points:+.2f}",
        )
        for market, stats in sorted(record.by_market.items()):
            LOGGER.info(
                "  %-7s n=%d | %d-%d | P&L %+.2f | ROI %+.1f%%",
                market,
                stats["n"],
                stats["wins"],
                stats["losses"],
                stats["pnl"],
                100 * stats["roi"],
            )
    LOGGER.info("veredicto: %s", record.verdict)
    if args.save:
        payload = {"record": record.as_dict(), "historial": log.to_dict("records")}
        path = Storage().write_artifact(
            payload, name="paper_record", version=CONFIG.model_version
        )
        LOGGER.info("artefacto en %s", path)
    return 0


def command_paper_report(args: argparse.Namespace) -> int:
    """Reporte de CLV con su potencia estadística."""
    from nba_sh.paper.report import report_from_ledger

    report = report_from_ledger(target=args.target)
    LOGGER.info("=== CLV del paper-trading ===")
    LOGGER.info(
        "apuestas: %d liquidadas (%d con CLV), %d pendientes",
        report.n_bets,
        report.n_with_clv,
        report.n_pending,
    )
    if report.n_with_clv:
        LOGGER.info(
            "CLV media %+.4f (%.2f pp) | sd %.4f | error estándar %.4f",
            report.mean_clv,
            100 * report.mean_clv,
            report.sd_clv,
            report.standard_error,
        )
        LOGGER.info(
            "t = %+.2f | p = %.4f | batió el cierre en el %.1f%% de las apuestas",
            report.t_statistic,
            report.p_value,
            100 * report.beat_close_rate,
        )
        LOGGER.info(
            "CLV en puntos de línea: %+.3f | ROI de papel: %s",
            report.mean_clv_points,
            "n/d" if report.roi != report.roi else f"{100 * report.roi:+.1f}%",
        )
        LOGGER.info(
            "para resolver %+.2f pp hacen falta ~%d apuestas; mínimo detectable "
            "ahora: %+.3f pp",
            100 * args.target,
            report.bets_needed,
            100 * report.detectable_clv,
        )
        if report.by_market:
            by_market = {
                market: round(value, 4) for market, value in report.by_market.items()
            }
            LOGGER.info("por mercado: %s", by_market)
    LOGGER.info("veredicto: %s", report.verdict)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de la CLI."""
    parser = argparse.ArgumentParser(
        prog="nba_sh",
        description="Modelo de situational handicapping NBA (ML, spread y totales)",
    )
    parser.add_argument("--version", action="version", version=f"nba_sh {__version__}")
    parser.add_argument("--verbose", action="store_true", help="logging detallado")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="pipeline completo + slate de un día")
    _add_common_arguments(demo)
    demo.add_argument("--save", action="store_true", help="guardar artefactos")
    demo.set_defaults(handler=command_demo)

    backtest = subparsers.add_parser("backtest", help="walk-forward + reporte")
    _add_common_arguments(backtest)
    backtest.add_argument(
        "--retrain-days",
        type=int,
        default=DEFAULT_RETRAIN_DAYS,
        help="cada cuántos días reentrenar",
    )
    backtest.add_argument("--save", action="store_true", help="guardar artefactos")
    backtest.set_defaults(handler=command_backtest)

    validate = subparsers.add_parser("validate", help="reporte de validación de data")
    _add_common_arguments(validate)
    validate.set_defaults(handler=command_validate)

    metrics = subparsers.add_parser("metrics", help="métricas fuera de muestra")
    _add_common_arguments(metrics)
    metrics.set_defaults(handler=command_metrics)

    monitor = subparsers.add_parser("monitor", help="drift y alertas")
    _add_common_arguments(monitor)
    monitor.set_defaults(handler=command_monitor)

    spots = subparsers.add_parser("spots", help="validación histórica de spots")
    _add_common_arguments(spots)
    spots.set_defaults(handler=command_spots)

    paper = subparsers.add_parser(
        "paper",
        help="paper-trading con CLV (no envía ninguna apuesta a ningún sitio)",
    )
    paper_subs = paper.add_subparsers(dest="paper_command", required=True)

    capture = paper_subs.add_parser("capture", help="guardar una captura de cuotas")
    capture.add_argument("--season", required=True, help='temporada, p.ej. "2026-27"')
    capture.add_argument(
        "--timing", choices=TIMINGS, default=TIMING_OPEN, help="momento de la captura"
    )
    capture.add_argument("--day", default=None, help="jornada (por defecto, hoy ET)")
    capture.add_argument("--book", default=None, help="clave del libro concreto")
    capture.add_argument(
        "--payload-file",
        default=None,
        help="JSON guardado en vez de la API (prueba de tubería, no medición)",
    )
    capture.set_defaults(handler=command_paper_capture)

    bet = paper_subs.add_parser("bet", help="registrar apuestas de papel de la jornada")
    bet.add_argument("--season", required=True, help='temporada, p.ej. "2026-27"')
    bet.add_argument("--day", default=None, help="jornada (por defecto, hoy ET)")
    bet.add_argument(
        "--timing",
        choices=TIMINGS,
        default=TIMING_PREGAME,
        help="timing de la captura que se usa como línea tomada",
    )
    # El flag se llama --sims porque es el nombre que lee ``_config_for``:
    # con cualquier otro se aceptaba y se ignoraba en silencio.
    bet.add_argument(
        "--sims", type=int, default=None, help="simulaciones Monte Carlo por juego"
    )
    bet.add_argument("--seed", type=int, default=None, help="semilla")
    bet.set_defaults(handler=command_paper_bet)

    settle_cmd = paper_subs.add_parser("settle", help="liquidar y calcular CLV")
    settle_cmd.add_argument("--season", required=True, help='temporada, p.ej. "2026-27"')
    settle_cmd.add_argument(
        "--resettle",
        action="store_true",
        help="reliquidar también lo ya liquidado (para una CLV que llegó tarde)",
    )
    settle_cmd.set_defaults(handler=command_paper_settle)

    reconstruct = paper_subs.add_parser(
        "reconstruct-close",
        help="rescatar el cierre de partidos ya jugados (pickcenter de ESPN)",
    )
    reconstruct.set_defaults(handler=command_paper_reconstruct)

    record_cmd = paper_subs.add_parser(
        "record", help="historial y record acumulado de las apuestas de papel"
    )
    record_cmd.add_argument(
        "--save", action="store_true", help="guardar el record como artefacto"
    )
    record_cmd.set_defaults(handler=command_paper_record)

    report_cmd = paper_subs.add_parser("report", help="CLV con potencia estadística")
    report_cmd.add_argument(
        "--target",
        type=float,
        default=DEFAULT_CLV_TARGET,
        help="CLV media que se considera un edge perseguible",
    )
    report_cmd.set_defaults(handler=command_paper_report)

    plan = subparsers.add_parser("plan", help="plan diario de producción")
    plan.set_defaults(handler=command_plan)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI."""
    args = build_parser().parse_args(argv)
    configure_logging(getattr(args, "verbose", False))
    return int(args.handler(args))


__all__ = [
    "BACKTEST_SIMULATIONS",
    "DEMO_SIMULATIONS",
    "build_parser",
    "configure_logging",
    "main",
]
