"""Tests del paper-trading: libro append-only, CLV y potencia estadística."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_sh.ingest.espn_source import SLATE_POSSESSIONS, events_to_slate
from nba_sh.paper.ledger import (
    CAPTURE_COLUMNS,
    SOURCE_API,
    SOURCE_FILE,
    PaperLedger,
    utc_stamp,
)
from nba_sh.paper.report import (
    MIN_BETS_FOR_VERDICT,
    bets_needed_for,
    clv_report,
    detectable_clv_with,
)
from nba_sh.schemas import GAME_COLUMNS, TIMING_OPEN

HOME, AWAY = "BOS", "NYK"
GAME = "g1"
OPEN_SPREAD = -3.5
CLOSE_SPREAD = -5.0
TOTAL = 228.5
STANDARD_PRICE = -110.0
SEASON = "2026-27"
Z_SUM = 1.959964 + 0.841621  # z(0.975) + z(0.80), la fórmula de potencia


def _odds(*, game_id: str = GAME, spread: float = OPEN_SPREAD) -> pd.DataFrame:
    """Una línea de mercado con el esquema canónico."""
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "book": "paperbook",
                "timing": TIMING_OPEN,
                "ml_home": -150.0,
                "ml_away": 130.0,
                "spread_home": spread,
                "spread_home_price": STANDARD_PRICE,
                "spread_away_price": STANDARD_PRICE,
                "total_line": TOTAL,
                "over_price": STANDARD_PRICE,
                "under_price": STANDARD_PRICE,
            }
        ]
    )


def _scheduled_event(
    *, event_id: str = "401910036", completed: bool = False
) -> dict[str, object]:
    """Evento de ESPN sin jugar, con el formato del scoreboard."""
    return {
        "id": event_id,
        "date": "2026-11-24T00:00Z",
        "season": {"year": 2027, "type": 2},
        "competitions": [
            {
                "date": "2026-11-24T00:00Z",
                "status": {
                    "type": {
                        "name": "STATUS_FINAL" if completed else "STATUS_SCHEDULED",
                        "completed": completed,
                    }
                },
                "competitors": [
                    {"homeAway": "home", "score": "0", "team": {"abbreviation": HOME}},
                    {"homeAway": "away", "score": "0", "team": {"abbreviation": AWAY}},
                ],
            }
        ],
    }


# --- jornada por jugar ------------------------------------------------


@pytest.mark.unit
def test_an_unplayed_game_becomes_a_slate_row() -> None:
    """Un partido programado entra sin marcador y con respaldos sensatos."""
    slate = events_to_slate([_scheduled_event()], season=SEASON)
    assert list(slate.columns) == list(GAME_COLUMNS)
    row = slate.iloc[0]
    assert row["home_team"] == HOME
    assert np.isnan(row["home_score"])
    assert np.isnan(row["away_score"])
    assert row["possessions"] == pytest.approx(SLATE_POSSESSIONS)
    assert row["game_date"] == pd.Timestamp("2026-11-23")


@pytest.mark.unit
def test_a_finished_game_is_not_a_slate_row() -> None:
    """Lo ya jugado no se apuesta: es el filtro inverso de events_to_games."""
    assert events_to_slate([_scheduled_event(completed=True)], season=SEASON).empty


# --- libro ------------------------------------------------------------


@pytest.mark.unit
def test_every_capture_is_its_own_file(tmp_path: Path) -> None:
    """El libro es append-only: dos capturas no se pisan."""
    ledger = PaperLedger(root=tmp_path)
    first = ledger.write_odds(
        _odds(),
        timing=TIMING_OPEN,
        captured_at=pd.Timestamp("2026-11-23T16:00", tz=UTC),
    )
    second = ledger.write_odds(
        _odds(spread=-4.0),
        timing=TIMING_OPEN,
        captured_at=pd.Timestamp("2026-11-23T20:00", tz=UTC),
    )
    assert first != second
    assert first.exists() and second.exists()
    captures = ledger.odds(timing=TIMING_OPEN)
    assert len(captures) == 2
    assert list(captures.columns) == list(CAPTURE_COLUMNS)


@pytest.mark.unit
def test_the_latest_capture_wins_per_game(tmp_path: Path) -> None:
    """Para apostar vale la última línea vista, no la primera."""
    ledger = PaperLedger(root=tmp_path)
    ledger.write_odds(
        _odds(spread=-3.0),
        timing=TIMING_OPEN,
        captured_at=pd.Timestamp("2026-11-23T16:00", tz=UTC),
    )
    ledger.write_odds(
        _odds(spread=-6.0),
        timing=TIMING_OPEN,
        captured_at=pd.Timestamp("2026-11-23T22:00", tz=UTC),
    )
    latest = ledger.latest_odds(timing=TIMING_OPEN)
    assert len(latest) == 1
    assert latest.loc[0, "spread_home"] == pytest.approx(-6.0)


@pytest.mark.unit
def test_the_capture_records_where_it_came_from(tmp_path: Path) -> None:
    """La fuente queda guardada: una captura de fichero no mide nada."""
    ledger = PaperLedger(root=tmp_path)
    ledger.write_odds(_odds(), timing=TIMING_OPEN, source=SOURCE_FILE)
    assert ledger.odds(timing=TIMING_OPEN).loc[0, "source"] == SOURCE_FILE


@pytest.mark.unit
def test_an_unknown_timing_is_refused(tmp_path: Path) -> None:
    """Solo los tres momentos de la guía."""
    with pytest.raises(ValueError, match="timing desconocido"):
        PaperLedger(root=tmp_path).write_odds(_odds(), timing="whenever")


@pytest.mark.unit
def test_an_empty_ledger_returns_the_schema(tmp_path: Path) -> None:
    """Un libro sin nada devuelve esquemas vacíos, no revienta."""
    ledger = PaperLedger(root=tmp_path)
    assert ledger.odds().empty
    assert ledger.bets().empty
    assert ledger.settled().empty
    assert ledger.pending_bets().empty


@pytest.mark.unit
def test_pending_bets_exclude_what_is_already_settled(tmp_path: Path) -> None:
    """Lo liquidado sale de la cola de pendientes."""
    ledger = PaperLedger(root=tmp_path)
    bets = pd.DataFrame(
        [
            {"bet_id": "b1", "game_id": GAME, "market": "spread", "stake": 100.0},
            {"bet_id": "b2", "game_id": "g2", "market": "total", "stake": 100.0},
        ]
    )
    ledger.write_bets(bets, odds_snapshot="snap")
    assert len(ledger.pending_bets()) == 2
    ledger.write_settled(bets.head(1).assign(result="win", pnl=90.0, clv=0.01))
    assert ledger.pending_bets()["bet_id"].tolist() == ["b2"]


@pytest.mark.unit
def test_resettling_a_bet_keeps_the_newest(tmp_path: Path) -> None:
    """Repetir la liquidación no duplica la apuesta."""
    ledger = PaperLedger(root=tmp_path)
    row = pd.DataFrame([{"bet_id": "b1", "game_id": GAME, "pnl": 10.0}])
    ledger.write_settled(row, settled_at=pd.Timestamp("2026-11-24T12:00", tz=UTC))
    ledger.write_settled(
        row.assign(pnl=20.0), settled_at=pd.Timestamp("2026-11-24T18:00", tz=UTC)
    )
    settled = ledger.settled()
    assert len(settled) == 1
    assert settled.loc[0, "pnl"] == pytest.approx(20.0)


@pytest.mark.unit
def test_the_stamp_is_utc() -> None:
    """Los nombres de fichero van en UTC, sin ambigüedad de zona."""
    assert utc_stamp(pd.Timestamp("2026-11-23T05:00", tz=UTC)) == "20261123T050000Z"


# --- CLV y potencia ---------------------------------------------------


def _settled(clv_values: list[float], *, source: str = SOURCE_API) -> pd.DataFrame:
    """Apuestas liquidadas con la CLV dada."""
    size = len(clv_values)
    return pd.DataFrame(
        {
            "bet_id": [f"b{index}" for index in range(size)],
            "game_id": [f"g{index}" for index in range(size)],
            "market": ["spread"] * size,
            "stake": [100.0] * size,
            "pnl": [90.0 if value > 0 else -100.0 for value in clv_values],
            "clv": clv_values,
            "clv_points": [value * 100 for value in clv_values],
            "source": [source] * size,
        }
    )


@pytest.mark.unit
def test_an_empty_report_asks_for_a_closing_capture() -> None:
    """Sin CLV, el veredicto dice exactamente qué falta."""
    report = clv_report(pd.DataFrame())
    assert report.n_with_clv == 0
    assert "capture --timing close" in report.verdict


@pytest.mark.unit
def test_the_report_computes_mean_and_standard_error() -> None:
    """La media y su error estándar salen de la muestra, no de un supuesto."""
    values = [0.01, 0.02, -0.01, 0.03]
    report = clv_report(_settled(values))
    assert report.mean_clv == pytest.approx(float(np.mean(values)))
    assert report.sd_clv == pytest.approx(float(np.std(values, ddof=1)))
    assert report.standard_error == pytest.approx(report.sd_clv / np.sqrt(len(values)))
    assert report.beat_close_rate == pytest.approx(0.75)


@pytest.mark.unit
def test_a_small_sample_refuses_to_conclude() -> None:
    """Con pocas apuestas el veredicto pide muestra, no da resultado."""
    report = clv_report(_settled([0.05] * 10))
    assert report.n_with_clv < MIN_BETS_FOR_VERDICT
    assert "Muestra insuficiente" in report.verdict


@pytest.mark.unit
def test_a_clear_positive_edge_is_called_out() -> None:
    """Con muestra y señal, el veredicto lo dice."""
    rng = np.random.default_rng(3)
    report = clv_report(_settled(rng.normal(0.02, 0.01, size=400).tolist()))
    assert report.mean_clv > 0
    assert report.p_value < 0.05
    assert "primera señal real" in report.verdict


@pytest.mark.unit
def test_a_clear_negative_edge_says_do_not_continue() -> None:
    """Una CLV significativamente negativa no se maquilla."""
    rng = np.random.default_rng(4)
    report = clv_report(_settled(rng.normal(-0.02, 0.01, size=400).tolist()))
    assert report.mean_clv < 0
    assert "no sigas" in report.verdict


@pytest.mark.unit
def test_synthetic_odds_block_the_verdict() -> None:
    """Las cuotas de fichero prueban la tubería; no se concluye con ellas."""
    rng = np.random.default_rng(5)
    report = clv_report(
        _settled(rng.normal(0.02, 0.01, size=400).tolist(), source=SOURCE_FILE)
    )
    assert "no mide nada" in report.verdict


@pytest.mark.unit
def test_bets_needed_shrinks_as_the_target_grows() -> None:
    """Detectar un edge grande cuesta menos muestra que uno pequeño."""
    sd = 0.05
    assert bets_needed_for(0.01, sd) > bets_needed_for(0.02, sd)
    assert bets_needed_for(0.01, sd) == pytest.approx(
        np.ceil(Z_SUM**2 * sd**2 / 0.01**2), rel=0.01
    )
    assert bets_needed_for(0.0, sd) == 0


@pytest.mark.unit
def test_the_detectable_edge_shrinks_with_more_bets() -> None:
    """Más muestra, edge mínimo detectable más pequeño."""
    sd = 0.05
    assert detectable_clv_with(400, sd) < detectable_clv_with(100, sd)
    assert np.isnan(detectable_clv_with(1, sd))


@pytest.mark.unit
def test_bets_without_a_closing_line_keep_clv_at_nan() -> None:
    """Sin cierre guardado no hay CLV, y no se inventa."""
    settled = _settled([0.01, 0.02])
    settled.loc[1, "clv"] = np.nan
    report = clv_report(settled)
    assert report.n_bets == 2
    assert report.n_with_clv == 1


@pytest.mark.unit
def test_the_report_separates_markets() -> None:
    """La CLV por mercado se reporta aparte: no todos son igual de blandos."""
    settled = _settled([0.01, 0.02, 0.03, 0.04])
    settled.loc[2:, "market"] = "total"
    report = clv_report(settled)
    assert set(report.by_market) == {"spread", "total"}
    assert report.by_market["total"] == pytest.approx(0.035)


@pytest.mark.unit
def test_gaining_line_points_scores_positive() -> None:
    """Comprobación de sentido del signo que da sentido a todo el montaje.

    Coger el local a -3.5 y ver el cierre en -5 es ganar puntos de línea, y
    eso tiene que puntuar positivo. El cálculo vive en ``betting.settle``;
    aquí se fija lo que el reporte espera de él.
    """
    assert CLOSE_SPREAD < OPEN_SPREAD  # el local se encarece tras apostarlo
    gained = clv_report(_settled([0.015]))
    assert gained.mean_clv > 0
    assert gained.mean_clv_points > 0


# --- cadena completa de liquidación -----------------------------------


@pytest.mark.unit
def test_settle_computes_clv_through_the_real_chain(tmp_path: Path) -> None:
    """La cadena entera: apertura, apuesta, cierre, resultado y CLV.

    Es la prueba que da sentido al montaje. Se apuesta el local a -3.5, el
    mercado cierra en -5 (el local se encarece), y el partido se gana por 10.
    Ganar punto y medio de línea frente al cierre tiene que dar CLV positiva.
    """
    from nba_sh.paper.session import settle
    from nba_sh.schemas import TIMING_CLOSE

    ledger = PaperLedger(root=tmp_path)
    ledger.write_odds(
        _odds(spread=OPEN_SPREAD),
        timing=TIMING_OPEN,
        captured_at=pd.Timestamp("2026-11-23T16:00", tz=UTC),
    )
    ledger.write_odds(
        _odds(spread=CLOSE_SPREAD),
        timing=TIMING_CLOSE,
        captured_at=pd.Timestamp("2026-11-24T00:00", tz=UTC),
    )
    ledger.write_bets(
        pd.DataFrame(
            [
                {
                    "bet_id": "b1",
                    "game_id": GAME,
                    "game_date": pd.Timestamp("2026-11-23"),
                    "home_team": HOME,
                    "away_team": AWAY,
                    "market": "spread",
                    "side": "home",
                    "line": OPEN_SPREAD,
                    "american": STANDARD_PRICE,
                    "stake": 100.0,
                }
            ]
        ),
        odds_snapshot="snap",
    )

    results = pd.DataFrame(
        [{"game_id": GAME, "home_score": 115.0, "away_score": 105.0}]
    )
    outcome = settle(season=SEASON, ledger=ledger, results=results)
    assert outcome.n_settled == 1
    assert outcome.n_with_clv == 1
    assert outcome.n_still_pending == 0

    settled = ledger.settled().iloc[0]
    assert settled["result"] == "win"          # gana por 10 cubriendo -3.5
    assert settled["pnl"] == pytest.approx(100.0 * 100 / 110)
    assert settled["closing_line"] == pytest.approx(CLOSE_SPREAD)
    assert settled["clv_points"] > 0           # se ganaron puntos de línea
    assert settled["clv"] > 0
    assert ledger.pending_bets().empty


@pytest.mark.unit
def test_settle_leaves_unplayed_games_pending(tmp_path: Path) -> None:
    """Una apuesta de un partido sin jugar no se liquida ni se descarta."""
    from nba_sh.paper.session import settle

    ledger = PaperLedger(root=tmp_path)
    ledger.write_bets(
        pd.DataFrame([{"bet_id": "b1", "game_id": GAME, "market": "spread"}]),
        odds_snapshot="snap",
    )
    outcome = settle(
        season=SEASON,
        ledger=ledger,
        results=pd.DataFrame(columns=["game_id", "home_score", "away_score"]),
    )
    assert outcome.n_settled == 0
    assert outcome.n_still_pending == 1
    assert len(ledger.pending_bets()) == 1


# --- predicción de la jornada ----------------------------------------


def _slate_rows(
    games: pd.DataFrame, *, day: pd.Timestamp, pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    """Filas de jornada por jugar, con el esquema canónico."""
    template = games.iloc[-1].to_dict()
    return pd.DataFrame(
        [
            {
                **template,
                "game_id": f"slate{index}",
                "game_date": day,
                "home_team": home,
                "away_team": away,
                "home_score": np.nan,
                "away_score": np.nan,
                "crew_id": None,
            }
            for index, (home, away) in enumerate(pairs)
        ]
    ).reindex(columns=list(GAME_COLUMNS))


@pytest.mark.unit
def test_predicting_a_synthetic_slate_works(sample, config) -> None:
    """Camino completo offline: histórico sintético + jornada por jugar."""
    from nba_sh.paper.slate import predict_slate

    games = sample.dataset.games
    day = pd.Timestamp(games.game_date.max()) + pd.Timedelta(days=2)
    teams = sorted(set(games.home_team))[:4]
    slate = _slate_rows(
        games, day=day, pairs=[(teams[0], teams[1]), (teams[2], teams[3])]
    )
    predictions = predict_slate(sample.dataset, slate, config=config)
    assert len(predictions) == 2
    assert predictions["p_home_ml"].between(0.0, 1.0).all()
    assert predictions["model_total_line"].gt(150.0).all()
    assert predictions["game_id"].tolist() == ["slate0", "slate1"]


@pytest.mark.unit
def test_an_empty_slate_is_refused(sample, config) -> None:
    """Sin jornada no hay nada que predecir."""
    from nba_sh.paper.slate import predict_slate

    with pytest.raises(ValueError, match="vacía"):
        predict_slate(
            sample.dataset, pd.DataFrame(columns=list(GAME_COLUMNS)), config=config
        )


@pytest.mark.unit
def test_a_slate_with_two_dates_is_refused(sample, config) -> None:
    """Una llamada, una fecha: si no, las features se contaminan entre sí."""
    from nba_sh.paper.slate import predict_slate

    games = sample.dataset.games
    day = pd.Timestamp(games.game_date.max()) + pd.Timedelta(days=2)
    teams = sorted(set(games.home_team))[:4]
    mixed = _slate_rows(
        games, day=day, pairs=[(teams[0], teams[1]), (teams[2], teams[3])]
    )
    mixed.loc[1, "game_date"] = day + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="mezcla 2 fechas"):
        predict_slate(sample.dataset, mixed, config=config)


@pytest.mark.unit
def test_a_team_twice_in_the_slate_is_refused(sample, config) -> None:
    """Un equipo dos veces el mismo día heredaría sus propios NaN."""
    from nba_sh.paper.slate import predict_slate

    games = sample.dataset.games
    day = pd.Timestamp(games.game_date.max()) + pd.Timedelta(days=2)
    teams = sorted(set(games.home_team))[:3]
    repeated = _slate_rows(
        games, day=day, pairs=[(teams[0], teams[1]), (teams[0], teams[2])]
    )
    with pytest.raises(ValueError, match="equipos repetidos"):
        predict_slate(sample.dataset, repeated, config=config)


@pytest.mark.unit
def test_a_slate_before_any_history_is_refused(sample, config) -> None:
    """Sin pasado no se puede predecir el futuro."""
    from nba_sh.paper.slate import predict_slate

    games = sample.dataset.games
    day = pd.Timestamp(games.game_date.min()) - pd.Timedelta(days=5)
    teams = sorted(set(games.home_team))[:2]
    early = _slate_rows(games, day=day, pairs=[(teams[0], teams[1])])
    with pytest.raises(ValueError, match="no hay histórico"):
        predict_slate(sample.dataset, early, config=config)


# --- CLI --------------------------------------------------------------


@pytest.mark.unit
def test_the_paper_subcommands_parse() -> None:
    """Los cuatro subcomandos existen y exigen temporada donde hace falta."""
    from nba_sh.cli import build_parser

    parser = build_parser()
    for argv in (
        ["paper", "capture", "--season", SEASON, "--timing", "close"],
        ["paper", "bet", "--season", SEASON],
        ["paper", "settle", "--season", SEASON],
        ["paper", "report"],
    ):
        assert parser.parse_args(argv).handler is not None


@pytest.mark.unit
def test_the_simulations_flag_reaches_the_config() -> None:
    """Regresión: el flag se llamaba --simulations y ``_config_for`` lee ``sims``.

    Con el nombre equivocado argparse lo aceptaba y la configuración lo
    ignoraba en silencio, corriendo con el valor por defecto.
    """
    from nba_sh.cli import DEMO_SIMULATIONS, _config_for, build_parser

    args = build_parser().parse_args(
        ["paper", "bet", "--season", SEASON, "--sims", "123"]
    )
    assert _config_for(args, default_sims=DEMO_SIMULATIONS).sim.n_sims == 123


@pytest.mark.unit
def test_the_report_counts_pending_bets(tmp_path: Path) -> None:
    """Regresión: tras registrar 5 apuestas el reporte decía "0 apuestas"."""
    from nba_sh.paper.report import report_from_ledger

    ledger = PaperLedger(root=tmp_path)
    ledger.write_bets(
        pd.DataFrame(
            [{"bet_id": f"b{index}", "game_id": f"g{index}"} for index in range(5)]
        ),
        odds_snapshot="snap",
    )
    report = report_from_ledger(ledger=ledger)
    assert report.n_pending == 5
    assert report.n_with_clv == 0
    assert "5 apuestas registradas" in report.verdict


@pytest.mark.unit
def test_a_close_capture_skips_games_already_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tras el tipoff la línea ya no es cierre: ese partido no se captura.

    Es el riesgo real de programar la captura: si corre tarde, una línea en
    vivo etiquetada como cierre envenena la CLV sin que nada falle.
    """
    from nba_sh.paper import session
    from nba_sh.schemas import TIMING_CLOSE

    started, upcoming = "started", "upcoming"
    now = pd.Timestamp.utcnow().tz_localize(None)
    slate = pd.DataFrame(
        [
            {
                "game_id": started,
                "game_date": pd.Timestamp("2026-11-23"),
                "tipoff_utc": now - pd.Timedelta(hours=1),
                "home_team": HOME,
                "away_team": AWAY,
            },
            {
                "game_id": upcoming,
                "game_date": pd.Timestamp("2026-11-23"),
                "tipoff_utc": now + pd.Timedelta(hours=3),
                "home_team": "LAL",
                "away_team": "DEN",
            },
        ]
    )
    monkeypatch.setattr(session, "fetch_slate", lambda *a, **k: slate)
    captured = pd.concat(
        [_odds(game_id=started), _odds(game_id=upcoming)], ignore_index=True
    )
    monkeypatch.setattr(session, "payload_to_odds", lambda *a, **k: captured)
    monkeypatch.setattr(session, "fetch_odds_payload", lambda *a, **k: [])

    result = session.capture_odds(
        timing=TIMING_CLOSE,
        season=SEASON,
        day=pd.Timestamp("2026-11-23"),
        ledger=PaperLedger(root=tmp_path),
    )
    assert result.n_games == 1
    stored = PaperLedger(root=tmp_path).odds(timing=TIMING_CLOSE)
    assert stored["game_id"].tolist() == [upcoming]


# --- cierre reconstruido del pickcenter -------------------------------


def _pickcenter(
    *, spread: float, total: float, provider: str = "DraftKings"
) -> list[dict]:
    """Entrada de `pickcenter` con el formato que devuelve ESPN."""
    return [
        {
            "provider": {"id": "100", "name": provider, "priority": 1},
            "details": f"MEM {spread:+.1f}",
            "spread": spread,
            "overUnder": total,
            "overOdds": -107.0,
            "underOdds": -113.0,
            "homeTeamOdds": {"moneyLine": 125, "spreadOdds": -105.0, "underdog": True},
            "awayTeamOdds": {"moneyLine": -148, "spreadOdds": -116.0, "favorite": True},
        }
    ]


@pytest.mark.unit
def test_pickcenter_uses_the_home_spread_convention() -> None:
    """El `spread` de ESPN es el del local, igual que ``spread_home``.

    Verificado contra 10 partidos reales: MIN favorito da −10.5 y CHA no
    favorito da +2.5. Equivocar este signo invertiría toda la CLV, así que
    queda fijado aquí.
    """
    from nba_sh.ingest.espn_source import pickcenter_to_odds_row

    row = pickcenter_to_odds_row(
        GAME, _pickcenter(spread=2.5, total=213.5), provider="DraftKings"
    )
    assert row is not None
    assert row["spread_home"] == pytest.approx(2.5)
    assert row["total_line"] == pytest.approx(213.5)
    assert row["ml_home"] == pytest.approx(125.0)
    assert row["ml_away"] == pytest.approx(-148.0)
    assert row["spread_home_price"] == pytest.approx(-105.0)
    assert row["timing"] == "close"


@pytest.mark.unit
def test_pickcenter_without_the_book_returns_nothing() -> None:
    """Si el libro pedido no está, no se inventa una línea."""
    from nba_sh.ingest.espn_source import pickcenter_to_odds_row

    assert (
        pickcenter_to_odds_row(
            GAME, _pickcenter(spread=2.5, total=213.5, provider="OtroLibro"),
            provider="DraftKings",
        )
        is None
    )
    assert pickcenter_to_odds_row(GAME, [], provider="DraftKings") is None


@pytest.mark.unit
def test_reconstruct_close_marks_its_own_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El cierre reconstruido no se hace pasar por una captura en vivo."""
    from nba_sh.paper import session
    from nba_sh.paper.ledger import SOURCE_PICKCENTER

    ledger = PaperLedger(root=tmp_path)
    ledger.write_bets(
        pd.DataFrame([{"bet_id": "b1", "game_id": GAME}]), odds_snapshot="snap"
    )
    monkeypatch.setattr(
        session,
        "fetch_closing_lines",
        lambda ids, **kwargs: _odds(game_id=list(ids)[0], spread=CLOSE_SPREAD),
    )
    result = session.reconstruct_close(ledger=ledger)
    assert result.source == SOURCE_PICKCENTER
    stored = ledger.odds(timing="close")
    assert stored["source"].tolist() == [SOURCE_PICKCENTER]
    assert stored["spread_home"].tolist() == [CLOSE_SPREAD]


@pytest.mark.unit
def test_reconstruct_close_skips_games_that_already_have_one(tmp_path: Path) -> None:
    """No se vuelve a pedir lo que ya está guardado."""
    from nba_sh.paper import session
    from nba_sh.schemas import TIMING_CLOSE

    ledger = PaperLedger(root=tmp_path)
    ledger.write_bets(
        pd.DataFrame([{"bet_id": "b1", "game_id": GAME}]), odds_snapshot="snap"
    )
    ledger.write_odds(_odds(spread=CLOSE_SPREAD), timing=TIMING_CLOSE)
    with pytest.raises(ValueError, match="ya tienen cierre"):
        session.reconstruct_close(ledger=ledger)


@pytest.mark.unit
def test_pickcenter_odds_do_not_block_the_verdict() -> None:
    """Un cierre de ESPN es mercado real: avisa, pero no invalida la medición."""
    from nba_sh.paper.ledger import SOURCE_PICKCENTER

    rng = np.random.default_rng(7)
    report = clv_report(
        _settled(rng.normal(0.02, 0.01, size=40).tolist(), source=SOURCE_PICKCENTER)
    )
    assert "no mide nada" not in report.verdict
    assert "pickcenter" in report.verdict


@pytest.mark.unit
def test_resettling_picks_up_a_closing_line_that_arrived_late(tmp_path: Path) -> None:
    """El caso real: el cierre llegó después de liquidar, y hay que recuperarlo.

    Sin `resettle`, `settle` solo mira pendientes y la CLV se queda a NaN para
    siempre aunque el cierre aparezca luego.
    """
    from nba_sh.paper import session
    from nba_sh.schemas import TIMING_CLOSE

    ledger = PaperLedger(root=tmp_path)
    ledger.write_odds(
        _odds(spread=OPEN_SPREAD),
        timing=TIMING_OPEN,
        captured_at=pd.Timestamp("2026-11-23T16:00", tz=UTC),
    )
    ledger.write_bets(
        pd.DataFrame(
            [
                {
                    "bet_id": "b1",
                    "game_id": GAME,
                    "game_date": pd.Timestamp("2026-11-23"),
                    "home_team": HOME,
                    "away_team": AWAY,
                    "market": "spread",
                    "side": "home",
                    "line": OPEN_SPREAD,
                    "american": STANDARD_PRICE,
                    "stake": 100.0,
                }
            ]
        ),
        odds_snapshot="snap",
    )
    results = pd.DataFrame(
        [{"game_id": GAME, "home_score": 115.0, "away_score": 105.0}]
    )

    first = session.settle(season=SEASON, ledger=ledger, results=results)
    assert first.n_settled == 1
    assert first.n_with_clv == 0  # todavía sin cierre

    ledger.write_odds(
        _odds(spread=CLOSE_SPREAD),
        timing=TIMING_CLOSE,
        captured_at=pd.Timestamp("2026-11-24T00:00", tz=UTC),
    )
    # Sin resettle no hay nada que hacer: ya está liquidada.
    assert session.settle(season=SEASON, ledger=ledger, results=results).n_settled == 0
    again = session.settle(
        season=SEASON, ledger=ledger, results=results, resettle=True
    )
    assert again.n_settled == 1
    assert again.n_with_clv == 1
    settled = ledger.settled()
    assert len(settled) == 1  # no se duplica
    assert settled.loc[0, "clv_points"] > 0


@pytest.mark.unit
def test_a_close_capture_too_far_from_tipoff_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un cron tardío cae en la jornada siguiente y guardaría su apertura.

    Pasó en producción: el cron de las 23:30 se disparó a las 05:37 y capturó
    la jornada del día siguiente, a 18 h de su tipoff, etiquetada como cierre.
    La guarda del tipoff no lo veía porque esos partidos no habían empezado.
    """
    from nba_sh.paper import session
    from nba_sh.schemas import TIMING_CLOSE

    now = pd.Timestamp.utcnow().tz_localize(None)
    far = pd.DataFrame(
        [
            {
                "game_id": "lejano",
                "game_date": pd.Timestamp("2026-11-25"),
                "tipoff_utc": now + pd.Timedelta(hours=18),
                "home_team": HOME,
                "away_team": AWAY,
            }
        ]
    )
    monkeypatch.setattr(session, "fetch_slate", lambda *a, **k: far)
    monkeypatch.setattr(
        session, "payload_to_odds", lambda *a, **k: _odds(game_id="lejano")
    )
    monkeypatch.setattr(session, "fetch_odds_payload", lambda *a, **k: [])
    result = session.capture_odds(
        timing=TIMING_CLOSE, season=SEASON, ledger=PaperLedger(root=tmp_path)
    )
    assert result.n_games == 0
    assert PaperLedger(root=tmp_path).odds(timing=TIMING_CLOSE).empty


@pytest.mark.unit
def test_a_close_capture_near_tipoff_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A media hora del tipoff sí es un cierre: la guarda no estorba."""
    from nba_sh.paper import session
    from nba_sh.schemas import TIMING_CLOSE

    now = pd.Timestamp.utcnow().tz_localize(None)
    soon = pd.DataFrame(
        [
            {
                "game_id": "pronto",
                "game_date": pd.Timestamp("2026-11-25"),
                "tipoff_utc": now + pd.Timedelta(minutes=30),
                "home_team": HOME,
                "away_team": AWAY,
            }
        ]
    )
    monkeypatch.setattr(session, "fetch_slate", lambda *a, **k: soon)
    monkeypatch.setattr(
        session, "payload_to_odds", lambda *a, **k: _odds(game_id="pronto")
    )
    monkeypatch.setattr(session, "fetch_odds_payload", lambda *a, **k: [])
    result = session.capture_odds(
        timing=TIMING_CLOSE, season=SEASON, ledger=PaperLedger(root=tmp_path)
    )
    assert result.n_games == 1


@pytest.mark.unit
def test_betting_without_captures_names_the_available_timings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El error dice qué timings hay, que es lo que faltaba para diagnosticar.

    El workflow capturaba con `open` y apostaba con el defecto `t_minus_60`;
    el mensaje viejo no daba ninguna pista de la discrepancia.
    """
    from nba_sh.paper import session
    from nba_sh.schemas import TIMING_PREGAME

    ledger = PaperLedger(root=tmp_path)
    ledger.write_odds(_odds(), timing=TIMING_OPEN)
    slate = pd.DataFrame(
        [
            {
                "game_id": GAME,
                "game_date": pd.Timestamp("2026-11-25"),
                "tipoff_utc": pd.Timestamp("2026-11-26T00:00"),
                "home_team": HOME,
                "away_team": AWAY,
            }
        ]
    )
    monkeypatch.setattr(session, "fetch_slate", lambda *a, **k: slate)
    with pytest.raises(ValueError, match="Hay capturas de: open"):
        session.place_bets(
            season=SEASON,
            day=pd.Timestamp("2026-11-25"),
            timing=TIMING_PREGAME,
            ledger=ledger,
        )


@pytest.mark.unit
def test_an_already_played_slate_rolls_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un cron tardío cae en una jornada ya jugada: se avanza al día siguiente.

    Pasó dos veces en producción, con "no hay jornada el 2026-11-26" a las
    00:19 y a las 04:08 UTC del 27, cuando en hora del mercado seguía siendo
    el 26 y sus partidos estaban disputados.
    """
    from nba_sh.paper import session

    today, tomorrow = pd.Timestamp("2026-11-26"), pd.Timestamp("2026-11-27")
    slate = pd.DataFrame(
        [
            {
                "game_id": GAME,
                "game_date": tomorrow,
                "tipoff_utc": pd.Timestamp("2026-11-28T00:00"),
                "home_team": HOME,
                "away_team": AWAY,
            }
        ]
    )
    asked: list[pd.Timestamp] = []

    def fake_slate(day: pd.Timestamp, **kwargs: object) -> pd.DataFrame:
        asked.append(pd.Timestamp(day))
        return slate if pd.Timestamp(day) == tomorrow else slate.iloc[0:0]

    monkeypatch.setattr(session, "fetch_slate", fake_slate)
    monkeypatch.setattr(session, "market_today", lambda: today)

    chosen, resolved = session.resolve_slate(
        season=SEASON, day=None, allow_roll_forward=True
    )
    assert chosen == tomorrow
    assert len(resolved) == 1
    assert asked[:2] == [today, tomorrow]


@pytest.mark.unit
def test_an_explicit_day_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si pides un día concreto, no se avanza: si dices el 26, es el 26."""
    from nba_sh.paper import session

    asked: list[pd.Timestamp] = []

    def fake_slate(day: pd.Timestamp, **kwargs: object) -> pd.DataFrame:
        asked.append(pd.Timestamp(day))
        return pd.DataFrame(columns=["game_id", "game_date", "tipoff_utc"])

    monkeypatch.setattr(session, "fetch_slate", fake_slate)
    chosen, resolved = session.resolve_slate(
        season=SEASON, day=pd.Timestamp("2026-11-26"), allow_roll_forward=False
    )
    assert chosen == pd.Timestamp("2026-11-26")
    assert resolved.empty
    assert set(asked) == {pd.Timestamp("2026-11-26")}


@pytest.mark.unit
def test_the_close_never_rolls_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Avanzar el cierre sería capturar una línea a un día del tipoff."""
    from nba_sh.paper import session
    from nba_sh.schemas import TIMING_CLOSE

    today = pd.Timestamp("2026-11-26")
    asked: list[pd.Timestamp] = []

    def fake_slate(day: pd.Timestamp, **kwargs: object) -> pd.DataFrame:
        asked.append(pd.Timestamp(day))
        return pd.DataFrame(columns=["game_id", "game_date", "tipoff_utc"])

    monkeypatch.setattr(session, "fetch_slate", fake_slate)
    monkeypatch.setattr(session, "market_today", lambda: today)
    monkeypatch.setattr(session, "payload_to_odds", lambda *a, **k: _odds())
    monkeypatch.setattr(session, "fetch_odds_payload", lambda *a, **k: [])
    session.capture_odds(
        timing=TIMING_CLOSE, season=SEASON, ledger=PaperLedger(root=tmp_path)
    )
    assert set(asked) == {today}


@pytest.mark.unit
def test_a_slate_is_not_bet_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Una jornada, un registro.

    Pasó en producción: la del 27 de noviembre se apostó a mano a las 04:24 y
    otra vez por el cron a las 23:56, dejando tres `bet_id` repetidos a líneas
    distintas. Liquidarlo así cuenta dos veces las mismas apuestas.
    """
    from nba_sh.paper import session

    day = pd.Timestamp("2026-11-28")
    slate = pd.DataFrame(
        [
            {
                "game_id": GAME,
                "game_date": day,
                "tipoff_utc": pd.Timestamp("2026-11-29T00:00"),
                "home_team": HOME,
                "away_team": AWAY,
            }
        ]
    )
    monkeypatch.setattr(session, "fetch_slate", lambda *a, **k: slate)
    ledger = PaperLedger(root=tmp_path)
    ledger.write_odds(_odds(), timing=TIMING_OPEN)
    ledger.write_bets(
        pd.DataFrame([{"bet_id": "ya", "game_id": GAME, "game_date": day}]),
        odds_snapshot="snap",
    )
    with pytest.raises(ValueError, match="ya tiene 1 apuestas"):
        session.place_bets(
            season=SEASON, day=day, timing=TIMING_OPEN, ledger=ledger
        )


@pytest.mark.unit
def test_allow_rebet_is_the_explicit_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rehacer a mano es posible, pero hay que pedirlo."""
    from nba_sh.paper import session

    day = pd.Timestamp("2026-11-28")
    slate = pd.DataFrame(
        [
            {
                "game_id": GAME,
                "game_date": day,
                "tipoff_utc": pd.Timestamp("2026-11-29T00:00"),
                "home_team": HOME,
                "away_team": AWAY,
            }
        ]
    )
    monkeypatch.setattr(session, "fetch_slate", lambda *a, **k: slate)
    ledger = PaperLedger(root=tmp_path)
    ledger.write_odds(_odds(), timing=TIMING_OPEN)
    ledger.write_bets(
        pd.DataFrame([{"bet_id": "ya", "game_id": GAME, "game_date": day}]),
        odds_snapshot="snap",
    )
    # Pasa la guarda y falla más adelante, al no haber histórico que cargar:
    # lo que se comprueba aquí es que la guarda ya no es lo que lo detiene.
    with pytest.raises((FileNotFoundError, ValueError)) as caught:
        session.place_bets(
            season=SEASON,
            day=day,
            timing=TIMING_OPEN,
            ledger=ledger,
            allow_rebet=True,
        )
    assert "ya tiene" not in str(caught.value)
