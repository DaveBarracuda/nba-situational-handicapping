"""Tests del camino arbitral de ESPN (sin red: respuestas sintéticas)."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_sh.ingest.espn_source import (
    MAX_OFFICIALS,
    MIN_OFFICIAL_GAMES,
    build_referee_tables,
    fetch_officials,
)
from nba_sh.schemas import REFEREE_COLUMNS

CREW = ("Scott Foster", "Tony Brothers", "Marc Davis")
FTA_PER_TEAM = 20.0
FOULS_PER_TEAM = 21.0
CREW_FTA = 2 * FTA_PER_TEAM
CREW_FOULS = 2 * FOULS_PER_TEAM
HISTORY_GAMES = MIN_OFFICIAL_GAMES + 2
FINAL_GAME_FTA_PER_TEAM = 60.0


class _Response:
    """Lo mínimo de ``requests.Response`` que usa el ingestor."""

    def __init__(self, payload: dict[str, object], *, fails: bool = False) -> None:
        self._payload = payload
        self._fails = fails

    def raise_for_status(self) -> None:
        if self._fails:
            import requests

            raise requests.exceptions.HTTPError("500")

    def json(self) -> dict[str, object]:
        return self._payload


def _summary(*names: str) -> dict[str, object]:
    """Respuesta de ``/summary`` con el formato que devuelve ESPN."""
    return {"gameInfo": {"officials": [{"fullName": name} for name in names]}}


def _patch_summary(
    monkeypatch: pytest.MonkeyPatch, response: _Response
) -> list[dict[str, object]]:
    """Intercepta ``requests.get`` y registra los parámetros enviados."""
    import requests

    calls: list[dict[str, object]] = []

    def fake_get(url: str, **kwargs: object) -> _Response:
        calls.append(dict(kwargs.get("params") or {}))
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def _games(count: int) -> pd.DataFrame:
    """Partidos consecutivos, uno por noche."""
    return pd.DataFrame(
        {
            "game_id": [f"g{index}" for index in range(count)],
            "game_date": pd.date_range("2024-01-01", periods=count, freq="D"),
        }
    )


def _officials(count: int, *, crew: tuple[str, ...] = CREW) -> pd.DataFrame:
    """El mismo cuadro en todos los partidos."""
    rows = [
        {
            "game_id": f"g{index}",
            "n_officials": len(crew),
            **{f"official_{position + 1}": name for position, name in enumerate(crew)},
        }
        for index in range(count)
    ]
    return pd.DataFrame(rows)


def _team_box(count: int, *, final_fta: float = FTA_PER_TEAM) -> pd.DataFrame:
    """Dos filas por partido; el último puede llevar otros tiros libres."""
    rows = [
        {
            "game_id": f"g{index}",
            "team": team,
            "free_throws_attempted": (
                final_fta if index == count - 1 else FTA_PER_TEAM
            ),
            "fouls": FOULS_PER_TEAM,
        }
        for index in range(count)
        for team in ("BOS", "NYK")
    ]
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_fetch_officials_reads_the_crew_from_the_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Los tres nombres llegan a official_1..3 con su recuento."""
    calls = _patch_summary(monkeypatch, _Response(_summary(*CREW)))
    officials = fetch_officials(["401585183"], pause_seconds=0.0)
    assert calls == [{"event": "401585183"}]
    row = officials.iloc[0]
    assert row["game_id"] == "401585183"
    assert row["n_officials"] == len(CREW)
    assert [row[f"official_{index + 1}"] for index in range(MAX_OFFICIALS)] == list(CREW)


@pytest.mark.unit
def test_fetch_officials_truncates_to_the_crew_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un cuarto nombre no cabe, y el recuento no lo cuenta."""
    _patch_summary(monkeypatch, _Response(_summary(*CREW, "Zach Zarba")))
    row = fetch_officials(["1"], pause_seconds=0.0).iloc[0]
    assert row["n_officials"] == MAX_OFFICIALS
    assert "Zach Zarba" not in row.to_dict().values()


@pytest.mark.unit
def test_fetch_officials_survives_a_failed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un partido sin árbitros deja la fila vacía, no corta la descarga."""
    _patch_summary(monkeypatch, _Response({}, fails=True))
    row = fetch_officials(["1"], pause_seconds=0.0).iloc[0]
    assert row["n_officials"] == 0
    assert row["official_1"] is None


@pytest.mark.unit
def test_crew_profile_uses_only_previous_games() -> None:
    """El perfil del cuadro son sus partidos anteriores, nunca el actual."""
    count = HISTORY_GAMES
    games, referees = build_referee_tables(
        _games(count),
        _officials(count),
        _team_box(count, final_fta=FINAL_GAME_FTA_PER_TEAM),
    )
    last_crew = games.loc[games.game_id == f"g{count - 1}", "crew_id"].iloc[0]
    profile = referees.loc[referees.crew_id == last_crew].iloc[0]
    assert profile["crew_fta_rate"] == pytest.approx(CREW_FTA)
    assert profile["crew_foul_rate"] == pytest.approx(CREW_FOULS)


@pytest.mark.unit
def test_crews_without_enough_history_get_no_profile() -> None:
    """Hasta los 10 partidos de cada colegiado no hay perfil que usar."""
    count = MIN_OFFICIAL_GAMES
    games, referees = build_referee_tables(
        _games(count), _officials(count), _team_box(count)
    )
    assert games.crew_id.notna().all()
    assert referees.empty
    assert list(referees.columns) == list(REFEREE_COLUMNS)


@pytest.mark.unit
def test_crew_id_ignores_the_order_of_the_names() -> None:
    """El mismo trío en otro orden es el mismo cuadro esa noche."""
    games, _ = build_referee_tables(
        _games(1), _officials(1, crew=tuple(reversed(CREW))), _team_box(1)
    )
    assert games.loc[0, "crew_id"] == f"{'|'.join(sorted(CREW))}@g0"


@pytest.mark.unit
def test_each_night_is_its_own_crew() -> None:
    """La composición cambia cada noche: un crew_id por partido."""
    count = HISTORY_GAMES
    games, referees = build_referee_tables(
        _games(count), _officials(count), _team_box(count)
    )
    assert games.crew_id.nunique() == count
    assert len(referees) == count - MIN_OFFICIAL_GAMES


@pytest.mark.unit
def test_missing_officials_leave_the_games_untouched() -> None:
    """Sin asignaciones no hay perfil, y los partidos pasan intactos."""
    games = _games(3)
    updated, referees = build_referee_tables(games, pd.DataFrame(), _team_box(3))
    assert referees.empty
    pd.testing.assert_frame_equal(updated, games)


@pytest.mark.unit
def test_the_adjustment_is_centred_and_symmetric() -> None:
    """Un cuadro medio no mueve el total, y el ajuste es simétrico."""
    from nba_sh.config import CONFIG
    from nba_sh.features.referees import referee_total_points

    centre = CONFIG.referee.league_avg_crew_fta
    points = referee_total_points(pd.Series([centre, centre + 2.0, centre - 2.0]))
    assert points.iloc[0] == pytest.approx(0.0)
    assert points.iloc[1] == pytest.approx(-points.iloc[2])
    assert points.iloc[1] == pytest.approx(
        2.0 * CONFIG.referee.points_per_fta_above_average
    )


@pytest.mark.unit
def test_the_adjustment_respects_its_cap() -> None:
    """Un cuadro extremo no puede mover el total más del tope configurado."""
    from nba_sh.config import CONFIG
    from nba_sh.features.referees import referee_total_points

    cap = CONFIG.referee.max_total_points
    extreme = CONFIG.referee.league_avg_crew_fta + 100.0
    points = referee_total_points(pd.Series([extreme, -extreme]))
    assert points.iloc[0] == pytest.approx(cap)
    assert points.iloc[1] == pytest.approx(-cap)
