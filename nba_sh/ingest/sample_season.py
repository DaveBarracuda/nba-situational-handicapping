"""Simulación de una temporada sintética (resultados, lesiones, árbitros).

La "verdad" generada aquí incluye efectos situacionales reales (B2B, viajes,
altitud, lookahead/letdown/sandwich/revenge, lesiones y árbitros) con las
mismas reglas de combinación que usa el modelo (:mod:`nba_sh.situational_rules`).
El mercado sintético (:mod:`nba_sh.ingest.sample_market`) los subvalora, que es
exactamente la inefficiencia que el modelo debe encontrar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.geo import venue_distance_miles
from nba_sh.schemas import STATUS_DOUBTFUL, STATUS_OUT, STATUS_QUESTIONABLE
from nba_sh.situational_rules import (
    apply_fatigue_multiplier,
    cap_total,
    combine_same_sign,
)
from nba_sh.teams import VALID_TEAMS, arena, is_rivalry, timezone_crossings

LEAGUE_OFF_MEAN = 114.5
LEAGUE_DEF_MEAN = 114.5
TEAM_OFF_SD = 3.6
TEAM_DEF_SD = 3.4
PACE_MEAN = 99.5
PACE_SD = 2.6
STRENGTH_PERSISTENCE = 0.65
HFA_POINTS = 2.6
TEAM_SCORE_NOISE_SD = 9.0
POSSESSION_NOISE_SD = 2.5
MIN_TEAM_SCORE = 78
OT_POINTS_MEAN = 12.0
OT_POINTS_SD = 3.0
MAX_OVERTIMES = 4
N_CREWS = 15
CREW_FTA_SD = 3.2
CREW_FOUL_MEAN = 40.0
CREW_FOUL_SD = 2.5
ROSTER_MINUTES = (36.0, 34.0, 31.0, 28.0, 25.0, 21.0, 17.0, 13.0, 9.0)
ROSTER_ON_OFF_BASE = (6.5, 5.0, 3.5, 2.5, 1.8, 1.2, 0.8, 0.4, 0.2)
ROSTER_ON_OFF_SD = 1.6
INJURY_GAME_PROBABILITY = 0.26
INJURY_EXTRA_LAMBDA = 0.35
MAX_INJURIES_PER_TEAM = 3
LATE_SCRATCH_PROBABILITY = 0.10
STATUS_OUT_THRESHOLD = 0.80
STATUS_DOUBTFUL_THRESHOLD = 0.90
SEASON_OPENER_REST_DAYS = 3
ROAD_TRIP_PRIOR_AWAY_GAMES = 3
LOOKBACK_DAYS = 7
LATE_SEASON_FIRST_GAME = 70
TANKING_LATENT_RANK = 26
PLAY_IN_LATENT_RANK_LOW = 10
PLAY_IN_LATENT_RANK_HIGH = 16


@dataclass(frozen=True)
class TeamSeasonStrength:
    """Fuerza latente de un equipo en una temporada."""

    off_rating: float
    def_rating: float
    pace: float

    @property
    def net_rating(self) -> float:
        """Net rating latente (positivo = mejor equipo)."""
        return self.off_rating - self.def_rating


@dataclass(frozen=True)
class Player:
    """Jugador de rotación con su impacto on/off."""

    player_id: str
    minutes_pg: float
    on_off_net: float


@dataclass(frozen=True)
class SimulatedSeason:
    """Salida de la simulación de una temporada."""

    games: pd.DataFrame
    injuries: pd.DataFrame
    truth: pd.DataFrame


@dataclass
class _History:
    """Historial mutable interno de un equipo durante la simulación."""

    dates: list[pd.Timestamp] = field(default_factory=list)
    venues: list[tuple[pd.Timestamp, str]] = field(default_factory=list)
    away_streak: int = 0
    last_margin: float = 0.0
    last_overtime: bool = False
    last_opponent_rank: int = 15
    h2h_margin: dict[str, float] = field(default_factory=dict)

    def rest_days(self, day: pd.Timestamp) -> int:
        """Días de descanso desde el último juego (B2B = 0).

        En el estreno de temporada se asume descanso completo.
        """
        if not self.dates:
            return SEASON_OPENER_REST_DAYS
        return max(int((day - self.dates[-1]).days) - 1, 0)

    def games_in_last(self, day: pd.Timestamp, days: int) -> int:
        """Juegos disputados en los últimos ``days`` días."""
        cutoff = day - pd.Timedelta(days=days)
        return sum(1 for date in self.dates if date > cutoff)

    def recent_venues(
        self, day: pd.Timestamp, days: int
    ) -> list[tuple[pd.Timestamp, str]]:
        """Sedes visitadas en la ventana reciente."""
        cutoff = day - pd.Timedelta(days=days)
        return [(date, venue) for date, venue in self.venues if date > cutoff]

    def travel_miles(self, day: pd.Timestamp, venue: str, days: int) -> float:
        """Millas voladas en la ventana reciente, incluido el viaje de hoy."""
        legs = [item[1] for item in self.recent_venues(day, days)] + [venue]
        return float(
            sum(
                venue_distance_miles(legs[index], legs[index + 1])
                for index in range(len(legs) - 1)
            )
        )

    def timezone_crossings(self, day: pd.Timestamp, venue: str, days: int) -> int:
        """Zonas horarias cruzadas en la ventana reciente."""
        legs = [item[1] for item in self.recent_venues(day, days)] + [venue]
        return int(
            sum(
                timezone_crossings(legs[index], legs[index + 1])
                for index in range(len(legs) - 1)
            )
        )

    def record(
        self,
        day: pd.Timestamp,
        *,
        venue: str,
        is_home: bool,
        margin: float,
        overtime: bool,
        opponent: str,
        opponent_rank: int,
    ) -> None:
        """Registra el resultado de un juego."""
        self.dates.append(day)
        self.venues.append((day, venue))
        self.away_streak = 0 if is_home else self.away_streak + 1
        self.last_margin = float(margin)
        self.last_overtime = bool(overtime)
        self.last_opponent_rank = int(opponent_rank)
        self.h2h_margin[opponent] = float(margin)


def draw_strengths(
    rng: np.random.Generator, previous: Mapping[str, TeamSeasonStrength] | None = None
) -> dict[str, TeamSeasonStrength]:
    """Sortea la fuerza latente de cada equipo para una temporada.

    Con ``previous`` aplica persistencia entre temporadas (los equipos no se
    reinventan cada año), que es lo que hace útil el carryover de Elo.
    """
    strengths: dict[str, TeamSeasonStrength] = {}
    for team in sorted(VALID_TEAMS):
        prior = previous.get(team) if previous else None
        off_prev = prior.off_rating - LEAGUE_OFF_MEAN if prior else 0.0
        def_prev = prior.def_rating - LEAGUE_DEF_MEAN if prior else 0.0
        pace_prev = prior.pace - PACE_MEAN if prior else 0.0
        persistence = STRENGTH_PERSISTENCE if prior else 0.0
        innovation = float(np.sqrt(1.0 - persistence**2))
        strengths[team] = TeamSeasonStrength(
            off_rating=LEAGUE_OFF_MEAN
            + persistence * off_prev
            + innovation * rng.normal(0.0, TEAM_OFF_SD),
            def_rating=LEAGUE_DEF_MEAN
            + persistence * def_prev
            + innovation * rng.normal(0.0, TEAM_DEF_SD),
            pace=PACE_MEAN
            + persistence * pace_prev
            + innovation * rng.normal(0.0, PACE_SD),
        )
    return strengths


def draw_rosters(rng: np.random.Generator) -> dict[str, tuple[Player, ...]]:
    """Plantillas sintéticas con minutos y on/off por jugador."""
    rosters: dict[str, tuple[Player, ...]] = {}
    for team in sorted(VALID_TEAMS):
        rosters[team] = tuple(
            Player(
                player_id=f"{team}-P{index + 1:02d}",
                minutes_pg=float(minutes),
                on_off_net=float(max(rng.normal(base, ROSTER_ON_OFF_SD), 0.1)),
            )
            for index, (minutes, base) in enumerate(
                zip(ROSTER_MINUTES, ROSTER_ON_OFF_BASE, strict=True)
            )
        )
    return rosters


def draw_referee_crews(
    rng: np.random.Generator, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Cuadros arbitrales con su perfil de faltas/tiros libres (guía 5.6)."""
    crew_ids = [f"CREW{index + 1:02d}" for index in range(N_CREWS)]
    return pd.DataFrame(
        {
            "crew_id": crew_ids,
            "crew_foul_rate": rng.normal(CREW_FOUL_MEAN, CREW_FOUL_SD, size=N_CREWS),
            "crew_fta_rate": rng.normal(
                config.referee.league_avg_crew_fta, CREW_FTA_SD, size=N_CREWS
            ),
        }
    )


def _strength_ranks(strengths: Mapping[str, TeamSeasonStrength]) -> dict[str, int]:
    """Ranking latente 1..30 (1 = mejor net rating)."""
    ordered = sorted(strengths.items(), key=lambda item: -item[1].net_rating)
    return {team: rank + 1 for rank, (team, _) in enumerate(ordered)}


def _team_game_positions(schedule: pd.DataFrame) -> dict[str, list[int]]:
    """Índices de fila de cada equipo, en orden cronológico."""
    positions: dict[str, list[int]] = {team: [] for team in sorted(VALID_TEAMS)}
    for index, row in enumerate(schedule.itertuples(index=False)):
        positions[row.home_team].append(index)
        positions[row.away_team].append(index)
    return positions


def _opponent_of(schedule: pd.DataFrame, row_index: int, team: str) -> tuple[str, bool]:
    """Oponente y si ``team`` juega en casa en esa fila del calendario."""
    row = schedule.iloc[row_index]
    if row["home_team"] == team:
        return str(row["away_team"]), True
    return str(row["home_team"]), False


def _is_marquee(
    opponent: str, team: str, ranks: Mapping[str, int], config: Config
) -> bool:
    """Oponente de alto perfil: top-6 latente o rivalidad."""
    return ranks[opponent] <= config.spots.marquee_elo_rank_top or is_rivalry(
        team, opponent
    )


def _is_weak(opponent: str, ranks: Mapping[str, int], config: Config) -> bool:
    """Oponente bottom-10 latente."""
    return (
        ranks[opponent] > len(VALID_TEAMS) - config.spots.weak_opponent_elo_rank_bottom
    )


def _spot_points(
    *,
    team: str,
    opponent: str,
    history: _History,
    schedule: pd.DataFrame,
    positions: Sequence[int],
    position_index: int,
    day: pd.Timestamp,
    rest_days: int,
    ranks: Mapping[str, int],
    config: Config,
) -> tuple[float, dict[str, bool]]:
    """Puntos de spots verdaderos para un equipo en un juego.

    Replica las definiciones operativas de la guía 5.5A.1 usando la fuerza
    latente (el modelo usará Elo, que es una estimación ruidosa de ésta).
    """
    spots = config.spots
    weak_today = _is_weak(opponent, ranks, config)
    flags = {"lookahead": False, "letdown": False, "sandwich": False, "revenge": False}
    values: list[float] = []

    next_index = (
        positions[position_index + 1] if position_index + 1 < len(positions) else None
    )
    prev_index = positions[position_index - 1] if position_index > 0 else None

    if next_index is not None and weak_today:
        next_opponent, _ = _opponent_of(schedule, next_index, team)
        next_day = pd.Timestamp(schedule.iloc[next_index]["game_date"])
        days_ahead = int((next_day - day).days)
        if days_ahead <= spots.lookahead_max_days_ahead and _is_marquee(
            next_opponent, team, ranks, config
        ):
            flags["lookahead"] = True
            values.append(spots.lookahead_points)

    if prev_index is not None and weak_today and rest_days >= spots.letdown_min_rest:
        emotional = history.last_overtime and history.last_margin > 0.0
        emotional = emotional or (
            history.last_margin >= spots.blowout_margin_for_emotional_win
            and history.last_opponent_rank <= spots.marquee_elo_rank_top
        )
        if emotional:
            flags["letdown"] = True
            values.append(spots.letdown_points)

    if prev_index is not None and next_index is not None and weak_today:
        prev_opponent, _ = _opponent_of(schedule, prev_index, team)
        next_opponent, _ = _opponent_of(schedule, next_index, team)
        if _is_marquee(prev_opponent, team, ranks, config) and _is_marquee(
            next_opponent, team, ranks, config
        ):
            flags["sandwich"] = True
            values.append(spots.sandwich_points)

    last_h2h = history.h2h_margin.get(opponent)
    if last_h2h is not None and last_h2h <= -spots.revenge_margin:
        flags["revenge"] = True
        values.append(spots.revenge_points)

    combined = combine_same_sign(values, secondary_weight=spots.secondary_spot_weight)
    combined = apply_fatigue_multiplier(
        combined, is_fatigued=rest_days <= 1, multiplier=spots.fatigued_spot_multiplier
    )
    return float(combined), flags


def _fatigue_points(
    *,
    history: _History,
    day: pd.Timestamp,
    venue: str,
    is_home: bool,
    rest_days: int,
    config: Config,
) -> float:
    """Puntos por fatiga de calendario y viajes (guía 5.1, 5.2)."""
    rest = config.rest
    points = 0.0
    if rest_days == 0:
        points -= rest.b2b_home_points if is_home else rest.b2b_away_points
    points += (
        rest.rest_diff_points_per_day
        * (min(rest_days, int(rest.rest_diff_cap_days)) - 1)
        / 2.0
    )
    if not is_home and history.away_streak >= ROAD_TRIP_PRIOR_AWAY_GAMES:
        points -= rest.road_trip_4plus_points
    crossings = min(
        history.timezone_crossings(day, venue, LOOKBACK_DAYS),
        rest.timezone_crossing_cap,
    )
    points -= rest.timezone_crossing_points * crossings
    miles = min(
        history.travel_miles(day, venue, LOOKBACK_DAYS), rest.travel_miles_cap_7d
    )
    points -= rest.travel_points_per_1000_miles_7d * miles / 1000.0
    if not is_home and arena(venue).is_altitude_venue:
        points -= rest.altitude_home_points
    congestion = max(
        history.games_in_last(day, LOOKBACK_DAYS) - rest.congestion_games_threshold, 0
    )
    points -= rest.congestion_points_per_extra_game_7d * congestion
    return float(points)


def _motivation_points(
    *, game_number: int, latent_rank: int, config: Config
) -> float:
    """Motivación verdadera en el tramo final (guía 5.5A.4, abril).

    Se aproxima con el ranking latente: los equipos del fondo tienen
    incentivo de perder y los de la zona de play-in juegan a vida o muerte.
    """
    if game_number <= LATE_SEASON_FIRST_GAME:
        return 0.0
    if latent_rank >= TANKING_LATENT_RANK:
        return config.motivation.tanking_points
    if PLAY_IN_LATENT_RANK_LOW <= latent_rank <= PLAY_IN_LATENT_RANK_HIGH:
        return config.motivation.must_win_points
    return 0.0


def _draw_injuries(
    rng: np.random.Generator,
    *,
    game_id: str,
    team: str,
    roster: Sequence[Player],
    config: Config,
) -> tuple[list[dict[str, object]], float]:
    """Sortea las bajas de un equipo y devuelve filas + impacto en puntos."""
    if rng.random() > INJURY_GAME_PROBABILITY:
        return [], 0.0
    count = int(
        min(1 + rng.poisson(INJURY_EXTRA_LAMBDA), MAX_INJURIES_PER_TEAM, len(roster))
    )
    chosen = np.atleast_1d(rng.choice(len(roster), size=count, replace=False))
    rows: list[dict[str, object]] = []
    impact = 0.0
    for index in chosen:
        player = roster[int(index)]
        status_draw = float(rng.random())
        if status_draw < STATUS_OUT_THRESHOLD:
            status, missing_weight = STATUS_OUT, 1.0
        elif status_draw < STATUS_DOUBTFUL_THRESHOLD:
            status = STATUS_DOUBTFUL
            missing_weight = 1.0 - config.injury.doubtful_probability
        else:
            status = STATUS_QUESTIONABLE
            missing_weight = 1.0 - config.injury.questionable_probability
        late = rng.random() < LATE_SCRATCH_PROBABILITY
        hours = float(rng.uniform(0.5, 2.9)) if late else float(rng.uniform(5.0, 30.0))
        impact += (
            player.minutes_pg
            * player.on_off_net
            * config.injury.points_per_minute_net_rating
            * missing_weight
        )
        rows.append(
            {
                "game_id": game_id,
                "team": team,
                "player_id": player.player_id,
                "minutes_pg": player.minutes_pg,
                "on_off_net": player.on_off_net,
                "status": status,
                "hours_before_tip": hours,
            }
        )
    return rows, float(min(impact, config.injury.impact_cap_points))


def _correlated_noise(
    rng: np.random.Generator, *, correlation: float
) -> tuple[float, float]:
    """Dos shocks normales correlacionados (ritmo común del partido)."""
    base = rng.normal(0.0, 1.0, size=2)
    second = correlation * base[0] + float(np.sqrt(1.0 - correlation**2)) * base[1]
    return float(base[0] * TEAM_SCORE_NOISE_SD), float(second * TEAM_SCORE_NOISE_SD)


def _resolve_overtime(
    rng: np.random.Generator, home_score: int, away_score: int
) -> tuple[int, int, int]:
    """Resuelve empates con periodos extra (el mercado los incluye)."""
    periods = 4
    while home_score == away_score and periods < 4 + MAX_OVERTIMES:
        periods += 1
        home_score += int(max(round(rng.normal(OT_POINTS_MEAN, OT_POINTS_SD)), 0))
        away_score += int(max(round(rng.normal(OT_POINTS_MEAN, OT_POINTS_SD)), 0))
    if home_score == away_score:
        home_score += 1
    return home_score, away_score, periods


def simulate_season(
    schedule: pd.DataFrame,
    strengths: Mapping[str, TeamSeasonStrength],
    crews: pd.DataFrame,
    rng: np.random.Generator,
    *,
    rosters: Mapping[str, Sequence[Player]] | None = None,
    config: Config = CONFIG,
) -> SimulatedSeason:
    """Simula los resultados de una temporada completa, juego a juego."""
    rosters = rosters or draw_rosters(rng)
    ranks = _strength_ranks(strengths)
    positions = _team_game_positions(schedule)
    histories = {team: _History() for team in sorted(VALID_TEAMS)}
    position_cursor = dict.fromkeys(sorted(VALID_TEAMS), 0)
    crew_lookup = crews.set_index("crew_id")

    game_rows: list[dict[str, object]] = []
    injury_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []

    for row_index in range(len(schedule)):
        row = schedule.iloc[row_index]
        game_id = str(row["game_id"])
        day = pd.Timestamp(row["game_date"])
        home, away = str(row["home_team"]), str(row["away_team"])
        venue = home

        crew_id = str(crews["crew_id"].iloc[int(rng.integers(0, len(crews)))])
        crew_fta = float(crew_lookup.loc[crew_id, "crew_fta_rate"])
        referee_total = float(
            np.clip(
                (crew_fta - config.referee.league_avg_crew_fta)
                * config.referee.points_per_fta_above_average,
                -config.referee.max_total_points,
                config.referee.max_total_points,
            )
        )

        side_adjustments: dict[str, float] = {}
        side_spots: dict[str, float] = {}
        side_injuries: dict[str, float] = {}
        side_flags: dict[str, dict[str, bool]] = {}
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            history = histories[team]
            rest_days = history.rest_days(day)
            fatigue = _fatigue_points(
                history=history,
                day=day,
                venue=venue,
                is_home=is_home,
                rest_days=rest_days,
                config=config,
            )
            spot_points, flags = _spot_points(
                team=team,
                opponent=opponent,
                history=history,
                schedule=schedule,
                positions=positions[team],
                position_index=position_cursor[team],
                day=day,
                rest_days=rest_days,
                ranks=ranks,
                config=config,
            )
            motivation = _motivation_points(
                game_number=position_cursor[team] + 1,
                latent_rank=ranks[team],
                config=config,
            )
            rows, injury_impact = _draw_injuries(
                rng, game_id=game_id, team=team, roster=rosters[team], config=config
            )
            injury_rows.extend(rows)
            side_spots[team] = spot_points
            side_injuries[team] = injury_impact
            side_flags[team] = flags
            side_adjustments[team] = cap_total(
                fatigue + spot_points + motivation,
                cap=config.spots.total_situational_cap,
            )

        # El on/off mide MARGEN, no puntos anotados: una baja hunde a su equipo
        # y beneficia al rival a partes iguales, así que el total no se mueve.
        injury_margin = side_injuries[away] - side_injuries[home]
        side_adjustments[home] += injury_margin / 2.0
        side_adjustments[away] -= injury_margin / 2.0

        possessions = float(
            np.clip(
                (strengths[home].pace + strengths[away].pace) / 2.0
                + rng.normal(0.0, POSSESSION_NOISE_SD),
                85.0,
                118.0,
            )
        )
        home_efficiency = strengths[home].off_rating + (
            strengths[away].def_rating - LEAGUE_DEF_MEAN
        )
        away_efficiency = strengths[away].off_rating + (
            strengths[home].def_rating - LEAGUE_DEF_MEAN
        )
        mu_home = (
            home_efficiency * possessions / 100.0
            + HFA_POINTS / 2.0
            + side_adjustments[home]
            + referee_total / 2.0
        )
        mu_away = (
            away_efficiency * possessions / 100.0
            - HFA_POINTS / 2.0
            + side_adjustments[away]
            + referee_total / 2.0
        )
        noise_home, noise_away = _correlated_noise(
            rng, correlation=config.sim.score_correlation
        )
        home_score = int(max(round(mu_home + noise_home), MIN_TEAM_SCORE))
        away_score = int(max(round(mu_away + noise_away), MIN_TEAM_SCORE))
        home_score, away_score, periods = _resolve_overtime(rng, home_score, away_score)

        game_rows.append(
            {
                "game_id": game_id,
                "season": row["season"],
                "season_type": row["season_type"],
                "game_date": day,
                "tipoff_utc": row["tipoff_utc"],
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "periods": periods,
                "possessions": possessions,
                "crew_id": crew_id,
            }
        )
        truth_rows.append(
            {
                "game_id": game_id,
                "mu_home": mu_home,
                "mu_away": mu_away,
                "true_margin": mu_home - mu_away,
                "true_total": mu_home + mu_away,
                "spot_margin": side_spots[home] - side_spots[away],
                "injury_margin": side_injuries[away] - side_injuries[home],
                "situational_margin": side_adjustments[home] - side_adjustments[away],
                "referee_total": referee_total,
                "possessions": possessions,
                "home_lookahead": side_flags[home]["lookahead"],
                "away_lookahead": side_flags[away]["lookahead"],
                "home_letdown": side_flags[home]["letdown"],
                "away_letdown": side_flags[away]["letdown"],
            }
        )

        margin = home_score - away_score
        overtime = periods > 4
        histories[home].record(
            day,
            venue=venue,
            is_home=True,
            margin=margin,
            overtime=overtime,
            opponent=away,
            opponent_rank=ranks[away],
        )
        histories[away].record(
            day,
            venue=venue,
            is_home=False,
            margin=-margin,
            overtime=overtime,
            opponent=home,
            opponent_rank=ranks[home],
        )
        position_cursor[home] += 1
        position_cursor[away] += 1

    return SimulatedSeason(
        games=pd.DataFrame(game_rows),
        injuries=pd.DataFrame(injury_rows),
        truth=pd.DataFrame(truth_rows),
    )


__all__ = [
    "HFA_POINTS",
    "LEAGUE_DEF_MEAN",
    "LEAGUE_OFF_MEAN",
    "PACE_MEAN",
    "Player",
    "SimulatedSeason",
    "TeamSeasonStrength",
    "draw_referee_crews",
    "draw_rosters",
    "draw_strengths",
    "simulate_season",
]
