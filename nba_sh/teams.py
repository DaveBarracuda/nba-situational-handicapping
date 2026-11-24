"""Catálogo de equipos: normalización de IDs, arenas, altitud y zonas horarias.

Guía 4 (problemas clásicos: equipos que se mudaron / renombraron) y 5.2
(viajes, zonas horarias, altitud).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

TIMEZONE_INDEX: Final[Mapping[str, int]] = {"ET": 0, "CT": 1, "MT": 2, "PT": 3}
ALTITUDE_GAME_FEET: Final[float] = 3_000.0


@dataclass(frozen=True)
class Arena:
    """Datos geográficos de la sede de un equipo."""

    team: str
    name: str
    city: str
    latitude: float
    longitude: float
    altitude_feet: float
    timezone: str

    @property
    def timezone_index(self) -> int:
        """Índice de zona horaria (ET=0 ... PT=3) para contar cruces."""
        return TIMEZONE_INDEX[self.timezone]

    @property
    def is_altitude_venue(self) -> bool:
        """True si la arena está en altitud relevante (Denver, Utah)."""
        return self.altitude_feet >= ALTITUDE_GAME_FEET


@dataclass(frozen=True)
class Team:
    """Identidad de franquicia."""

    code: str
    full_name: str
    conference: str
    division: str


_TEAM_ROWS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("ATL", "Atlanta Hawks", "East", "Southeast"),
    ("BOS", "Boston Celtics", "East", "Atlantic"),
    ("BKN", "Brooklyn Nets", "East", "Atlantic"),
    ("CHA", "Charlotte Hornets", "East", "Southeast"),
    ("CHI", "Chicago Bulls", "East", "Central"),
    ("CLE", "Cleveland Cavaliers", "East", "Central"),
    ("DAL", "Dallas Mavericks", "West", "Southwest"),
    ("DEN", "Denver Nuggets", "West", "Northwest"),
    ("DET", "Detroit Pistons", "East", "Central"),
    ("GSW", "Golden State Warriors", "West", "Pacific"),
    ("HOU", "Houston Rockets", "West", "Southwest"),
    ("IND", "Indiana Pacers", "East", "Central"),
    ("LAC", "Los Angeles Clippers", "West", "Pacific"),
    ("LAL", "Los Angeles Lakers", "West", "Pacific"),
    ("MEM", "Memphis Grizzlies", "West", "Southwest"),
    ("MIA", "Miami Heat", "East", "Southeast"),
    ("MIL", "Milwaukee Bucks", "East", "Central"),
    ("MIN", "Minnesota Timberwolves", "West", "Northwest"),
    ("NOP", "New Orleans Pelicans", "West", "Southwest"),
    ("NYK", "New York Knicks", "East", "Atlantic"),
    ("OKC", "Oklahoma City Thunder", "West", "Northwest"),
    ("ORL", "Orlando Magic", "East", "Southeast"),
    ("PHI", "Philadelphia 76ers", "East", "Atlantic"),
    ("PHX", "Phoenix Suns", "West", "Pacific"),
    ("POR", "Portland Trail Blazers", "West", "Northwest"),
    ("SAC", "Sacramento Kings", "West", "Pacific"),
    ("SAS", "San Antonio Spurs", "West", "Southwest"),
    ("TOR", "Toronto Raptors", "East", "Atlantic"),
    ("UTA", "Utah Jazz", "West", "Northwest"),
    ("WAS", "Washington Wizards", "East", "Southeast"),
)

TEAMS: Final[Mapping[str, Team]] = {
    code: Team(code=code, full_name=name, conference=conf, division=div)
    for code, name, conf, div in _TEAM_ROWS
}

VALID_TEAMS: Final[frozenset[str]] = frozenset(TEAMS)

_ARENA_ROWS: Final[tuple[tuple[str, str, str, float, float, float, str], ...]] = (
    ("ATL", "State Farm Arena", "Atlanta", 33.7573, -84.3963, 1050.0, "ET"),
    ("BOS", "TD Garden", "Boston", 42.3662, -71.0621, 20.0, "ET"),
    ("BKN", "Barclays Center", "Brooklyn", 40.6826, -73.9754, 30.0, "ET"),
    ("CHA", "Spectrum Center", "Charlotte", 35.2251, -80.8392, 751.0, "ET"),
    ("CHI", "United Center", "Chicago", 41.8807, -87.6742, 594.0, "CT"),
    ("CLE", "Rocket Arena", "Cleveland", 41.4965, -81.6882, 653.0, "ET"),
    ("DAL", "American Airlines Center", "Dallas", 32.7905, -96.8104, 430.0, "CT"),
    ("DEN", "Ball Arena", "Denver", 39.7487, -105.0077, 5280.0, "MT"),
    ("DET", "Little Caesars Arena", "Detroit", 42.3411, -83.0553, 600.0, "ET"),
    ("GSW", "Chase Center", "San Francisco", 37.7680, -122.3877, 10.0, "PT"),
    ("HOU", "Toyota Center", "Houston", 29.7508, -95.3621, 50.0, "CT"),
    ("IND", "Gainbridge Fieldhouse", "Indianapolis", 39.7640, -86.1555, 715.0, "ET"),
    ("LAC", "Intuit Dome", "Inglewood", 33.9452, -118.3418, 125.0, "PT"),
    ("LAL", "Crypto.com Arena", "Los Angeles", 34.0430, -118.2673, 233.0, "PT"),
    ("MEM", "FedExForum", "Memphis", 35.1382, -90.0506, 240.0, "CT"),
    ("MIA", "Kaseya Center", "Miami", 25.7814, -80.1870, 7.0, "ET"),
    ("MIL", "Fiserv Forum", "Milwaukee", 43.0451, -87.9172, 617.0, "CT"),
    ("MIN", "Target Center", "Minneapolis", 44.9795, -93.2760, 830.0, "CT"),
    ("NOP", "Smoothie King Center", "New Orleans", 29.9490, -90.0821, 7.0, "CT"),
    ("NYK", "Madison Square Garden", "New York", 40.7505, -73.9934, 33.0, "ET"),
    ("OKC", "Paycom Center", "Oklahoma City", 35.4634, -97.5151, 1200.0, "CT"),
    ("ORL", "Kia Center", "Orlando", 28.5392, -81.3839, 100.0, "ET"),
    ("PHI", "Wells Fargo Center", "Philadelphia", 39.9012, -75.1720, 39.0, "ET"),
    ("PHX", "Footprint Center", "Phoenix", 33.4457, -112.0712, 1086.0, "MT"),
    ("POR", "Moda Center", "Portland", 45.5316, -122.6668, 50.0, "PT"),
    ("SAC", "Golden 1 Center", "Sacramento", 38.5802, -121.4997, 25.0, "PT"),
    ("SAS", "Frost Bank Center", "San Antonio", 29.4270, -98.4375, 650.0, "CT"),
    ("TOR", "Scotiabank Arena", "Toronto", 43.6435, -79.3791, 251.0, "ET"),
    ("UTA", "Delta Center", "Salt Lake City", 40.7683, -111.9011, 4226.0, "MT"),
    ("WAS", "Capital One Arena", "Washington", 38.8981, -77.0209, 25.0, "ET"),
)

ARENAS: Final[Mapping[str, Arena]] = {
    code: Arena(
        team=code,
        name=name,
        city=city,
        latitude=lat,
        longitude=lon,
        altitude_feet=alt,
        timezone=tz,
    )
    for code, name, city, lat, lon, alt, tz in _ARENA_ROWS
}

# Franquicias mudadas / renombradas y alias habituales de otras fuentes.
TEAM_ALIASES: Final[Mapping[str, str]] = {
    "SEA": "OKC",
    "NJN": "BKN",
    "BRK": "BKN",
    "NOH": "NOP",
    "NOK": "NOP",
    "CHO": "CHA",
    "CHH": "CHA",
    "VAN": "MEM",
    "PHO": "PHX",
    "GOS": "GSW",
    "GS": "GSW",
    "SAN": "SAS",
    "SA": "SAS",
    "WSB": "WAS",
    "KCK": "SAC",
    "NY": "NYK",
    "NO": "NOP",
    "UTH": "UTA",
    # Códigos propios de ESPN.
    "UTAH": "UTA",
    "WSH": "WAS",
}

_FULL_NAME_TO_CODE: Final[Mapping[str, str]] = {
    team.full_name.lower(): code for code, team in TEAMS.items()
}

# Rivalidades clásicas (usadas por los spots de lookahead/sandwich, guía 5.5A).
RIVALRIES: Final[frozenset[frozenset[str]]] = frozenset(
    frozenset(pair)
    for pair in (
        ("BOS", "LAL"),
        ("BOS", "NYK"),
        ("BOS", "PHI"),
        ("LAL", "LAC"),
        ("LAL", "GSW"),
        ("GSW", "CLE"),
        ("MIA", "BOS"),
        ("NYK", "BKN"),
        ("CHI", "DET"),
        ("DAL", "HOU"),
        ("DAL", "SAS"),
        ("HOU", "SAS"),
        ("MEM", "GSW"),
        ("DEN", "MIN"),
        ("MIL", "CHI"),
        ("PHI", "TOR"),
        ("UTA", "LAC"),
        ("POR", "OKC"),
        ("PHX", "LAL"),
        ("IND", "NYK"),
    )
)


def normalize_team(code: str | None) -> str:
    """Normaliza un identificador de equipo a su código canónico de 3 letras.

    Acepta códigos actuales, alias históricos y nombres completos.

    Raises:
        ValueError: si el identificador no corresponde a ningún equipo.
    """
    if code is None:
        raise ValueError("team code is None")
    raw = str(code).strip()
    if not raw:
        raise ValueError("team code is empty")
    upper = raw.upper()
    if upper in VALID_TEAMS:
        return upper
    if upper in TEAM_ALIASES:
        return TEAM_ALIASES[upper]
    lowered = raw.lower()
    if lowered in _FULL_NAME_TO_CODE:
        return _FULL_NAME_TO_CODE[lowered]
    raise ValueError(f"unknown NBA team identifier: {code!r}")


def is_valid_team(code: str | None) -> bool:
    """True si el identificador puede normalizarse a un equipo válido."""
    try:
        normalize_team(code)
    except ValueError:
        return False
    return True


def arena(team: str) -> Arena:
    """Arena del equipo (normaliza el código primero)."""
    return ARENAS[normalize_team(team)]


def conference(team: str) -> str:
    """Conferencia del equipo."""
    return TEAMS[normalize_team(team)].conference


def is_rivalry(team_a: str, team_b: str) -> bool:
    """True si el par de equipos es una rivalidad catalogada."""
    return frozenset((normalize_team(team_a), normalize_team(team_b))) in RIVALRIES


def timezone_crossings(from_team: str, to_team: str) -> int:
    """Zonas horarias cruzadas al viajar de la sede A a la sede B."""
    return abs(arena(from_team).timezone_index - arena(to_team).timezone_index)


__all__ = [
    "ALTITUDE_GAME_FEET",
    "ARENAS",
    "Arena",
    "RIVALRIES",
    "TEAMS",
    "TEAM_ALIASES",
    "TIMEZONE_INDEX",
    "Team",
    "VALID_TEAMS",
    "arena",
    "conference",
    "is_rivalry",
    "is_valid_team",
    "normalize_team",
    "timezone_crossings",
]
