"""Configuración central del modelo.

Todas las constantes numéricas del sistema viven aquí (regla: cero magic
numbers repartidos por el código). Los objetos son dataclasses *frozen*:
para variar un parámetro se crea una copia con :func:`dataclasses.replace`,
nunca se mutan en sitio.

Los valores por defecto son los de la guía (secciones 5.5A, 6, 7, 8, 9, 10, 11).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

# --------------------------------------------------------------------------
# Constantes de liga (se re-estiman desde la data cuando hay histórico)
# --------------------------------------------------------------------------
LEAGUE_AVG_PTS_PRIOR = 113.0
LEAGUE_AVG_POSS_PRIOR = 99.5
LEAGUE_MARGIN_SIGMA_PRIOR = 12.0
MINUTES_PER_GAME = 48.0
POSSESSIONS_SCALE = 100.0


@dataclass(frozen=True)
class EloConfig:
    """Elo dinámico (guía 6.1)."""

    base_rating: float = 1500.0
    k_base: float = 20.0
    margin_divisor: float = 40.0
    margin_cap: float = 40.0
    hfa_elo: float = 55.0
    carryover: float = 0.75
    carryover_target: float = 1505.0
    # HFA dinámico: suma de ajustes en puntos Elo.
    hfa_altitude_bonus: float = 12.0
    hfa_away_b2b_bonus: float = 15.0
    hfa_away_long_trip_bonus: float = 8.0
    elo_scale: float = 400.0


@dataclass(frozen=True)
class PowerConfig:
    """Ratings de potencia ofensiva/defensiva (guía 6.2)."""

    weekly_decay: float = 0.94
    prior_off_rating: float = LEAGUE_AVG_PTS_PRIOR / LEAGUE_AVG_POSS_PRIOR * 100.0
    prior_def_rating: float = LEAGUE_AVG_PTS_PRIOR / LEAGUE_AVG_POSS_PRIOR * 100.0
    prior_pace: float = LEAGUE_AVG_POSS_PRIOR
    prior_weight_games: float = 6.0
    injury_weight_floor: float = 0.5
    season_carryover: float = 0.60


@dataclass(frozen=True)
class FormConfig:
    """Medias móviles de forma (guía 5.2).

    Las ventanas de 10 partidos cruzan el cambio de temporada: el primer
    partido de octubre hereda la forma de abril. ``season_carryover_weight``
    pondera esos partidos de la temporada anterior — 1.0 los cuenta igual
    (como siempre se hizo), 0.0 es reset duro.
    """

    season_carryover_weight: float = 1.0


@dataclass(frozen=True)
class RestConfig:
    """Fatiga y calendario (guía 5.1, 5.2). Puntos sobre el nivel base."""

    b2b_away_points: float = 3.0
    b2b_home_points: float = 1.75
    rest_diff_points_per_day: float = 0.55
    rest_diff_cap_days: float = 3.0
    road_trip_4plus_points: float = 0.8
    timezone_crossing_points: float = 0.18
    timezone_crossing_cap: int = 6
    travel_points_per_1000_miles_7d: float = 0.22
    travel_miles_cap_7d: float = 6000.0
    altitude_home_points: float = 0.9
    congestion_points_per_extra_game_7d: float = 0.35
    congestion_games_threshold: int = 3


@dataclass(frozen=True)
class SpotsConfig:
    """Módulo de spots y momentum (guía 5.5A)."""

    lookahead_points: float = -1.5
    letdown_points: float = -1.25
    sandwich_points: float = -1.15
    revenge_points: float = 0.75
    letdown_decay_second_game: float = 0.5

    # Regla (a): no sumar linealmente.
    secondary_spot_weight: float = 0.4
    # Regla (c): la fatiga ya castiga.
    fatigued_spot_multiplier: float = 0.5
    # Regla (b): momentum solo si es extremo.
    momentum_sigma_gate: float = 1.5
    momentum_points_per_sigma: float = 0.35
    momentum_points_cap: float = 1.0
    momentum_pt_diff_weight: float = 0.6
    momentum_net_rating_weight: float = 0.4
    # Regla (e): cap total.
    total_situational_cap: float = 4.0

    # Definiciones operativas.
    marquee_elo_rank_top: int = 6
    weak_opponent_elo_rank_bottom: int = 10
    lookahead_max_days_ahead: int = 2
    blowout_margin_for_emotional_win: float = 20.0
    revenge_margin: float = 25.0
    letdown_min_rest: int = 2
    close_game_margin: float = 5.0
    luck_binomial_pvalue: float = 0.85

    # Regla (d): validación por backtest.
    min_effect_points: float = 0.7
    min_sample_games: int = 200


@dataclass(frozen=True)
class SeasonStageConfig:
    """Ajustes por etapa de temporada (guía 5.5A.4)."""

    early_last_game: int = 20
    mid_last_game: int = 45
    golden_last_game: int = 70
    late_games_count: int = 12

    spot_multiplier: Mapping[str, float] = field(
        default_factory=lambda: {"early": 0.30, "mid": 0.70, "golden": 1.00, "late": 1.00}
    )
    momentum_multiplier: Mapping[str, float] = field(
        default_factory=lambda: {"early": 0.00, "mid": 0.70, "golden": 1.00, "late": 0.90}
    )
    injury_multiplier: Mapping[str, float] = field(
        default_factory=lambda: {"early": 0.70, "mid": 1.00, "golden": 1.00, "late": 1.00}
    )
    motivation_multiplier: Mapping[str, float] = field(
        default_factory=lambda: {"early": 0.20, "mid": 0.60, "golden": 1.00, "late": 1.40}
    )
    rest_multiplier: Mapping[str, float] = field(
        default_factory=lambda: {"early": 1.00, "mid": 1.00, "golden": 1.00, "late": 1.00}
    )


@dataclass(frozen=True)
class InjuryConfig:
    """Impacto de lesiones por minutos ponderados por on/off (guía 5.3)."""

    points_per_minute_net_rating: float = 2.2 / 100.0
    late_scratch_hours: float = 3.0
    late_scratch_extra_multiplier: float = 1.25
    questionable_probability: float = 0.5
    doubtful_probability: float = 0.25
    impact_cap_points: float = 6.0
    pace_impact_points_per_unit: float = 0.25
    # Cuánto del impacto de una baja se traslada al TOTAL del partido. El
    # on/off mide margen (diferencial por 100 posesiones), no puntos anotados:
    # si falta una estrella, su equipo anota menos pero el rival anota más, así
    # que el total apenas se mueve. Medido con data real: trasladarlo al total
    # 1:1 empeoraba el MAE de totales en +0.14 pts sin ganar nada a cambio.
    injury_total_share: float = 0.0


@dataclass(frozen=True)
class RefereeConfig:
    """Árbitros (guía 5.6) — solo mercado de totales."""

    # Medido sobre 3 temporadas reales con las asignaciones de ESPN: la
    # pendiente bruta del total contra los TL previos del cuadro es +0.99 por
    # tiro libre y la parcial (sobre el residuo del modelo) +1.33. Se usa 0.9,
    # encogido hacia el 0.42 de la guía, lo que deja ~4 pts entre cuadros
    # extremos: dentro del rango 3-5 que la propia guía afirma.
    points_per_fta_above_average: float = 0.90
    max_total_points: float = 5.0
    # Media de los perfiles de cuadro sobre las 4 temporadas reales (no la
    # media de TL por partido de la liga: el perfil es una media de partidos
    # anteriores y sale algo más alta). Centrar aquí deja el ajuste con media
    # 0; con 44.6 metía +1.35 pts de sesgo a todos los totales. Con 3
    # temporadas salía 46.1, con 4 sale 45.83: conviene remedirlo al añadir
    # data, porque el sesgo no lo denuncia ningún test.
    league_avg_crew_fta: float = 45.83


@dataclass(frozen=True)
class MotivationConfig:
    """Motivación y contexto de temporada (guía 5.5)."""

    must_win_points: float = 1.1
    tanking_points: float = -1.6
    clinched_no_stakes_points: float = -0.9
    rivalry_points: float = 0.3


@dataclass(frozen=True)
class SimConfig:
    """Simulación Monte Carlo (guía 7.3)."""

    n_sims: int = 20_000
    score_correlation: float = 0.20
    # La guía sugiere theta ~= 35, pero con mu ~= 113 eso implica sd ~= 22 pts
    # por equipo (la real es ~12). El prior se fija coherente con la NBA y el
    # valor final se estima por MLE, como pide la propia guía (7.2).
    negbin_theta_prior: float = 400.0
    negbin_theta_bounds: tuple[float, float] = (20.0, 5_000.0)
    random_seed: int = 20240217


@dataclass(frozen=True)
class ModelConfig:
    """Pesos de los modelos (guía 8.1, 9.2)."""

    # Mezcla de la probabilidad de ML. La guía (8.1) describe dos enfoques
    # complementarios —Elo dinámico y logit— y aquí se añade un tercero: la
    # probabilidad implícita en la distribución simulada del margen. Es la
    # estimación más informativa (usa el margen, no solo ganar/perder) y es la
    # que mantiene coherentes ML, spread y total del mismo partido.
    elo_prior_weight: float = 0.20
    logit_weight: float = 0.35
    simulation_weight: float = 0.45
    spread_gaussian_weight: float = 0.60
    spread_linear_weight: float = 0.40
    # Los totales llevan la misma mezcla que el spread (guía 9.2): vía
    # estructural (ritmo x eficiencia) + vía aprendida sobre las mismas
    # features agregadas. Medido con data real: el ridge aporta ~0.6 pts de
    # MAE, así que se lleva el peso mayor.
    # Pesos elegidos en validación (entrenar 2022-23, evaluar 2023-24), sin
    # mirar la temporada de test: 0.85 al ridge fue el mínimo del MAE.
    total_structural_weight: float = 0.15
    total_linear_weight: float = 0.85
    margin_sigma_prior: float = LEAGUE_MARGIN_SIGMA_PRIOR
    ridge_alpha: float = 3.0
    logit_c: float = 0.5
    # "auto" usa isotónica cuando la muestra de calibración es grande y Platt
    # cuando es pequeña (la isotónica con pocos cientos de filas degenera en
    # una escalera de 3-4 valores). También acepta "isotonic" o "platt".
    calibration_method: str = "auto"
    calibration_fraction: float = 0.25
    min_train_games: int = 400


@dataclass(frozen=True)
class MarketConfig:
    """Umbrales de apuesta (guía 7.4, 8.3, 10.2)."""

    ml_min_prob_edge: float = 0.03
    ml_min_value: float = 0.03
    ml_odds_min: int = -400
    ml_odds_max: int = 400
    spread_min_points: float = 1.5
    spread_min_prob_edge: float = 0.02
    total_min_points: float = 3.5
    total_min_prob_edge: float = 0.03
    require_line_move_confirmation: bool = False


@dataclass(frozen=True)
class BankrollConfig:
    """Kelly fraccionado, límites y protecciones (guía 11)."""

    starting_bankroll: float = 10_000.0
    kelly_fraction_main: float = 0.25
    kelly_fraction_derived: float = 0.15
    kelly_fraction_situational: float = 0.10
    max_stake_pct: float = 0.02
    max_daily_exposure_pct: float = 0.06
    max_team_exposure_pct: float = 0.03
    daily_stop_loss_pct: float = 0.04
    max_bets_per_day: int = 12
    confidence_multiplier_high: float = 1.0
    confidence_multiplier_medium: float = 0.6
    confidence_multiplier_low: float = 0.3
    clv_degradation_window_days: int = 30
    clv_degradation_multiplier: float = 0.5
    correlated_market_discount: float = 0.6
    # El edge medido incluye el error de estimación del modelo (guía 1), así
    # que el stake se calcula sobre el edge encogido hacia el mercado. No
    # afecta a la detección de valor, solo al tamaño: Kelly sobre un edge
    # inflado es la vía rápida a la ruina.
    edge_shrink_for_sizing: float = 0.5


@dataclass(frozen=True)
class ValidationConfig:
    """Reglas de validación de data (guía 4)."""

    min_game_total: float = 150.0
    # El récord NBA es 370 puntos (Pistons 186 - Nuggets 184, 1983, 3 OT) y el
    # máximo moderno 351 (Kings 176 - Clippers 175, 2023, 2 OT). El techo debe
    # dejar pasar los extremos reales y seguir atrapando data corrupta.
    max_game_total: float = 400.0
    valid_periods: tuple[int, ...] = (4, 5, 6, 7, 8)
    min_american_odds: int = -10_000
    max_american_odds: int = 10_000
    max_abs_spread: float = 30.0
    min_total_line: float = 170.0
    max_total_line: float = 290.0
    cross_source_points_tolerance: float = 2.0


@dataclass(frozen=True)
class MonitoringConfig:
    """Monitoreo en producción (guía 14.2)."""

    psi_alert_threshold: float = 0.10
    psi_bins: int = 10
    logloss_window_days: int = 15
    logloss_alert_delta: float = 0.02
    clv_alert_threshold: float = 0.0
    late_scratch_on_off_alert: float = 3.0


@dataclass(frozen=True)
class MetricsTargets:
    """Objetivos mínimos de la tabla de la sección 13."""

    max_log_loss: float = 0.66
    max_brier: float = 0.215
    max_mae_spread: float = 10.5
    max_mae_total: float = 14.0
    max_ece: float = 0.02
    min_clv_pct: float = 1.5
    max_drawdown_pct: float = 30.0
    min_bets_for_conclusions: int = 300


@dataclass(frozen=True)
class Config:
    """Configuración raíz del sistema."""

    elo: EloConfig = field(default_factory=EloConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    form: FormConfig = field(default_factory=FormConfig)
    rest: RestConfig = field(default_factory=RestConfig)
    spots: SpotsConfig = field(default_factory=SpotsConfig)
    stage: SeasonStageConfig = field(default_factory=SeasonStageConfig)
    injury: InjuryConfig = field(default_factory=InjuryConfig)
    referee: RefereeConfig = field(default_factory=RefereeConfig)
    motivation: MotivationConfig = field(default_factory=MotivationConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    bankroll: BankrollConfig = field(default_factory=BankrollConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    targets: MetricsTargets = field(default_factory=MetricsTargets)
    model_version: str = "v1.0.0"

    def with_seed(self, seed: int) -> Config:
        """Devuelve una copia con otra semilla de simulación."""
        return replace(self, sim=replace(self.sim, random_seed=int(seed)))

    def with_sims(self, n_sims: int) -> Config:
        """Devuelve una copia con otro número de simulaciones."""
        return replace(self, sim=replace(self.sim, n_sims=int(n_sims)))


CONFIG = Config()

__all__ = [
    "CONFIG",
    "Config",
    "EloConfig",
    "FormConfig",
    "PowerConfig",
    "RestConfig",
    "SpotsConfig",
    "SeasonStageConfig",
    "InjuryConfig",
    "RefereeConfig",
    "MotivationConfig",
    "SimConfig",
    "ModelConfig",
    "MarketConfig",
    "BankrollConfig",
    "ValidationConfig",
    "MonitoringConfig",
    "MetricsTargets",
    "LEAGUE_AVG_PTS_PRIOR",
    "LEAGUE_AVG_POSS_PRIOR",
    "LEAGUE_MARGIN_SIGMA_PRIOR",
    "MINUTES_PER_GAME",
    "POSSESSIONS_SCALE",
]
