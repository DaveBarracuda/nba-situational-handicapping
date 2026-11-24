# NBA Situational Handicapping — modelo completo (ML, spread y totales)

Implementación ejecutable de una guía de situational handicapping NBA: las
siete capas, los tres mercados, el motor de apuestas con Kelly y límites, el
backtest walk-forward y el monitoreo de producción. Las referencias a secciones
(§5.5A.3, §14.1...) apuntan a ese documento, que **no se incluye en el repo**
por no constarme su autoría.

El paquete incluye una **liga sintética determinista** para verificar el
pipeline completo sin red ni credenciales, adaptadores para data real
(ESPN, hoopR, The Odds API) y un **paper-trading con CLV** que mide si el
modelo sirve sin arriesgar dinero.

**El resultado honesto, por delante de todo lo demás: este modelo no bate al
mercado.** Contra el spread de cierre de ESPN es 0,35-0,8 puntos peor en MAE en
las cuatro ventanas medidas, y su desacuerdo con la línea rinde 50,0% ATS
frente al 52,38% que exige el −110. El §4b lo detalla, con la salvedad de
potencia estadística que lo matiza. Lo que hay aquí es una medición rigurosa de
un modelo que no gana, no un sistema para ganar dinero.

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m nba_sh demo
```

```powershell
python -m nba_sh backtest --seasons 3 --save
```

---

## 1. Arquitectura (capas de la guía §2)

| Capa | Qué hace | Módulos |
|---|---|---|
| 1. Ingesta | nba_api, ESPN, box scores completos, play-by-play, odds API, lesiones, liga sintética | `ingest/nba_api_source.py`, `ingest/espn_source.py`, `ingest/hoopr_source.py`, `ingest/pbp_source.py`, `ingest/odds_source.py`, `ingest/injury_source.py`, `ingest/sample_*.py` |
| 2. Almacén raw | parquet inmutable + SQLite + artefactos versionados | `storage.py` |
| 3. Limpieza y validación | checks fila a fila, dedupe, normalización de franquicias, cruce de fuentes, vig | `validation.py`, `odds_utils.py` |
| 4. Feature store | rest, viajes, rolling con peso por temporada, spots, lesiones, árbitros, calidad de tiro, etapa, motivación | `features/` |
| 5. Modelos | Elo dinámico, ratings de potencia, NegBin + Monte Carlo, logit calibrado, ridge | `ratings/`, `models/` |
| 6. Motor de apuestas | detección de valor, Kelly fraccionado, límites, registro, liquidación | `betting/` |
| 6b. Paper-trading | captura de cuotas, jornada por jugar, libro append-only, record y CLV con potencia | `paper/` |
| 7. Backtest y monitoreo | walk-forward, ROI por subgrupo, CLV, PSI, alertas | `backtest/`, `metrics.py`, `monitoring.py` |

Orquestación y entrada: `pipeline.py`, `cli.py`, `scheduler.py`.
Toda la configuración numérica está en `config.py` (dataclasses inmutables).

## 2. Comandos

| Comando | Qué produce |
|---|---|
| `python -m nba_sh demo` | pipeline completo + líneas propias y apuestas de una jornada |
| `python -m nba_sh backtest --seasons 3` | walk-forward, ROI, CLV, drawdown, validación de spots |
| `python -m nba_sh validate` | reporte de validación de data con el motivo de cada rechazo |
| `python -m nba_sh metrics` | log loss, Brier, ECE y MAE fuera de muestra |
| `python -m nba_sh monitor` | PSI por feature, log loss rolling, alertas |
| `python -m nba_sh spots` | efecto histórico de cada spot y si se mantiene |
| `python -m nba_sh plan` | plan diario de producción (hora ET) con su comando |
| `python -m nba_sh paper ...` | **paper-trading con CLV**: `capture`, `bet`, `settle`, `reconstruct-close`, `record`, `report` (ver §4c) |

Flags comunes: `--seasons`, `--seed`, `--sims`,
`--timing {open,t_minus_60,close}`, `--bankroll`, `--save`.

## 3. Lo que implementa de la guía

| Sección de la guía | Implementación |
|---|---|
| §4 validación, vig | `validate_game_row`, `validate_odds_row`, Shin y multiplicativo |
| §5.1 rest / B2B | `features/rest.py` (B2B = 0 días de descanso) |
| §5.2 viajes | `features/travel.py` (haversine, husos, giras, altitud) |
| §5.3 lesiones | `features/injuries.py` (minutos × on/off, late scratch) |
| §5.4 ritmo y estilo | `features/rolling.py` y `features/shooting.py` (eFG, 3PA rate, FT rate, pintura), con `shift(1)` |
| §5.5 motivación | `features/season_stage.py` (standings, must-win, tanking, clinched) |
| §5.5A spots y momentum | `features/spots.py` + `situational_rules.py` (reglas a–e) |
| §5.6 árbitros | `features/referees.py` (solo totales) |
| §5.7 anti look-ahead | garantizado por diseño y verificado con un test |
| §6 Elo + potencia | `ratings/elo.py`, `ratings/power.py`, `ratings/base_points.py` |
| §7 totales | `models/totals.py` (NegBin + ridge agregado) + `models/simulate.py` (cópula) |
| §8 moneyline | `models/moneyline.py` (logit calibrado + Elo) y mezcla en `bundle.py` |
| §9 spread | `models/spread.py` (simulado + ridge, push incluido) |
| §10 valor | `betting/edge.py` (umbrales 3% / 1.5 pts / 3.5 pts) |
| §11 bankroll | `betting/kelly.py`, `betting/limits.py` |
| §12 backtest | `backtest/walkforward.py` (walk-forward puro, línea T-60) |
| §13 métricas | `metrics.py` con los objetivos de la tabla |
| §14 producción | `scheduler.py`, `monitoring.py` |
| §15 errores comunes | ver §6 de este README |
| §17 responsabilidad | ver §9 de este README |

## 4. Resultados verificados (liga sintética, 3 temporadas)

`python -m nba_sh backtest --seasons 3 --seed 2024 --sims 4000`
(3 688 juegos, entrenando solo con el pasado y reentrenando cada 14 días):

| Métrica | Objetivo de la guía | Obtenido |
|---|---|---|
| Log loss (ML) | < 0.66 | **0.587** |
| Brier (ML) | < 0.215 | **0.201** |
| ECE | < 0.02 | **0.009** |
| MAE spread | < 10.5 pts | **9.16** |
| MAE totales | < 14 pts | **12.98** |
| CLV media | > 0 consistente | **+5.1 pts de probabilidad** |
| ROI | > 0 con IC 95% positivo | **+6.3%**, IC [+1.1%, +11.5%] |
| Drawdown máximo | < 30% | **28.9%** |

Los **8 objetivos** de la guía se cumplen sobre la liga sintética.

1 783 apuestas de 19 686 oportunidades evaluadas: el sistema **no apuesta** el
88% de lo que mira, que es justo la disciplina que pide la guía. El ROI
sobrevive al recorte de sensibilidad del 30% (§12.3).

**Lee esto antes de extrapolar:** el libro sintético tiene un error de ~2 pts
en el margen (`ingest/sample_market.py`) para que el motor de valor sea
observable. Un libro real es más afilado, así que estos números verifican que
**el pipeline funciona**, no el edge que tendrás con data real. Ahí el ROI
honesto se mide con CLV sobre apuestas reales, no con un backtest.

## 4b. Resultados con data real (4 temporadas NBA + la que está en curso,
5.158 juegos)

### Data: calendario, box scores, disponibilidad, árbitros y tiros

Cinco fuentes, todas verificadas y con los mismos `game_id`:

| Pieza | Fuente | Módulo | Qué aporta |
|---|---|---|---|
| Calendario y marcadores | ESPN scoreboard | `ingest/espn_source.py` | 5.158 juegos, 42 peticiones |
| Box score de equipo | hoopR-nba-data (ESPN) | `ingest/hoopr_source.py` | **posesiones exactas** (`FGA + 0.44·FTA − OREB + TOV`) y **perfil de tiro** (eFG, 3PA rate, FT rate, pintura, faltas) |
| Box score de jugador | hoopR-nba-data (ESPN) | `ingest/hoopr_source.py` | minutos, `plus_minus`, DNP → **disponibilidad y on/off** |
| Cuadro arbitral | ESPN summary | `ingest/espn_source.py` | los 3 colegiados de cada noche → **perfil de faltas y tiros libres** |
| Play-by-play | hoopR-nba-data (ESPN) | `ingest/pbp_source.py` | 1.135.445 tiros con coordenadas → **calidad de tiro** y el **spread de cierre** |

Los parquet de box score pesan ~0,7 MB cada uno y bajan en segundos; el
play-by-play, ~21 MB por temporada. 109 MB de data cruda en total. Resultado
tras limpiar: **5.158/5.158 juegos válidos**, posesiones exactas en todos
(101,6 por 48 min, antes estimadas con MAE 2,93 y correlación 0,76 contra las
reales) y **19.110 ausencias de rotación** en 4.708 juegos (1,85 por
equipo-partido).

De 2026-27 entran solo los **235 juegos que hoopR ya tiene**, no los 240 que
ESPN da por finalizados: los de anoche aún no tienen box score ni
play-by-play, y meterlos dejaría cinco juegos con posesiones estimadas y sin
calidad de tiro. El pipeline los aceptaría (degrada, no falla), pero un
control contaminado no sirve para medir. Esa espera de un día es el estado
normal de un pipeline en vivo.

La disponibilidad se reconstruye del box score: para cada partido, los
jugadores que estaban en la rotación de los 10 partidos anteriores (≥15 min de
media, ≥3 apariciones) y no juegan ese día. Sus pesos —minutos y on/off por 100
posesiones— salen **solo de partidos anteriores**, con encogimiento hacia 0.
El on/off es real, no un proxy: `plus_minus` es el margen con el jugador en
pista (verificado: la suma por equipo-partido es exactamente 5× el margen), así
que `on_off = net_on − net_off`.

Dos avisos honestos: (1) *saber quién faltó* sale del box score final, lo que
equivale a suponer que conocías el lineup al apostar —cierto para el informe
oficial de las 17:30 ET, optimista para un scratch de última hora—; (2) por eso
`late_scratch_flag` vale 0 en toda la data real. Con un feed histórico de
informes de lesiones se sustituye `build_availability` y nada más cambia.

### Métricas

Descargados con [`ingest/espn_source.py`](nba_sh/ingest/espn_source.py), porque
`stats.nba.com` no es alcanzable desde muchos entornos (timeout) y
basketball-reference devuelve 403. La API de ESPN —que la guía lista como
fuente— acepta meses completos: 4 temporadas en 40 segundos y 33 peticiones.

Sanidad de la data: 1.230 / 1.231 / 1.231 / 1.231 juegos por temporada, 5%
de prórrogas, **14,1 back-to-backs por equipo y temporada**. El Kings 176 –
Clippers 175 de 2023 (351 puntos, 2 OT) obligó a subir el techo de validación
de totales: era 350 y rechazaba un juego real.

Los All-Star se descartan solos, con cualquiera de sus formatos (`SHQ`/`CAN` en
2023, `STARS`/`STRIPES`/`WORLD` en 2026). Y eso **no es cosmético**: hoopR
etiqueta los cuatro partidos del mini-torneo del All-Star 2026 como temporada
regular (`season_type = 2`), con marcadores de 35-37 y 47-21. Si el filtro
fuera el campo de la fuente y no la validación de equipos, esos cuatro
entrarían en el dataset y envenenarían las medias de la temporada.

La liga se vuelve más impredecible cada año, y eso marca el techo de
cualquier modelo:

| Temporada | Total medio | sd del margen | sd del total | MAE spread ciego |
|---|---|---|---|---|
| 2022-23 | 229,4 | 13,67 | 19,49 | 10,91 |
| 2023-24 | 228,4 | 15,64 | 20,27 | 12,48 |
| 2024-25 | 227,6 | 15,89 | 19,83 | 12,68 |
| 2025-26 | **231,2** | 16,41 | 19,87 | 13,09 |
| 2026-27 (235 j.) | 227,2 | **16,44** | 19,2 | 13,26 |

El salto de anotación de 2025-26 **no ha continuado**: en lo que va de 2026-27
se anota 227,2, cuatro puntos menos. La dispersión del margen, en cambio,
sigue subiendo.

2025-26 anotó 3,6 puntos más por partido que 2024-25 y su margen se dispersó
a 16,41. El reentreno cada 14 días absorbió el desplazamiento sin tocar nada:
el sesgo del total en esa temporada es **−0,01 pts**.

Y aquí la capa de monitoreo (§14) se gana el sueldo por primera vez con data
real. El PSI de las features entre 2022-25 y 2025-26 dispara **10 alertas de
44**, encabezadas por las que miden exactamente lo que cambió:

| Feature | PSI |
|---|---|
| `total_base` | **0.73** |
| `expected_possessions` | 0.24 |
| `three_rate_sum` | 0.24 |
| `fouls_sum` | 0.21 |
| `rim_rate_sum` | 0.16 |

Es decir: el sistema habría avisado de que la liga cambió —más ritmo, más
triples, más faltas— en vez de degradarse en silencio. No es una prueba
sintética: es el cambio real de la NBA entre 2024-25 y 2025-26.

Validación externa del Elo al cierre de 2025-26 — ordena la liga real sin que
nadie se lo diga, y nadie se lo dice: el Elo solo ve resultados partido a
partido, nunca la clasificación.

| Elo | Victorias reales |
|---|---|
| OKC 1749 | OKC 64 |
| SAS 1748 | SAS 62 |
| BOS 1705 | DET 60 |
| DET 1695 | BOS 56 |
| DEN 1679 | DEN 54 |

Los cinco primeros son los cinco primeros, con BOS y DET intercambiados. Al
fondo: WAS 1195, BKN 1251, UTA 1265. Es la mejor prueba de que la temporada
nueva entró bien.

Prediciendo con reentreno cada 14 días, la cadencia de producción de la guía
§14.3. **Tres temporadas de test**, cada una entrenando solo con lo anterior:

| Test | Log loss | Brier | ECE | MAE spread | MAE total | MAE del cierre |
|---|---|---|---|---|---|---|
| 2023-24 | 0.6095 | 0.2111 | – | 11.08 | 14.76 | **10.53** |
| 2024-25 | **0.5971** | **0.2057** | 0.038 | **10.87** | **14.70** | **10.52** |
| 2025-26 | 0.6038 | 0.2080 | **0.029** | 11.49 | 15.21 | **10.82** |
| 2026-27 (235 j.) | 0.6180 | 0.2143 | – | 11.86 | **14.70** | **11.08** |

La temporada que publicaba antes este README (2024-25) era **la mejor de las
cuatro**. Sobre las tres completas la media está en log loss 0.603, MAE spread
11,14 y MAE total 14,89.

La última fila es la temporada **en curso**, como control: 235 juegos hasta el
20 de noviembre de 2026, entrenando con las cuatro anteriores. Sale la peor en
probabilidad y spread, pero **235 juegos no distinguen 0.618 de 0.604** —el
error estándar del Brier a esa muestra es 0,013 y la diferencia con 2025-26 es
0,006, medio error estándar—. Es un control, no un veredicto; vale para
confirmar que el pipeline ingiere la temporada viva sin romperse y que no hay
una degradación grosera.

Un detalle que el control destapó: las medias móviles de 10 partidos **cruzan
el cambio de temporada**. Los 30 equipos arrancan 2026-27 con `pace_last_10`,
`efg_last_10` y `shot_quality_last_10` ya poblados, heredados del final de
2025-26. El Elo sí trata ese salto con carryover explícito (regresión a la
media); las medias de forma no reseteaban nada. En octubre, por tanto,
describen plantillas que ya cambiaron.

El desglose por capas de abajo sigue medido sobre 2024-25, para que las
comparaciones sean entre iguales:

| Modelo | Log loss | Brier | MAE spread | MAE total |
|---|---|---|---|---|
| **Completo (walk-forward)** | **0.5971** | **0.2057** | **10.87** | **14.70** |
| Completo (entrenado una vez) | 0.5967 | 0.2056 | 10.86 | 14.74 |
| Cierre de ESPN (el mercado) | – | – | **10.52** | – |
| Sin capa situacional | 0.6093 | 0.2110 | 11.08 | 14.99 |
| Solo Elo | 0.6142 | 0.2132 | 11.07 | – |
| Baseline (ventaja local fija) | 0.6894 | 0.2481 | 12.63 | 15.84 |

Qué aportó cada pieza de data nueva (entrenando 2022-24, prediciendo 2024-25):

| Variante | Log loss | Brier | MAE spread | MAE total |
|---|---|---|---|---|
| Base (posesiones estimadas, sin lesiones) | 0.6064 | 0.2099 | 10.92 | 15.05 |
| + posesiones exactas | 0.6052 | 0.2094 | 10.92 | 15.02 |
| + disponibilidad reconstruida | 0.5973 | 0.2059 | 10.89 | 15.03 |
| + vía aprendida de totales (ridge) | 0.5971 | 0.2058 | 10.89 | 14.69 |
| + cuadro arbitral | 0.5973 | 0.2059 | 10.89 | 14.74 |
| + calidad de tiro (play-by-play) | **0.5967** | **0.2056** | **10.86** | 14.74 |

La disponibilidad mueve el mercado de ganador (**−0.009 de log loss**); la vía
aprendida mueve los totales (**−0.34 de MAE**); el cuadro arbitral no mueve
nada; la calidad de tiro mueve el margen (**−0.03 pts**) y la probabilidad
(**−0.0014 de log loss**).

Una corrección: una versión anterior de esta tabla atribuía los −0.34 de
totales al "perfil de tiro" del box score. Era falso, y lo destapó la propia
capa de play-by-play: al validar, el `Dataset` se reconstruía con su
constructor y `team_box` llegaba **vacío** al feature store por su valor por
defecto. El eFG y compañía valían 0 y nada fallaba. Con el box score vivo su
aportación medida es **−0.002 de MAE en totales**: cero. Los −0.34 eran de la
vía aprendida, no del perfil de tiro. El arreglo (`replace` en vez del
constructor, en `validation.py`) y su test de regresión están en el repo.

Estado frente a los objetivos de la guía: **cumple log loss** (<0.62) en las
tres temporadas completas y **Brier** (<0.21) en dos de tres (2023-24 se pasa
por 0.001); el control en curso se sale de los dos, con la muestra que tiene.
Queda a 0,6 pts de MAE spread (11,1 de media vs <10.5) y a 0,9 de MAE totales
(14,9 vs <14). Y el objetivo que no estaba en la guía: el cierre de ESPN le
gana el MAE de spread las cuatro veces.

### Por qué los totales no bajan de 14

Los totales tenían vía estructural (ritmo × eficiencia) pero **no** vía
aprendida, al contrario que el spread. Añadirla —ridge sobre 23 features
agregadas: ritmo, eficiencia, eFG propio y concedido, 3PA rate, FT rate,
faltas, contraataque, pintura, acierto y reparto de zonas— baja el MAE de
15.03 a 14.69 y sube la
correlación con el total real de 0.325 a 0.390. El peso de la mezcla (0.85 al
ridge) se eligió en validación, entrenando con 2022-23 y evaluando en 2023-24,
sin mirar la temporada de test.

El techo está medido: dándole al modelo las **posesiones reales del partido**
(trampa imposible en producción), el MAE baja solo a 13.52. Y el ritmo es
predecible hasta correlación 0.40, nada más. Es decir, bajar de 14 exige
adivinar el ritmo casi exactamente; el objetivo de la guía asume una
desviación del total menor que la real (19,8-20,3 según la temporada).

Sobre el ECE (0.032-0.039 según la temporada, vs objetivo <0.02): **no es
miscalibración medible**. Con
1.231 juegos, simulando resultados desde las propias probabilidades del modelo,
el ECE esperado bajo calibración *perfecta* es 0.029 de media con p95 = 0.042.
El observado cae dentro de ese ruido: para resolver un 0.02 de verdad hacen
falta ~4 temporadas, no una.

Para dimensionar el resto: la desviación real del margen llegó a **16,41** en
2025-26 (no 12, como asume la guía), así que no saber nada da MAE 13,09. El
modelo baja a 11,49, un **12% mejor**; en 2024-25 la ventaja era del 14%, así
que parte del empeoramiento es la liga y parte es el modelo —el cierre de
ESPN perdió un 2,9% de precisión entre esas dos temporadas y el modelo un
5,7%. En totales el modelo saca un 4-7% al ciego: ahí ya no falta data de
posesión, ni perfil de tiro, ni cuadro arbitral, ni calidad de tiro —los
cuatro están medidos—, sino ritmo, que es lo que no se deja predecir.

Un hallazgo de modelado que salió de estos datos: aplicar el impacto de una
baja al **total** empeoraba el MAE de totales (+0.14) sin ganar nada. El on/off
mide margen, no puntos anotados —si falta una estrella, su equipo anota menos
pero el rival anota más—, así que ahora las bajas mueven solo el margen
(`InjuryConfig.injury_total_share = 0`). El generador sintético se corrigió
igual, para que la verdad simulada y el modelo asuman lo mismo.

### El cuadro arbitral: efecto real, ganancia nula

La guía pide una capa arbitral (§5.6) y da una cifra: 3-5 puntos entre los
cuadros extremos. Se descargó entera —una petición a `summary` por partido,
5.158 en ~50 minutos— y se midió antes de creérsela.

Cobertura: **5.128 de 5.158 juegos** (99,4%) con sus tres colegiados, **94
árbitros** distintos, 4.935 cuadros con historial suficiente (≥10 partidos por
colegiado). El perfil de un cuadro es la media de sus tres colegiados, y el de
cada colegiado son **solo sus partidos anteriores**: un `crew_id` por noche,
porque la composición cambia cada noche. Los tiros libres por partido del
cuadro van de 43,9 (p5) a 48,3 (p95), con media 45,81.

El efecto existe y pasa el filtro de la propia guía (regla (d) de §5.5A.3:
efecto > 0,7 pts con muestra > 200): la pendiente bruta del total contra los
tiros libres previos del cuadro es **+0,99 pts por tiro libre** y la parcial,
sobre el residuo del modelo, **+1,33**. El coeficiente de producción es 0,90
—la pendiente medida, encogida hacia el 0,42 de la guía—, lo que deja **4,1
puntos entre el cuadro del p5 y el del p95**, dentro del rango 3-5 que la guía
afirma. Y casi ningún partido llega al recorte de ±5.

Medir esto destapó un sesgo: el ajuste estaba centrado en los tiros libres por
partido de la liga (44,6) y no en la media de los **perfiles de cuadro**
(45,83, algo más alta porque un perfil es una media de partidos anteriores).
Esa media hay que remedirla al añadir temporadas —con tres salía 46,1, con
cuatro 45,83— porque ningún test denuncia un ajuste descentrado.
Centrado mal añadía +1,35 pts al total estructural —+0,20 al total final, que
pesa el estructural al 15%— en todos los partidos; ahora el ajuste tiene media
**−0,04** y desviación 1,28.

Y aun así **no mejora la precisión**. En validación (2023-24) la rejilla de
coeficientes 0 / 0,42 / 0,90 / 1,33 se mueve 0,028 puntos de MAE en total
—14,826 sin árbitros, 14,798 con el mejor—, o sea que la rejilla es plana y no
sirve para elegir; por eso el coeficiente sale de la pendiente medida, no de
ella. En test el MAE de totales va de 14,67 a **14,71**. Es ruido en ambas
direcciones. La explicación es que el modelo ya conoce las faltas y los tiros
libres **de los propios equipos**, y el cuadro arbitral apenas añade sobre eso.

Se queda en el modelo, por tres razones: la data es real y causal, el efecto
descriptivo pasa la regla de la guía, y el coste es cero. Pero queda dicho:
**no es de aquí de donde van a salir los totales por debajo de 14**.

### Calidad de tiro: dónde se tira, no si entra

El box score dice si el tiro entró; el play-by-play dice **de dónde salió**.
Con eso se parte el eFG en sus dos mitades, que no se comportan igual:

    puntos por tiro = calidad (dónde tiras) + acierto (si lo metes)

La calidad de un tiro son los puntos que vale a conversión de liga, sacados de
una tabla `(banda de distancia × familia de tiro) → puntos por intento`. La
tabla de cada temporada se estima **con la anterior** —la de 2023-24 sale de
2022-23—, así que ningún partido entra en su propia referencia; por eso se
baja una temporada extra (2021-22) que solo sirve de referencia. Entre
temporadas consecutivas la tabla correlaciona **0.96-0.99**, de modo que la
elección no puede estar decidiendo el resultado.

Data: **1.135.445 tiros de campo** en 6 temporadas, coordenadas en el 100%,
cobertura **5.157/5.158 juegos** con los dos equipos. La geometría se validó
sola: el 99,9% de los triples cae más allá de 21 pies y el 99,93% de los dobles
dentro de 23,75. La línea de tres se reconstruye con el arco de 23,75 y la
esquina de 22, y clasifica bien el **99,88%** de los tiros contra la verdad de
2024-25 (en los anotados no hace falta: `score_value` lo dice exacto).

El patrón que sale es el de libro, y es justo lo que hace útil la métrica:

| Distancia | Intentos (2024-25) | Puntos por intento |
|---|---|---|
| 0-4 pies | 50.589 | **1.34** |
| 4-8 | 28.454 | 0.90 |
| 8-16 | 29.537 | 0.88 |
| 16-20 | 8.313 | 0.83 |
| 20-23,75 | 19.080 | 1.09 |
| 23,75-27 | 54.821 | **1.09** |
| 27-35 | 21.166 | 1.03 |

Tres hallazgos, dos de ellos contra lo que yo esperaba:

**1. El acierto es más estable que la calidad.** Partiendo 2024-25 en partidos
pares e impares, la fiabilidad (Spearman-Brown) del acierto es **0.905** y la
de la calidad **0.790**. Esperaba lo contrario: que la selección fuera lo
repetible y el acierto, varianza. En la NBA actual los equipos con tiradores
baten su expectativa de forma persistente.

**2. La calidad predice el margen, no el total.** Contra el resultado de
2024-25: `shot_quality_diff` correlaciona **+0.290** con el margen (el net
rating de 10 partidos, +0.415), pero la suma de la calidad de los dos equipos
correlaciona **+0.006** con el total. Tiene sentido visto de frente: elegir
mejor los tiros decide quién gana, no cuántos puntos se anotan. Por eso la
calidad está en la matriz de margen y **no** en la de totales, donde sí está el
acierto (+0.197, al nivel del eFG).

**3. Aporta poco, pero aporta en los dos splits.** A diferencia del cuadro
arbitral, aquí los signos coinciden en validación y en test:

| Variante (walk-forward) | Log loss | MAE spread | MAE total |
|---|---|---|---|
| Sin nada de tiro | 0.5991 | 10.894 | 14.715 |
| Solo perfil de box score | 0.5991 | 10.894 | 14.713 |
| **+ calidad de tiro del pbp** | **0.5977** | **10.865** | **14.699** |

El box score no aporta nada (−0.002 de MAE). El play-by-play sí: −0.0014 de log
loss, −0.029 de MAE spread y −0.014 de MAE total en test, con validación de
acuerdo en los tres signos. Son migajas, pero son migajas consistentes.

De las seis features que generó, cuatro entraron en la matriz de totales y dos
se cayeron en validación: `shot_quality_sum` y `shot_quality_allowed_sum`, las
de correlación ~0. La matriz de totales tiene 23 features y la de margen 22.

### El carryover de las medias de forma: no cambia nada

Si las medias de 10 partidos arrastran la temporada anterior, lo lógico es
probar a cortarlas. Se implementó con un único peso,
`FormConfig.season_carryover_weight`, que pondera los partidos de temporadas
anteriores dentro de la ventana:

    media = (suma_en_temporada + w · suma_anterior) /
            (n_en_temporada    + w · n_anterior)

Con `w = 1` sale exactamente la media de siempre (verificado valor a valor en
los tests) y con `w = 0`, el reset duro. El cambio toca **536 de 5.158 juegos
(10%)**, todos dentro de los 10 primeros de cada temporada: quirúrgico, como
debe ser.

Medido sobre el arranque de las tres temporadas completas juntas (n=466, para
tener muestra):

| Peso | Arranque: log loss | Arranque: MAE spread | Arranque: MAE total | Todo: MAE total |
|---|---|---|---|---|
| **1.0 (producción)** | 0.6139 | 10.304 ±0.374 | **14.522** | **14.890** |
| 0.5 | 0.6140 | 10.302 | 14.537 | 14.892 |
| 0.0 (reset duro) | 0.6129 | 10.304 | 14.710 | 14.914 |

**El MAE de spread es idéntico a los tres pesos** —0,002 puntos de diferencia
contra un error estándar de 0,374—, el log loss se mueve 0,001, y el reset
duro **empeora** los totales 0,19 pts en el arranque. Esa pérdida tiene causa
conocida y predicha antes de medir: sin carryover, `efg_last_10` vale 0 en el
estreno de temporada, y un eFG de 0 no es neutro, es absurdo.

Así que el peso se queda en **1.0**: el comportamiento de siempre. La fuga
entre temporadas es real como hecho de diseño, pero no cuesta precisión
medible, probablemente porque el Elo —que sí tiene carryover propio— lleva la
mayor parte del peso predictivo en octubre.

Lo que sí se gana es código: las tres expresiones de media móvil duplicadas
(`rolling.py`, `shooting.py`, `shot_quality.py`) viven ahora en un solo helper
probado, `features/windows.py`, y el knob queda documentado para quien quiera
revisarlo con más temporadas.

Y un aviso de los tests: la primera versión del helper dejaba el estreno de
temporada en NaN con cualquier `w` entre 0 y 1, porque la suma móvil dentro de
la temporada sale NaN —no 0— cuando no hay partidos previos, y eso anulaba
toda la mezcla. **Las primeras cifras de `w = 0,5` que medí estaban mal por
eso.** Lo cazó un test escrito a propósito para el caso límite, y la medición
se repitió.

### El modelo contra el mercado: no lo bate

El fichero de play-by-play traía algo que no buscaba: `home_team_spread`, el
**spread de cierre de ESPN**, en 5.157 de 5.158 juegos. Es la única línea real
contra la que medirse en todo el proyecto, y la respuesta es la que menos
gusta.

**Precisión: el mercado gana, las tres temporadas.**

| Temporada | MAE del cierre | MAE del modelo | corr(modelo, cierre) |
|---|---|---|---|
| 2023-24 | **10.53** | 11.08 | +0.87 |
| 2024-25 | **10.52** | 10.87 | +0.85 |
| 2025-26 | **10.82** | 11.48 | +0.86 |
| 2026-27 (235 j.) | **11.08** | 11.86 | +0.86 |

**ATS: tampoco.** Apostando el lado donde el modelo discrepa del cierre, a
−110 (break-even 52,38%):

| Temporada | Acierto (desacuerdo ≥2 pts) | n | ROI |
|---|---|---|---|
| 2023-24 | 48,1% | 690 | −8,1% |
| 2024-25 | 54,8% | 757 | +4,7% |
| 2025-26 | 47,7% | 757 | −9,0% |
| 2026-27 (en curso) | 46,1% | 152 | −12,1% |
| **Agregado** | **50,0%** | **2.356** | **−4,5%** |

Y agregado por umbral de desacuerdo:

| Umbral | n | Acierto | z vs 52,38% | ROI |
|---|---|---|---|---|
| ≥ 0 pts | 3.878 | 50,8% | −1,94 | −3,0% |
| ≥ 1 pt | 3.077 | 50,7% | −1,83 | −3,1% |
| ≥ 2 pts | 2.356 | 50,0% | −2,31 | −4,5% |
| ≥ 3 pts | 1.746 | 49,4% | −2,47 | −5,6% |
| ≥ 5 pts | 807 | 50,6% | −1,04 | −3,5% |

Ningún umbral llega al break-even, y en los de 2 y 3 puntos el modelo queda
**significativamente por debajo** (z ≈ −2). Donde el modelo se aparta de la
línea, se aparta para mal.

Con una salvedad de potencia que conviene decir: con 2.356 apuestas el edge
**mínimo detectable** era 2,89 pp, y un edge explotable típico es de 1-3 pp.
Esto descarta un edge grande; no descarta uno pequeño. Resolver eso por
resultados exigiría ~19.600 apuestas, y es justo para lo que está el
paper-trading con CLV de §4c.

Y aquí está la lección, que es la del §15 de la guía en carne propia, vivida
en tres pasos dentro de este mismo proyecto:

1. Con **solo 2024-25**, el umbral de 2 puntos daba **54,8% y +4,7% de ROI**
   sobre 757 apuestas. Parecía un sistema.
2. Al añadir 2023-24 (48,1%) bajó a 51,6%, y esta sección pasó a decir "no hay
   edge detectable".
3. Con 2025-26 (47,7%) y el arranque de 2026-27 (46,1%) está en **50,0%**, ya
   significativamente por debajo del break-even.

Cada bloque de data nuevo lo ha empujado en la misma dirección, y de las
cuatro ventanas solo una quedó por encima. Una temporada de 1.231 juegos no
distingue un 54,8% real de un 50% con suerte: la desviación del acierto a ese
tamaño es 1,8 puntos porcentuales.

Tres avisos sobre esta medición: (1) ESPN no documenta el momento de captura de
`home_team_spread` —se asume cierre, pero no está certificado—; (2) no hay
precios, así que se asume −110 a dos caras, cuando un libro real cobra más caro
el lado popular; (3) faltan el total y el moneyline, y el CLV sigue sin poder
medirse porque hace falta la **apertura** además del cierre.

Lo que esto dice del proyecto: el modelo correlaciona 0,85-0,87 con el cierre
y es 0,35-0,8 pts peor en MAE. Reproduce la línea razonablemente bien, y lo que añade
por su cuenta no es información, es error. Para batir al mercado falta lo que
el mercado tiene y este modelo no: lineups confirmados en tiempo real, minutos
esperados jugador a jugador, y el propio movimiento de la línea.

El motor de apuestas **sigue sin poder ejercitarse del todo**: con el cierre se
puede medir acierto ATS, pero sin precios no hay ROI fiel y sin apertura no hay
CLV, que es la métrica que de verdad decide si esto vale algo. `ODDS_API_KEY`
no está definida en este entorno.

Lo que queda, en orden de impacto: (1) un feed de cuotas con apertura, cierre y
**precios** de los tres mercados, (2) lineups confirmados en tiempo real y
minutos esperados por jugador —lo que el mercado sabe y aquí falta—, (3) un
feed histórico de informes de lesiones que sustituya la reconstrucción.

Artefactos: `data/artifacts/real_metrics_4seasons_*.json` (métricas),
`data/artifacts/control_2026_27_*.json` (la temporada en curso),
`data/artifacts/market_comparison_4splits_*.json` (modelo contra el cierre),
`data/artifacts/carryover_pooled_*.json` (el peso entre temporadas; vale el más
reciente, el anterior lleva las cifras previas al arreglo del helper),
`data/raw/hoopr/` (box scores y play-by-play crudos, 109 MB),
`data/raw/officials/officials.parquet` (asignaciones arbitrales de los 5.158
juegos), `data/raw/real_5seasons_*/` (snapshot inmutable del dataset limpio y
completo, seis tablas).

```python
from dataclasses import replace

from nba_sh.ingest.espn_source import build_referee_tables, fetch_dataset, fetch_officials
from nba_sh.ingest.hoopr_source import (
    apply_exact_possessions,
    build_availability,
    load_box_scores,
)
from nba_sh.ingest.pbp_source import load_shot_quality
from nba_sh.pipeline import model_quality, prepare

seasons = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
dataset = fetch_dataset(seasons)
box = load_box_scores(seasons)
# box.canonical_team es la tabla del Dataset (con el detalle de tiro);
# box.team solo trae lo justo para calcular posesiones.
team_box = box.canonical_team

games, _ = apply_exact_possessions(dataset.games, team_box)
# Una petición por partido: ~45 min para 4 temporadas. Guárdalo en parquet.
officials = fetch_officials(games.game_id)
games, referees = build_referee_tables(games, officials, team_box)
# El pbp baja también 2021-22: es la referencia de la primera temporada pedida.
shot_quality = load_shot_quality(seasons)
dataset = replace(
    dataset,
    games=games,
    team_box=team_box,
    referees=referees,
    shot_quality=shot_quality,
    injuries=build_availability(games, box.player),
)
print(model_quality(prepare(dataset)))
```

## 4c. Paper-trading con CLV: cómo saber si esto sirve antes de 2040

**Nada de esto envía una apuesta a ningún sitio.** El paquete `paper/` solo
escribe ficheros en `data/paper/`.

### Por qué CLV y no ROI

Los números que siguen son del propio proyecto y son la razón de montarlo:

| Edge real sobre el break-even | Apuestas para probarlo (80% potencia) | Equivalente |
|---|---|---|
| 1 pp (53,4% ATS) | **19.622** | ~16 temporadas apostando todo |
| 2 pp (54,4%) | 4.906 | 4 temporadas |
| 3 pp (55,4%) | 2.180 | 1,8 temporadas |

Y una corrección a lo que decía este README: la medición contra el cierre de
ESPN usó 2.356 apuestas, con lo que su **edge mínimo detectable era 2,89 pp**.
Un edge explotable típico es de 1-3 pp. Así que aquel 50,0% ATS descarta un
edge *grande*; **no descarta uno pequeño**. Fui demasiado categórico.

La CLV resuelve eso. No pregunta si acertaste, pregunta si cogiste un número
mejor que el que el mercado acabó fijando. Quita el azar del partido —sd del
margen 16,4 pts— y deja el del movimiento de línea, un orden de magnitud
menor: ~100 veces menos muestra para la misma precisión. De 16 temporadas a
unos cientos de apuestas.

### El ciclo de un día

Los horarios son los de §14.1 de la guía, y `python -m nba_sh plan` los
imprime ya con su comando:

```powershell
python -m nba_sh paper capture --season 2026-27 --timing open
```

```powershell
python -m nba_sh paper bet --season 2026-27
```

```powershell
python -m nba_sh paper capture --season 2026-27 --timing close
```

```powershell
python -m nba_sh paper settle --season 2026-27
```

```powershell
python -m nba_sh paper record
```

```powershell
python -m nba_sh paper report
```

La CLV sale de comparar el cierre con la línea tomada al apostar, así que el
orden importa: sin las dos puntas no hay nada que medir. El libro guarda qué
captura se usó para cada apuesta.

Si la captura de cierre no ocurrió, hay rescate:

```powershell
python -m nba_sh paper reconstruct-close
```

```powershell
python -m nba_sh paper settle --season 2026-27 --resettle
```

`record` responde «¿qué ha pasado?» (historial, balance corrido, ROI por
mercado) y `report` responde «¿tiene esto edge?» (CLV con su potencia). Son
preguntas distintas y la primera engaña más, así que el record imprime siempre
la probabilidad de ese récord bajo puro azar.

### Decisiones de diseño que importan

**El libro es append-only.** Cada operación escribe un fichero nuevo con su
timestamp y nunca toca uno anterior (`data/paper/odds/<timing>/<ts>.parquet`,
`bets/<ts>.parquet`, `settled/<ts>.parquet`). Un proceso que muere a medias no
corrompe nada, dos capturas del mismo día no se pisan, y repetir una
liquidación no duplica la apuesta: vale la más reciente.

**La jornada por jugar tiene su propia vía.** `pipeline.run_daily` evalúa un
día que ya está en el frame validado, con marcador: sirve para backtest, no
para apostar esta noche. Y la validación de la guía rechaza —con razón— filas
sin marcador. Así que `paper/slate.py` valida el histórico como siempre y mete
el slate **después**, solo al feature store. Es seguro porque las features son
causales por construcción, y exige una sola fecha por llamada: si un equipo
apareciera dos veces, su segunda fila heredaría sus propios NaN. Esas guardas
son errores explícitos, no comentarios.

**La fuente de cada captura queda registrada.** `--payload-file` permite
probar la tubería con un JSON guardado, y el reporte se **niega a concluir** si
alguna cuota no viene de la API: *"esto prueba la tubería, no mide nada"*.

**Sin cierre no hay CLV.** Una apuesta sin captura de cierre se liquida con
resultado y P&L, pero su CLV queda a NaN. No se inventa.

### Lo que imprime el reporte

Media de CLV, desviación, **error estándar**, t, p, el porcentaje de apuestas
que batió el cierre, la CLV en puntos de línea, el ROI de papel, el desglose
por mercado y —lo que lo hace una herramienta de decisión— **cuántas apuestas
faltan** para resolver el umbral pedido y el mínimo detectable con la muestra
actual. El veredicto es una frase, y una de ellas es "CLV significativamente
NEGATIVA: no sigas".

Por debajo de 100 apuestas con CLV se niega a dar veredicto.

### Estado: la primera jornada real, y lo que enseñó

El 23 de noviembre de 2026 se capturó la apertura de DraftKings de los 10
partidos, el modelo evaluó 60 oportunidades y se registraron 5 apuestas de
papel. Salieron **5-0, +602,90 sobre 600 de stake (ROI +100,5%)**.

Ese número no significa nada, y el propio `paper record` lo dice: **un 5-0 sale
por azar el 3,1% de las veces** si cada apuesta fuera una moneda. Por eso el
record imprime siempre esa probabilidad, para que nadie —incluido quien lo
escribió— lea una racha como habilidad.

Lo que sí dice algo es la CLV, y cuenta una historia distinta del 5-0:

| Pick | Apertura | Cierre | Puntos | CLV |
|---|---|---|---|---|
| MEM −3.5 | −3.5 | −2.5 | **−1,0** | −0.024 |
| POR +3.5 | +3.5 | **−1.5** | **+5,0** | +0.145 |
| POR ml +142 | +142 | −110 | — | +0.105 |
| CHI +8.5 | +8.5 | +8.5 | 0,0 | −0.007 |
| SAS −10.5 | −10.5 | −8.5 | **−2,0** | −0.088 |

Media **+0,40 puntos de línea** y +0.027 de CLV, pero dos de las cuatro cogieron
un número **peor** que el cierre. El 5-0 dice «perfecto»; la CLV dice «un número
buenísimo, dos malos y uno neutro». Esa es exactamente la diferencia por la que
se montó todo esto.

### La captura de cierre falló, y el rescate que salió de ahí

La tarea programada corrió a su hora (23:30 UTC) y **paró**: `ODDS_API_KEY` no
estaba en el entorno de la tarea. Comportamiento correcto —avisó en vez de
inventar— pero el resultado era que esas 5 apuestas se quedaban sin CLV para
siempre.

De ahí salió `paper reconstruct-close`: el endpoint `summary` de ESPN publica un
`pickcenter` con la línea del libro **después** del partido. Su campo `spread`
es el del equipo local, el mismo convenio que `spread_home` —verificado contra
los 10 partidos: MIN favorito da −10.5 y CHA no favorito da +2.5—, y equivocar
ese signo invertiría toda la medición, así que hay un test que lo fija.

Dos salvaguardas alrededor:

- La captura reconstruida se guarda con **fuente propia** (`espn-pickcenter`),
  no se hace pasar por una captura a T−30. El reporte lo avisa en el veredicto.
  Las cuotas sintéticas (`file`) sí siguen invalidando la medición entera.
- `paper settle --resettle` reliquida lo ya liquidado, que es lo que hace falta
  cuando el cierre llega tarde. Es idempotente: el libro conserva la
  liquidación más reciente por apuesta, así que repetirlo no duplica nada.

### Lo que falta para que esto mida de verdad

**`ODDS_API_KEY` en el entorno de la app**, no solo en una terminal. El entorno
de la app se fija al arrancarla, así que hay que reiniciarla después de
definirla. Sin eso, cada jornada depende de reconstruir el cierre a posteriori,
que funciona pero no es una captura propia a T−30.

Y el punto frágil que no desaparece: capturar **tarde** es peor que no capturar.
Una línea tomada después del tipoff es una línea en vivo, y etiquetarla como
cierre envenenaría la CLV sin que nada fallara. Por eso
`capture --timing close` descarta los partidos que ya han empezado y avisa de
cuántos.

## 5. Decisiones donde el código se aparta de la letra de la guía

Todas están comentadas en el módulo correspondiente:

1. **`theta` de la binomial negativa.** La guía sugiere ~35; con medias de 113
   puntos eso implica σ ≈ 22 por equipo (la real es ~12). Se mantiene la forma
   funcional, se estima por MLE —como la propia guía indica— y se añade el
   "ajuste de dispersión" de §7.1, porque los puntos NBA están en el límite de
   Poisson (`models/totals.py`).
2. **Correlación entre marcadores.** La guía da 0.15–0.25 como referencia; se
   **estima** del histórico, porque de ella dependen a la vez la dispersión del
   margen y la del total.
3. **Probabilidad de ML.** A los dos enfoques de §8.1 (Elo + logit) se añade un
   tercero: la probabilidad implícita en la distribución simulada del margen.
   Sin él, el ML contradecía al spread del mismo partido (correlación 0.85 →
   0.97). La mezcla final se recalibra con Platt, que es lo que mete el ECE por
   debajo del objetivo (0.033 → 0.015).
4. **Matriz de diseño.** No incluye el agregado situacional junto a sus
   componentes: esa colinealidad volteaba los signos del logit.
5. **Sizing.** El stake se calcula sobre el edge *encogido* hacia el mercado
   (`edge_shrink_for_sizing`, 0.5). Los umbrales de la guía deciden si se
   apuesta; el encogimiento decide cuánto, porque parte del edge medido es
   error de estimación propio (§1).
6. **`rest_days`** son días *de descanso* (B2B = 0), no días de calendario.
7. **HFA.** El dinámico (B2B visitante, altitud, gira) entra en el Elo; los
   puntos base usan el estático para no contar la fatiga dos veces.
8. **Carryover de las medias de forma.** La guía no dice nada del salto entre
   temporadas para las medias móviles (solo para los ratings). Aquí hay un peso
   explícito, `FormConfig.season_carryover_weight`, medido y dejado en 1.0
   —arrastrar la temporada anterior no cuesta precisión y cortarla empeora los
   totales— (`features/windows.py`).
9. **Coeficiente arbitral.** La guía da 0.42 pts por tiro libre sobre la media;
   la pendiente medida en temporadas reales es 0.99 bruta y 1.33 parcial. Se
   usa 0.90 —medida y encogida hacia la guía— porque así el recorrido entre
   cuadros extremos cae en los 3-5 pts que la propia guía afirma
   (`config.RefereeConfig`).

## 6. Los errores comunes de §15, y cómo se evitan aquí

| Error | Mitigación en el código |
|---|---|
| Look-ahead bias | features causales por construcción + test que trunca el futuro y exige features idénticas |
| Sobreajuste al backtest | 20 features, ridge/logit regularizados, walk-forward, reporte de sensibilidad |
| Ignorar el vig | todo se compara contra probabilidades sin vig (multiplicativo o Shin) |
| Apostar cada juego | umbrales por mercado; el 88% de las oportunidades se descarta |
| Stakes progresivos | Kelly fraccionado con cap del 2% y stop-loss diario |
| Ignorar correlación | descuento por segundo mercado del mismo juego y cap de exposición por equipo |
| No registrar lo no apostado | `evaluated` guarda las seis salidas de cada juego con el motivo del rechazo |
| Cambiar el modelo tras 3 días malos | reentrenamiento cada 14 días y degradación por CLV, no por resultados |
| Sistemas públicos | cada spot pasa la validación histórica de §5.5A.3d o se descarta |

Con las 4 temporadas esa validación mantiene **solo** `lookahead` (+0.84 pts,
960 juegos) y descarta `sandwich` (+0.38 < 0.70), `revenge` (−0.53) y `letdown`
(37 juegos, muestra insuficiente). Así mueren los spots del folklore — y nótese
que el efecto de `lookahead` bajó de +1.10 con tres temporadas a +0.84 con
cuatro: incluso el que sobrevive se encoge en cuanto hay más data.

## 6b. Automatización en GitHub Actions

El ciclo de paper-trading corre solo, en la nube, sin depender de que ninguna
app esté abierta. Dos workflows:

| Workflow | Cuándo | Qué hace | Necesita secret |
|---|---|---|---|
| `tests.yml` | push y PR | ruff, pytest con cobertura y humo del pipeline sintético | no |
| `paper-trading.yml` | cron ×3 al día | apertura → apuestas → cierre → cuadre | solo los dos primeros |

Los tres momentos, en UTC: **16:00** captura la apertura y registra las
apuestas, **23:30** captura el cierre, **14:00** del día siguiente liquida,
calcula la CLV y actualiza el record. El libro se commitea de vuelta al repo,
así que el historial de la medición queda público y auditable.

**El paso de cuadre no necesita la clave**, y eso es deliberado: si la captura
de cierre falla, reconstruye la línea del pickcenter de ESPN. La medición no
puede perderse por una clave ausente — que es exactamente lo que pasó la
primera noche con una tarea programada local.

### El secret

Uno solo, en *Settings → Secrets and variables → Actions*:

| Secret | Para qué |
|---|---|
| `ODDS_API_KEY` | Clave de [The Odds API](https://the-odds-api.com). Sin ella el paso de apertura falla con un error claro y el de cierre avisa sin romperse. |

La clave se lee únicamente de la variable de entorno; no se escribe en disco ni
en los logs, y el workflow solo comprueba si está vacía, nunca la imprime.

Para correr un paso a mano: *Actions → paper-trading → Run workflow*, eligiendo
`open`, `close` o `reconcile`.

### Lo que no está en el repo

- **Los volcados de hoopR** (109 MB de box scores y play-by-play de ESPN): no
  son fuente de este proyecto y se redescargan solos. El workflow los cachea.
- **El snapshot limpio sí está** (1 MB), así que el modelo entrena sin esperar
  ninguna descarga.
- **Las asignaciones arbitrales también** (52 KB): reconstruirlas cuesta ~50
  minutos de peticiones a ESPN.
- **La guía** de la que sale todo esto: no me consta su autoría, así que no la
  publico. Las referencias a secciones del README apuntan a ella.

## 7. Data real

```python
from dataclasses import replace

from nba_sh.ingest.injury_source import load_injuries
from nba_sh.ingest.nba_api_source import fetch_dataset
from nba_sh.ingest.odds_source import fetch_odds_payload, payload_to_odds
from nba_sh.pipeline import prepare, run_daily

dataset = fetch_dataset(["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"])
odds = payload_to_odds(fetch_odds_payload())          # requiere ODDS_API_KEY
dataset = replace(dataset, odds=odds, injuries=load_injuries("data/injuries.csv"))
slate = run_daily(prepare(dataset))
```

- La clave de The Odds API se lee de la variable de entorno `ODDS_API_KEY`;
  nunca se escribe en el código ni en los logs.
- Guarda **apertura y cierre** de cada línea: sin los dos extremos no hay CLV.
- Las lesiones entran por fichero (`game_id, team, player_id, minutes_pg,
  on_off_net, status, hours_before_tip`) y se validan al cargarlas.
- Los árbitros son opcionales: sin asignación, su ajuste es 0.
- El play-by-play también: sin él, las features de calidad de tiro valen 0 y
  los modelos no las usan. Baja una temporada más de la que pides, porque la
  referencia de la primera sale de la anterior.

Producción: `python -m nba_sh plan` imprime el cron de §14.1 y
`scheduler.build_scheduler` lo monta con APScheduler.

## 8. Tests

```powershell
python -m pytest -q
```

```powershell
python -m pytest -q --cov=nba_sh --cov-report=term-missing
```

322 tests, 90% de cobertura. Incluyen la prueba fuerte de causalidad
(recalcular las features sin los juegos futuros y exigir que el pasado no
cambie), la verificación de las cinco reglas de peso situacionales, la masa de
push en líneas enteras, los límites de bankroll y la liquidación de los tres
mercados con push.

Los tests de las dos últimas capas encontraron dos fallos reales, los dos del
mismo tipo —un error tapado por un valor por defecto o por un `except`
amplio—:

1. `fetch_officials` referenciaba una constante inexistente y un
   `except Exception` convertía el `NameError` en «este partido no tiene
   árbitros». La función nunca había funcionado. Ahora solo se capturan los
   errores de red y de parseo.
2. Al validar, el `Dataset` se reconstruía con su constructor y las tablas que
   `validation.py` no valida (`team_box`, y luego `shot_quality`) llegaban
   **vacías** al feature store por sus valores por defecto. El perfil de tiro
   valía 0 y nada fallaba; falseó una medición publicada en este mismo README.
   Ahora se usa `replace`, y hay un test de regresión que lo fija.
3. La primera versión del helper de ventanas con peso por temporada dejaba el
   estreno de temporada en NaN con cualquier peso entre 0 y 1, porque la suma
   móvil dentro de la temporada sale NaN —no 0— sin partidos previos. Falseó
   las primeras cifras de `w = 0,5` que medí.
4. En el paper-trading, el motor de apuestas recibía las líneas con el índice
   reseteado y las descartaba todas con «sin linea de mercado»; el histórico se
   elegía por orden alfabético, con lo que `real_shots` (3 temporadas) ganaba a
   `real_5seasons`; y el flag `--simulations` de `paper bet` no se llamaba como
   el que lee la configuración (`--sims`), así que argparse lo aceptaba y se
   ignoraba en silencio. Los tres salieron de correr el ciclo contra la jornada
   real, no de los tests. Ahora los tres tienen uno.
5. `paper record --save` llamaba a `Storage.write_artifact` sin el argumento
   `version`, que es obligatorio: reventaba **después** de imprimir el record
   entero, así que parecía haber funcionado.

El patrón se repite tanto que ya es una regla de la casa: **ruff y pytest en
comandos separados**, nunca con `&&`, y desconfiar de cualquier total que no
cuadre.

## 9. Responsabilidad

- Verifica la legalidad de las apuestas deportivas en tu jurisdicción y usa
  solo operadores licenciados.
- El vig medio de la NBA (−110) equivale a ~4.8% de desventaja: el modelo
  reduce la desventaja, no la elimina.
- No apuestes dinero destinado a gastos esenciales; fija un límite mensual.
- Criterio de salida a dinero real de la guía: CLV positiva en ≥ 250 apuestas
  de paper trading **y** ROI de backtest > 2% con IC 95% que no incluya 0.
- Este código es educativo y no constituye asesoría financiera ni garantía de
  ganancias.

## 10. Limitaciones conocidas

- La liga sintética sirve para verificar el pipeline, no para estimar tu edge.
- **El modelo no bate el cierre de ESPN**: es 0,35-0,8 pts menos preciso en
  MAE las cuatro ventanas medidas, y su desacuerdo con la línea rinde 50,0%
  ATS, bajo el 52,38% del −110 (z = −2,31). 2024-25 a solas daba 54,8% y era
  ruido. Eso descarta un edge grande, **no uno de 1-2 pp**: a esa muestra el
  mínimo detectable era 2,89 pp.
- El paper-trading con CLV está midiendo, pero con **5 apuestas**: nada
  concluible (un 5-0 sale por azar el 3,1% de las veces). Hacen falta ~100 con
  CLV, unos dos meses.
- La clave no estaba en el entorno de la app cuando corrió la tarea programada,
  así que el primer cierre hubo que **reconstruirlo** del pickcenter de ESPN.
  Funciona y queda marcado como tal, pero no es una captura propia a T−30.
- La liga se vuelve más impredecible: la sd del margen pasó de 13,67 (2022-23)
  a 16,44 (2026-27). Los MAE de temporadas distintas no son comparables sin
  mirar antes esa cifra.
- La calidad de tiro sí está (§3.4, desde el play-by-play), pero aporta migajas:
  −0,03 pts de MAE spread. Lo que falta para los totales es ritmo, no tiro.
- 2026-27 **sí** está en el dataset (235 juegos hasta el 20 de noviembre de
  2026), pero como control, no como test: a esa muestra el error estándar del
  Brier es 0,013, así que no sirve para aceptar ni descartar nada.
- Las medias móviles de forma (`*_last_10`) arrastran la temporada anterior:
  en octubre describen plantillas que ya cambiaron. Se midió cortarlas y no
  cambia nada (ver 4b), así que el peso se queda en 1.0; el knob está en
  `FormConfig` por si con más temporadas la respuesta cambia.
- La disponibilidad real se reconstruye del box score final, así que asume que
  conocías el lineup al apostar; `late_scratch_flag` vale 0 en data real.
- El informe oficial de lesiones no se scrapea (cambia de formato); se carga
  desde fichero a propósito.
- El `letdown` no detecta remontadas de 15+ puntos. Ahora el play-by-play está
  descargado, así que es viable: solo se usa para tiros, no para el marcador.
- El drawdown del backtest (30.8%) roza el objetivo del 30% por la varianza de
  las primeras 200 apuestas; con 0.25 Kelly y cap del 2% es lo esperado, y la
  guía ya advierte de no concluir nada con menos de 300 apuestas.
