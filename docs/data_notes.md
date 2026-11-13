# Notas de datos: API de XM (Fase 0)

Reconocimiento hecho el **2026-11-10 entre 22:30 y 22:47 (hora de Bogotá, UTC−5)** con `pydataxm` 0.3.19, Python 3.11.18 y pandas 3.0.3.
Cada afirmación remite al script de `explore/` que la reproduce (`uv run python explore/<script>.py`, ejecutado desde `explore/`). Los scripts `01` y `02` usan `pydataxm`, que desde la Fase 1 ya no es dependencia del proyecto: se corren con `uv run --with pydataxm python explore/<script>.py`. Las salidas crudas quedan en `explore/output/` (ignorado por git).

## Resumen

| Pregunta | Respuesta |
|---|---|
| Métrica elegida | **`DemaCome`, entidad `Sistema`** (demanda comercial horaria del SIN, kWh) |
| Formato | Una fila por día y 24 columnas `Hour01` a `Hour24`, con valores numéricos **como texto** en kWh |
| Convención horaria | `HourNN` = intervalo `[NN−1:00, NN:00)` en hora local. Timestamp de inicio de hora = `Date + (NN−1) h` |
| Retraso de publicación | Último día **publicado**: D−3. Último día **completo**: **D−5**. Los días D−4 y D−3 llegan **parciales y sin ninguna marca** |
| Inicio de la historia | **2000-01-01**, con el mismo formato hasta hoy: 9 808 días, sin huecos ni nulos |

## 1. Métricas de demanda disponibles

`explore/01_inventory.py`. El inventario tiene 193 combinaciones métrica-entidad (142 `MetricId`). Estas son las candidatas horarias a nivel `Sistema`:

| MetricId | Descripción de XM (resumida) | Unidad |
|---|---|---|
| `DemaReal` | Demanda de usuarios regulados y no regulados del SIN. **No incluye alumbrado público** | kWh |
| `DemaCome` | Demanda propia de cada comercializador **más su participación en las pérdidas del STN** y los consumos propios de los generadores | kWh |
| `DemaRealReg` / `DemaRealNoReg` | `DemaReal` partida en mercado regulado y no regulado | kWh |
| `DemaComeReg` / `DemaComeNoReg` | `DemaCome` partida en mercado regulado y no regulado | kWh |
| `Gene` | Generación del sistema (no es demanda, pero se comporta casi igual; ver §6) | kWh |

`DemaSIN` (la "Demanda Energía SIN" oficial) también existe, pero **solo es diaria**, así que no sirve como objetivo horario.

**Elección: `DemaCome` / `Sistema`.** Motivos:
1. Es la serie horaria más cercana a la "demanda del SIN" que publica XM. Su total diario coincide con `DemaSIN` con un error de ±0,6 % (por ejemplo, 2026-11-05: 246,13 frente a 246,19 GWh), mientras que `DemaReal` queda sistemáticamente ~1,5 % por debajo porque excluye las pérdidas del STN y el alumbrado público.
2. Su historia, formato y retraso son idénticos a los de `DemaReal` (§3, §4), así que en esos aspectos ninguna de las dos tiene ventaja.
3. Coincide hora a hora con la generación (`Gene`): diferencia media de −0,08 % y correlación de 0,9994. Eso permite contrastarla con una fuente física independiente.

## 2. Formato: columnas, tipos, unidades y horas

`explore/02_recent_sample.py`, con JSON crudo en `explore/output/raw_DemaReal_2026-11-10.json`.

Respuesta cruda de `POST https://servapibi.xm.com.co/hourly`:

```json
{"Metric": {...},
 "Items": [{"Date": "2026-10-20",
            "HourlyEntities": [{"Id": "Sistema",
                                "Values": {"code": "Sistema", "Hour01": "9233757.32000", ..., "Hour24": "9574227.10000"}}]}]}
```

- **Columnas** (después del `json_normalize` de `pydataxm`): `Id`, `Values_code`, `Values_Hour01` a `Values_Hour24` y `Date`.
- **Tipos**: las horas llegan como **cadenas** con 5 decimales. `pydataxm` no las convierte, porque su `pd.api.types.is_integer(serie)` siempre devuelve `False`, y con pandas 3 quedan con dtype `str`. Solo `Date` se convierte a `datetime64`. **Hay que convertir explícitamente con `pd.to_numeric`.**
- **Unidades**: kWh por hora, según el inventario (`MetricUnits`). Magnitudes de 2026: entre 8,3 y 12,2 GWh por hora y entre 214 y 254 GWh por día.
- **Siempre 24 columnas.** Colombia no tiene horario de verano. En los 9 808 días de la historia nunca apareció una columna 25 ni una celda vacía.
- **Convención horaria** (`explore/06_outliers_and_hours.py`): comprobada con la generación solar agregada de 369 plantas el 2026-11-03. Es ≈ 0 hasta `Hour06`, arranca en `Hour07` (419 MWh), llega al máximo en `Hour12` (2 830 MWh) y cae en `Hour18`. Con un amanecer en torno a las 05:45 y el mediodía solar de Bogotá cerca de las 11:45, solo cuadra si `Hour07` = 06:00–07:00 y `Hour12` = 11:00–12:00. Por lo tanto `HourNN` cubre `[NN−1:00, NN:00)` y **el timestamp de inicio de hora es `Date + (NN−1) horas`**, en `America/Bogota`. La curva de demanda lo confirma: el mínimo está en `Hour04` (03:00–04:00) y el pico nocturno en `Hour19`–`Hour20` (18:00–20:00).
- **Faltantes**: la API no devuelve nulos (§4). El problema real son los días parciales (§3).

## 3. Retraso de publicación `L`

`explore/02_recent_sample.py`, `explore/04_partial_days.py` y `explore/08_snapshot.py`. Medido el martes 2026-11-10 a las 22:46 hora de Bogotá:

| Métrica | Último día devuelto | Último día completo |
|---|---|---|
| `DemaCome` | 2026-11-07 (D−3) | **2026-11-05 (D−5)** |
| `DemaReal` | 2026-11-07 (D−3) | 2026-11-05 (D−5) |
| `Gene` | 2026-11-07 (D−3) | 2026-11-07 (D−3) |
| `DemaSIN` (diaria) | 2026-11-08 (D−2, con apenas 0,6 GWh) | 2026-11-07 (D−3) |

**Los días D−4 y D−3 llegan parciales y la API no lo indica de ninguna forma:**

| Día | `DemaCome` (GWh) | Relación con la mediana de 28 días | Agentes que reportan `DemaReal` |
|---|---|---|---|
| 2026-11-04 | 245,73 | 1,002 | 146 |
| 2026-11-05 | 246,13 | 1,002 | 145 |
| 2026-11-06 | **71,71** | **0,29** | **100** |
| 2026-11-07 | **61,06** | **0,25** | **100** |

En esos días parciales, además, `DemaCome` = `DemaReal` exactamente y `DemaRealNoReg` ≈ 0,7 GWh (lo normal son ~74). Falta casi todo el mercado no regulado, y la curva horaria tiene otra forma (pico al mediodía). **Un valor publicado no es un valor definitivo.**

Consecuencias para el diseño:
- La ingesta **debe detectar días incompletos** y no tratarlos como verdad. La regla y sus umbrales están en §3.1. La propuesta inicial (≥ 90 % de la mediana de los 28 días anteriores) se descartó el 2026-11-11 porque rechazaba domingos y festivos completos.
- La ingesta incremental debe **re-descargar y sobrescribir los últimos N días**, no solo agregar días nuevos, porque los días parciales se completan después.
- **Lag mínimo utilizable** si se emite en D para D+1: el último dato completo es D−5, así que la distancia hasta el día objetivo es de **6 días**. El rezago semanal (D+1−7 = D−6) **sí está disponible**, pero con un solo día de margen. El rezago de 1 día y el de 2 días **no se pueden usar**. Propuesta tentativa: `DATA_LAG_DAYS = 5`.

**Pendiente (por verificar):**
- **Más mediciones.** La primera se hizo un martes por la noche (la segunda está en §3.2). Ese día el jueves 5 estaba completo y el viernes 6 no, lo que encaja con dos hipótesis distintas:
  - **A. Días calendario:** siempre está completo hasta D−5.
  - **B. Días hábiles:** un día queda completo cuando han pasado 3 días hábiles después de él.

  Si se cumple B, el último día completo varía entre D−3 y D−6 según el día de la semana y los festivos. Con un festivo un lunes el rezago semanal se queda sin margen, y en Semana Santa no estaría disponible. Para distinguirlas, correr `uv run python explore/08_snapshot.py` cada noche del 11 al 19 de noviembre de 2026 (esa semana incluye el festivo del lunes 16) y luego `uv run python explore/08_snapshot.py compare`.
- **Revisiones posteriores.** XM liquida en varias versiones (preliminar, mensual, final). La segunda medición (§3.2) mostró que los días ya completos sí cambian, hasta un 0,039 % del total diario. Falta saber si hay revisiones mayores a más de 31 días.
- **Hora del día de publicación.** No se sabe a qué hora se actualiza. Importa para fijar el cron del job diario. La tercera medición (§3.3) la acota: un viernes a las 11:02 el avance del día ya estaba publicado.

### 3.1 Umbral de completitud (agregado el 2026-11-11)

`explore/09_completeness_threshold.py` (sin red; lee los CSV de `05_history.py DemaCome` y `05_history.py Gene`).

**Señal 1: total del día frente a la mediana de los 28 días anteriores.** Sobre los 9 778 días completos de `DemaCome` entre 2000-01-29 y 2026-11-05:

| Umbral | Días completos rechazados |
|---|---|
| 0,90 | 1 706 (17,4 %) |
| 0,85 | 625 (6,4 %) |
| 0,80 | 74 (0,8 %) |
| 0,75 | 13 (0,1 %) |
| 0,70 | 0 |
| 0,60 | 0 |

- De los 1 706 rechazados con 0,90, 1 262 son domingos y 246 son lunes, el día al que se trasladan la mayoría de los festivos (no se cruzó con un calendario de festivos). En 2026, 34 de los 44 domingos quedan por debajo de 0,90; el rango de los domingos es 0,83–0,95.
- El día completo más bajo de la historia es el 2006-01-01, con 0,725. Los días parciales observados están en 0,25–0,29.
- **Umbral elegido: 0,60**, en el hueco entre 0,29 y 0,725. `08_snapshot.py` ya lo usa.

**Señal 2: total diario de `DemaCome` frente al de `Gene`.** En los 9 806 días completos entre 2000-01-01 y 2026-11-05 la razón va de 0,9989 a 1,0617: `DemaCome` nunca quedó más de un 0,11 % por debajo de `Gene`. En los dos días parciales es 0,28 y 0,26. No depende del día de la semana ni de los festivos. (La primera versión de esta nota usaba solo los últimos 12 meses; el 2026-11-11 se descargó la historia completa de `Gene` y la regla se mantuvo.)

**Límite conocido de la señal 1.** Solo se han visto días parciales en un estado (~0,27). Si un día pasa por un estado intermedio con el mercado regulado completo y sin el no regulado, su razón sería ≈ 0,70 (el no regulado aporta ~74 de ~246 GWh) y superaría el umbral de 0,60. Ningún umbral sobre el total separa ese caso de un 1 de enero (0,725). Por eso la Fase 1 agrega la señal 2 como segunda condición para `DemaCome`, con umbral 0,95. Queda **por verificar** con los snapshots del 11 al 19 de noviembre de 2026 si esos estados intermedios existen: `08_snapshot.py summary` imprime las dos razones de los últimos 6 días de cada snapshot.

### 3.2 Segunda medición (2026-11-11, miércoles, 21:54)

`explore/08_snapshot.py` (`summary` y `compare`).

| Métrica | Último día devuelto | Último día completo |
|---|---|---|
| `DemaCome` | 2026-11-08 (D−3) | **2026-11-06 (D−5)** |
| `DemaReal` | 2026-11-08 (D−3) | 2026-11-06 (D−5) |
| `Gene` | 2026-11-08 (D−3) | 2026-11-08 (D−3) |

- **Mismo retraso que el martes:** `DemaCome` en D−5 y `Gene` en D−3.
- **La hipótesis B, tal como estaba escrita, queda descartada.** Si un día se completara al pasar 3 días hábiles, el sábado 7 y el domingo 8 se habrían completado el miércoles 11 junto con el viernes 6 (lunes, martes y miércoles). Siguen parciales (0,25).
- **Queda abierta una variante:** que XM solo avance en días hábiles. Los snapshots del jueves 12 y el viernes 13 no la distinguen de A (ambas predicen el sábado 7 y el domingo 8). Las noches que deciden son las del **sábado 14 al martes 17**: con A, el sábado 14 debe estar completo el lunes 9; con la variante, se queda en el domingo 8 hasta el martes 17.
- **Un día pasa de parcial a completo de un salto.** El viernes 6 pasó de 71,71 a 254,54 GWh (razón de 0,29 a 1,04) entre las dos noches. El sábado 7 no cambió nada (61,06 GWh). No se vio ningún estado intermedio, aunque es una sola transición.
- **Los días completos sí se revisan, pero muy poco.** Entre los dos snapshots cambiaron 9 días ya completos de `DemaCome` (11 a 13, 19 a 21 y 26 a 28 de octubre), con un cambio máximo del total diario de 0,039 %. `Gene` cambió en los mismos días y en la misma proporción. El 11 de octubre es el primer día de la ventana de 31 días, así que puede haber revisiones más antiguas que no se ven.
- **El umbral de 0,60 se confirmó en vivo:** `Gene` del domingo 8 está completa con razón 0,885. Con 0,90 se habría reportado como incompleta.

### 3.3 Tercera medición (2026-11-13, viernes, 11:02)

`explore/08_snapshot.py` (`summary` y `compare`). No hay snapshot del jueves 12.

| Métrica | Último día devuelto | Último día completo |
|---|---|---|
| `DemaCome` | 2026-11-10 (D−3) | **2026-11-08 (D−5)** |
| `DemaReal` | 2026-11-10 (D−3) | 2026-11-08 (D−5) |
| `Gene` | 2026-11-10 (D−3) | 2026-11-10 (D−3) |

- **Mismo retraso por tercera vez, ahora medido por la mañana.** Las dos mediciones anteriores eran de alrededor de las 22:00. A las 11:02 `DemaCome` ya estaba completa hasta D−5 y `Gene` hasta D−3, así que ese día XM publicó antes de las 11:00. Un job que corra después de las 11:00 de Bogotá (16:00 UTC) habría visto lo mismo que uno nocturno. Es una sola medición diurna y no dice cuánto antes de las 11:00 se publica.
- **No distingue la hipótesis A de la variante de días hábiles**, como se esperaba. El sábado 7 y el domingo 8 se completaron entre el miércoles a las 21:54 y el viernes a las 11:02; sin el snapshot del jueves no se sabe si entraron uno por día o los dos juntos. Siguen decidiendo las noches del sábado 14 al martes 17.
- **Tampoco esta vez hubo estados intermedios.** La razón `DemaCome` / `Gene` del 7 y del 8 pasó de 0,26 y 0,29 a 1,00. Con el viernes 6 ya son tres transiciones observadas, todas de un salto.
- **Ningún día completo de `DemaCome` cambió** entre el 11 y el 13 de noviembre (días del 13 de octubre al 6 de noviembre). `Gene` del 7 y del 8, que ya estaba completa el miércoles, cambió un 0,011 % y un 0,014 %, justo cuando se completó `DemaCome` de esos días.
- La ingesta de ese mismo momento dejó `demand` completa hasta el 2026-11-08 y `generation` hasta el 2026-11-10. El backtest de los baselines no cambió (3,40 % y 3,03 %).

## 4. Profundidad de la historia y estabilidad del formato

`explore/05_history.py` (logs en `explore/output/history_*.log`). Historia completa descargada mes a mes: 323 peticiones en ~9 minutos.

- **Primer día con datos: 2000-01-01**, tanto en `DemaCome` como en `DemaReal`. Antes de 2000 la API responde `400 "No existen datos..."`.
- **Formato estable**: un único conjunto de columnas desde 2000-01-01 hasta 2026-11-07, con `Id` = `code` = `Sistema` en todas las filas.
- **Integridad**: 9 808 días, **0 días faltantes**, 0 fechas duplicadas, 0 celdas nulas o vacías, 0 celdas no numéricas y 0 valores ≤ 0.
- **`Gene`** (descargada completa el 2026-11-11 con `05_history.py Gene`): también empieza el 2000-01-01, con 9 809 días sin huecos, nulos ni duplicados. Su día completo más bajo frente a la mediana de 28 días es 0,726 (2006-01-01), igual que en `DemaCome`.
- **Tendencia**: la media diaria pasa de 113 GWh en 2000 a 242 GWh en 2026. La caída de 2020 (COVID) se ve claramente: 193 frente a 197 en 2019. Hay deriva de nivel de largo plazo, lo que importa para elegir la ventana de entrenamiento.
- **Atípicos reales (no errores de formato)**: 19 horas de `DemaCome` se desvían más de un 35 % de la mediana móvil de 15 días para esa hora. La mayoría son festivos (1 de enero, Semana Santa) y un evento claro:
  - **2007-04-26, horas 11 a 13**: 0,12 a 2,4 GWh frente a ~6,8 GWh. Es el apagón nacional de ese día.
  - 2002-03-16 `Hour24` y 2002-03-17 `Hour01`: caídas del 58 % al 69 %, sin causa identificada (por verificar).

  Política propuesta para la Fase 1: **marcar y no borrar** (columna de indicador) y decidir caso por caso si se excluyen del entrenamiento y de la evaluación.

## 5. Comportamiento de la API y de `pydataxm` (además de lo que ya dice PLAN.md)

| Hallazgo | Evidencia | Impacto |
|---|---|---|
| Rango máximo por petición: `EndDate − StartDate ≤ 31 días` (32 días inclusivos). Si se excede, la API responde `400 "Rango de consulta supera el maximo..."` | `explore/05_history.py` y la sonda manual | Partir las descargas por mes |
| Si no hay datos, la API responde **HTTP 400 con texto plano**, no un JSON vacío | Sonda manual de 1995 y 1999 | `pydataxm` llama a `response.json()` sin revisar el estado, así que **fallaría con un error confuso** |
| Una ventana que **empieza** antes de 2000-01-01 devuelve "No existen datos" **aunque contenga días válidos**. En cambio, una que **termina** en el futuro funciona y devuelve lo disponible | 1999-12-20 a 2000-01-10 → 400; 2026-10-20 a 2026-11-10 → 200 | La descarga histórica debe empezar exactamente en 2000-01-01 |
| `pydataxm` usa `aiohttp`, que valida TLS contra el almacén de certificados del sistema. **En esta máquina Windows falla** (`CERTIFICATE_VERIFY_FAILED`) porque falta la raíz `GlobalSign Root R46`. El servidor envía la cadena completa (`openssl s_client`: verify OK) y con el bundle de `certifi` (`requests`) funciona | `explore/03_tls_check.py` | Para explorar se usó `SSL_CERT_FILE=<certifi>`. Linux en CI probablemente tenga esa raíz (por verificar) |
| `pydataxm` no tiene timeout ni reintentos y lanza las peticiones de cada mes **en paralelo**, todas a la vez | Lectura de `pydataxm.py` | Riesgo de saturar o de ser bloqueado al descargar la historia completa |
| Los textos del inventario vienen en UTF-8 correcto. Los caracteres raros ("M�xima") eran solo de la consola de Windows | `od -c` sobre `inventory.csv` | Ninguno |

## 6. Observación: `Gene` como indicador adelantado

`explore/07_gene_vs_demand.py`. Comparación horaria entre noviembre de 2025 y noviembre de 2026 (8 880 horas):

| | Diferencia media | Percentil 5 | Percentil 95 | Máxima |
|---|---|---|---|---|
| `Gene` frente a `DemaCome` | −0,08 % | −0,04 % | 0,00 % | 3,99 % |
| `Gene` frente a `DemaReal` | +1,48 % | +0,68 % | +2,30 % | 3,45 % |

`Gene` está **completa hasta D−3**, dos días antes que `DemaCome`. Usar `Gene` de D−3 y D−4 como feature acortaría el lag efectivo de 6 a 4 días, pero agrega una segunda serie a ingestar y validar. **Queda como pregunta para el usuario**, no como decisión.

## 7. Preguntas que surgen para el usuario

> **Respuestas del 2026-11-10** (ver `docs/decisions.md`): 1. sí, se usa `DemaCome`; 2. se usa el REST directo; 3. `Gene` entra en el MVP; 4. la medición sigue pendiente.

1. **Objetivo**: ¿estás de acuerdo con `DemaCome` / `Sistema`?
2. **Cliente de datos**: `pydataxm` dio problemas reales (TLS en Windows, falla ante un 400, sin timeout, peticiones en paralelo sin control). PLAN.md prevé llamar al REST directamente con `requests` en ese caso. ¿Lo hacemos así en `xm_client.py`? `requests` ya está en el entorno como dependencia transitiva; habría que declararlo explícitamente.
3. **`Gene` como feature adelantada** (§6): ¿dentro del MVP o como extensión?
4. **Segunda medición del retraso**: ¿puedes correr `08_snapshot.py` otro día, o se diseña con `DATA_LAG_DAYS = 5` y una detección dinámica de días completos, y se confirma después?
