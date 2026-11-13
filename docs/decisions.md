# Decision log

Cambios respecto a PLAN.md y decisiones tomadas por el camino. Las más recientes van primero.

## 2026-11-13: Fase 2, modelo y experimentos

### Features

- **Cada fila se construye con un corte por serie.** `build_features` recibe, para cada día objetivo, el último día publicado de la demanda y el de `Gene`, y no lee nada posterior. `training_frame` arma las filas de entrenamiento como si cada día se hubiera pronosticado en vivo: demanda hasta 6 días antes del objetivo y `Gene` hasta 4. Un corte igual o posterior al día objetivo es un error.
- **Ventana de 28 días publicados que termina en el corte de la demanda.** Son cuatro semanas enteras, así que todas las ventanas tienen la misma mezcla de días. De ella salen el nivel (la media de la ventana) y el perfil.
- **El modelo pronostica la demanda relativa al perfil.** El perfil es la media, hora a hora, de los días de la ventana que tienen el mismo tipo de día que el objetivo. Para un festivo entre semana es el perfil de los domingos y festivos recientes, que es como se comporta (`notebooks/01_exploration.ipynb` §7). El pronóstico es la salida del modelo por el perfil.
- **Ninguna feature de carga está en kWh.** Todas son razones contra el nivel o contra el perfil, porque la demanda se duplicó entre 2000 y 2025 y los árboles no extrapolan. Un test comprueba que duplicar las dos series no cambia ninguna feature.
- **20 features.**
  - Calendario del día objetivo (7): hora, día de la semana, tipo de día, festivo, víspera de festivo, día siguiente a un festivo y día del año.
  - Ventana (7): forma del perfil, cuántos días del mismo tipo hay, el mismo día de la semana más reciente y la media de los cuatro (los dos baselines, relativos al perfil), cuántos de esos cuatro son de otro tipo de día, el nivel de la última semana frente al de la ventana y cuántos festivos hay en la ventana.
  - Días más recientes (6): cuánto se desvió de su propio perfil el último día publicado de la demanda y los dos últimos de `Gene`, en el día entero y en la hora objetivo. Es la señal de que la semana viene más alta o más baja de lo normal.
- **Lo que quedó fuera.** El mes (el día del año lo cubre), la antigüedad de cada corte (con un retraso fijo es una constante) y el clima (PLAN.md §2).
- **Las horas marcadas como `anomaly` no se excluyen del entrenamiento.** Casi todas son del 1 de enero y del 25 de diciembre, que es justo lo que el modelo tiene que aprender. Las del apagón de 2007 quedan fuera de los años de entrenamiento.

### Modelo

- **Un solo LightGBM para las 24 horas, con la hora como feature.** La variante de 24 modelos no se probó: PLAN.md la deja para cuando sobre tiempo.
- **Objetivo L1**, decidido antes de ver resultados: la métrica es el MAPE y la variable que se ajusta es una razón cercana a 1, así que el error absoluto de la razón es casi el error porcentual.
- **En el entrenamiento los cortes salen de los retrasos configurados; en la predicción, de la historia que el modelo recibe.** El corte es el último día que haya de cada serie. En el backtest las dos cosas coinciden. En producción, si XM se atrasa, el modelo usa lo que hay en lugar de fallar.
- **Reproducible en una misma máquina:** semilla fija, `deterministic` y `force_row_wise`. Entre sistemas operativos LightGBM no garantiza resultados idénticos.

### Selección de hiperparámetros

- **Ventana de selección: los 12 meses anteriores a la ventana de evaluación** (2024-11-01 a 2025-10-31), con el mismo backtest walk-forward. Las features, la variable que se ajusta y los hiperparámetros se decidieron mirando solo esa ventana. La de evaluación se corrió una vez, al final, con todo ya fijado.
- **Búsqueda de 27 candidatos** (`python -m demand_forecast.cli tune`, `TUNING_GRID` en `config.py`): años de entrenamiento (3, 5, 10), hojas (15, 31, 63) y rondas (200, 400, 800), con tasa de aprendizaje 0,05.

  | Años de entrenamiento | MAPE del mejor candidato (%) | MAPE del peor (%) |
  |---|---|---|
  | 3 | 2,30 | 2,35 |
  | 5 | 2,19 | 2,24 |
  | 10 | 2,09 | 2,12 |

  En esa ventana los baselines dan 3,40 % (ingenuo estacional) y 3,43 % (promedio de 4 semanas).
- **Lo que sí se conocía de la ventana de evaluación antes de diseñar el modelo:** los errores de los baselines de la Fase 1 (por tipo de día y por hora) y el notebook exploratorio, que usa toda la historia. De ahí salió la idea de tratar los festivos como domingos. Ningún pronóstico del modelo sobre esa ventana se vio antes de la corrida final.
- **Lo único que mueve el resultado son los años de entrenamiento.** Dentro de cada valor, hojas y rondas cambian el MAPE como mucho 0,05 puntos.
- **Se tomó el ganador literal: 10 años, 63 hojas y 200 rondas (2,091 %).** El segundo (10 años, 15 hojas, 800 rondas) quedó a 0,001 puntos; elegir siempre el primero evita decidir a ojo.
- **10 años es el borde de la búsqueda** y no se probó con más. Queda como mejora posible, a medir en la ventana de selección.
- La ventana de selección se mueve junto con la de evaluación: siempre son los 12 meses anteriores, así que nunca se solapan (hay un test).

### Resultado del backtest

Ventana del 2025-11-01 al 2026-10-31, corrido el 2026-11-13 con datos completos hasta el 2026-11-08.

| Modelo | MAE (MWh) | MAPE (%) |
|---|---|---|
| Ingenuo estacional | 338,6 | 3,40 |
| Promedio de 4 semanas | 300,5 | 3,03 |
| LightGBM | 194,5 | 1,96 |

- **La meta se cumple:** 1,96 % frente a un máximo de 2,58 %. Es un 35 % menos que el mejor baseline.
- Dónde falla y dónde gana está en `docs/error_analysis.md`. En corto: los festivos entre semana pasan de un 11 % a un 2,5 %, y los peores días que quedan son caídas de la demanda que no se ven con datos de hace 4 a 6 días.

### Registro y seguimiento

- **`models/registry.json` más un archivo de texto por modelo** (`models/lightgbm-<versión>.txt`, formato de LightGBM). La versión es el momento del entrenamiento en UTC. Cada entrada guarda el hash del archivo, el commit y si había cambios sin commitear, el rango y la huella de los datos, los hiperparámetros, la lista de features y el backtest con los tres modelos.
- **Al cargar un modelo se comprueba el hash y la lista de features.** Un archivo alterado o un modelo entrenado con otras features se rechaza con un error.
- **`train` corre el backtest, entrena con todo lo publicado y registra el modelo como vigente.** La promoción condicionada (solo si es mejor que el vigente) es de la Fase 4.
- **Cada modelo pesa 1,2 MB.** Cabe en el límite de 2 MB del pre-commit, pero un reentrenamiento semanal suma unos 60 MB al año al historial de git. En la Fase 4 habrá que decidir si se conservan los archivos de los modelos anteriores.
- **MLflow registra parámetros, métricas y la versión de los datos, sin artefactos.** El modelo ya vive en `models/` con su hash, y sin artefactos MLflow no crea `mlruns/`. El almacén es `mlflow.db` (`MLFLOW_TRACKING_URI` lo cambia). `mlflow` se importa solo al registrar una corrida, así que la API de la Fase 3 no lo necesita.
- **Huella de los datos:** hash de los timestamps y valores de las dos series procesadas. Cambia cuando XM revisa cualquier día.

### Costos conocidos

- **El comando `backtest` tarda unos 30 segundos** (12 entrenamientos con 10 años de datos) porque ahora corre siempre el modelo junto con los baselines, para que la comparación sea en la misma ventana.
- **La suite de tests pasó de 6 a unos 25 segundos.** Los tests de la CLI usan un modelo de 10 rondas y ventanas de 2 meses; uno solo corre los 12 meses.
- **El backtest usa datos ya revisados.** En producción los días recientes son preliminares (revisiones de hasta un 0,04 %).

## 2026-11-13: preparación de la Fase 2

- **Retraso propio para `Gene`: `GENERATION_LAG_DAYS = 3`.** Hasta ahora solo existía `DATA_LAG_DAYS` (5), que es el de la demanda, y `available_history` lo usa por defecto. Cortar `Gene` con ese valor desperdicia los dos días que la hacen útil; cortarla con un número escrito a mano en el código de features sería un valor mágico. Las tres mediciones dan D−3 (`data_notes.md` §3 a §3.3). Es provisional, igual que el de la demanda.
- **Test que ata los retrasos a lo medido** (`tests/test_config.py`). Con los fixtures reales descargados el 2026-11-10 comprueba que ninguno de los dos retrasos configurados es menor que el observado ese día (5 y 3). Si alguien baja un retraso por debajo de la evidencia, el test falla. Subirlo sigue permitido.
- **Tercera medición del retraso, por la mañana** (`data_notes.md` §3.3). Mismo resultado que las nocturnas. `DATA_LAG_DAYS` sigue provisional hasta los snapshots del 14 al 17 de noviembre.
- **Las features se definirán respecto al corte de disponibilidad, no con rezagos fijos.** Si XM resulta no publicar en fines de semana y festivos, el retraso dependerá del día de emisión y un rezago fijo de 7 días dejaría de existir algunos días. Los baselines ya funcionan así (`_same_weekday_days` busca la semana más reciente publicada). Con esta regla la Fase 2 puede empezar antes del 17 de noviembre y el resultado de los snapshots solo cambia el corte.
- **Dependencias de la Fase 2 instaladas, con aprobación del usuario.** `lightgbm` (4.7.0) va como dependencia de ejecución, porque la API la necesita para predecir. `mlflow` (3.18.0) va en el grupo `dev`: PLAN.md §3 lo limita al desarrollo, y así no entra en la imagen de Docker. Las dos resuelven con pandas 3.0.3 y se probaron en Windows: un `LGBMRegressor` entrena con un DataFrame de pandas 3 y MLflow registra parámetros, métricas y un artefacto.
- **MLflow 3 guarda las corridas en `mlflow.db`, no en `mlruns/`.** Sin configurar nada, el almacén por defecto es SQLite (`sqlite:///mlflow.db` en el directorio de trabajo) y `mlruns/` solo recibe los artefactos. PLAN.md §3 suponía que todo iba a `mlruns/`. Se agregó `mlflow.db` a `.gitignore`. La ruta se fijará en `config.py` cuando se escriba `models/train.py`.
- **Datos actualizados** con la ingesta del 2026-11-13: `demand` hasta el 2026-11-08 y `generation` hasta el 2026-11-10.

## 2026-11-11: Fase 1, datos y baselines

### Datos

- **Un archivo Parquet por año y por serie.** `data/raw/<serie>/<año>.parquet` y `data/processed/<serie>/<año>.parquet`. Con un solo archivo por serie, cada corrida diaria reescribiría toda la historia y el repositorio crecería varios MB por commit. Así solo cambia el archivo del año en curso.
- **Capa `raw`: una fila por día, tal como la publica XM**, incluidos los días parciales, más la columna `fetched_at`. Un día cuyos valores no cambiaron no se reescribe, así que `fetched_at` dice cuándo se vio cambiar ese día por última vez y dos corridas seguidas dejan los archivos idénticos byte a byte.
- **Capa `processed`: serie horaria (`timestamp`, `value`, `anomaly`) solo con días completos y sin huecos.** Termina en el último día del tramo continuo de días completos que empieza en 2000-01-01. Si un día intermedio queda incompleto, los días posteriores se quedan fuera y la ingesta lo avisa con un warning; cuando XM lo complete, entran solos.
- **Regla de completitud.** Un día vale si su total es ≥ 60 % de la mediana de los 28 días completos anteriores. Para `DemaCome` se exige además que sea ≥ 95 % del total de `Gene` de ese día, y que `Gene` esté completa. La comprobación cruzada dejó de ser una suposición: en los 9 806 días completos de la historia la razón nunca bajó de 0,9989 (`data_notes.md` §3.1). Los primeros 28 días de la historia no tienen referencia y se aceptan.
- **Faltantes: no se imputa nada.** Un día con alguna hora nula cuenta como incompleto y no entra en `processed`. En 26 años de historia no hay ninguno.
- **Atípicos: se marcan, no se borran.** La columna `anomaly` marca las horas que se desvían más de un 35 % de la mediana de esa misma hora en los 28 días anteriores. La regla solo mira hacia atrás, para que la marca de un día no dependa de datos publicados después. Las horas marcadas **siguen contando en la evaluación**; si se excluyen del entrenamiento se decide en la Fase 2.
- **Rango plausible: entre 0,05 y 25 GWh por hora.** Fuera de ese rango la ingesta falla sin escribir nada. Cubre el apagón de 2007 (0,12 GWh) y deja margen sobre el máximo actual (12,2 GWh); un cambio de unidades de XM quedaría fuera.
- **Ventanas de 31 días calendario por petición** (el límite de la API es una diferencia de 31 días entre fechas, es decir 32 días). La re-descarga diaria de 31 días cabe en una petición por serie.
- **La ingesta descarga y valida todo antes de escribir.** Si XM falla a mitad de camino o una tabla no pasa el esquema, el disco queda como estaba.
- **`DATA_DIR` es relativo al directorio de trabajo** (`data/`), y se puede cambiar con la variable de entorno `DEMAND_FORECAST_DATA_DIR`. Los comandos se corren desde la raíz del repo.

### Evaluación

- **Ventana de evaluación: los últimos 12 meses calendario completos**, con un fold por mes. "Los últimos 12 meses" de PLAN.md §4 se concretó así, antes de ver ningún resultado, para que la ventana no se mueva cada día y los folds coincidan con el reentrenamiento mensual.
- **Corte de disponibilidad en el backtest.** Toda lectura del pasado pasa por `available_history`: un pronóstico emitido en `D` solo ve datos hasta `D − DATA_LAG_DAYS`. Hay tests que lo comprueban, entre ellos uno que altera los datos no publicados y verifica que el pronóstico no cambia.
- **Baselines.** El ingenuo estacional repite el día más reciente ya publicado con el mismo día de la semana que el objetivo; con el retraso actual es el de 7 días antes. El promedio móvil promedia, hora a hora, los 4 más recientes. Ninguno trata los festivos de forma especial.
- **Tipos de día: laboral, sábado, y domingo o festivo.** Un festivo cuenta como festivo caiga el día que caiga.

### Resultado del backtest de los baselines

Ventana del 2025-11-01 al 2026-10-31 (12 folds mensuales, 8 760 horas), pronóstico emitido un día antes con datos hasta 5 días antes de la emisión. Corrido el 2026-11-11 con datos completos hasta el 2026-11-06.

| Modelo | MAE (MWh) | MAPE (%) |
|---|---|---|
| Ingenuo estacional | 338,6 | 3,40 |
| Promedio de 4 semanas | 300,5 | 3,03 |

- **Meta para la Fase 2:** MAPE ≤ 2,58 %, un 15 % menos que el mejor baseline (3,03 %).
- **Dónde fallan:** el ingenuo estacional tiene un MAPE de 4,87 % en domingos y festivos, 3,13 % en días laborales y 2,67 % en sábados. Por hora, el peor tramo es de 14:00 a 17:00 (hasta 4,11 %).
- **Atípicos en la ventana:** ninguna hora marcada, así que la decisión de no excluirlos no afecta estos números. En toda la historia hay 111 horas marcadas en 33 días, casi todas el 1 de enero y el 25 de diciembre, más las tres horas del apagón del 2007-04-26.
- La ventana se mueve el primer día de cada mes, cuando el mes anterior queda completo. Los números de esta tabla valen para la ventana indicada.

### Costos conocidos

- **Tamaño de los datos en el repo:** 12 MB hoy (108 archivos). Cada corrida diaria reescribe los cuatro archivos del año en curso, unos 400 KB en total, lo que suma del orden de 150 MB al año en el historial de git. Si llega a molestar, las salidas son particionar el año en curso por mes o dejar de versionar `processed/`, que se puede regenerar desde `raw/`.
- **`uv sync` a secas desinstala el grupo `notebook`.** Para conservarlo hay que usar siempre `uv sync --group notebook`.
- **`jupyter execute` en Windows** lee el notebook con la codificación del sistema y falla con tildes. Hay que definir `PYTHONUTF8=1` antes de ejecutarlo.

### Dependencias

- **`holidays` se adelantó de la Fase 2 a la Fase 1** porque el desglose por tipo de día lo necesita. Aprobado por el usuario.
- **Grupo `notebook` (`matplotlib`, `ipykernel`, `nbclient`)**, aparte del grupo `dev`, solo para el notebook exploratorio. Aprobado por el usuario. Se instala con `uv sync --group notebook`.
- **`pydataxm` se quitó de `pyproject.toml`.** Los scripts `explore/01_inventory.py` y `explore/02_recent_sample.py` lo siguen usando y se corren con `uv run --with pydataxm python explore/<script>.py`.

## 2026-11-11: preparación de la Fase 1

- **Umbral de completitud: 0,60 en lugar de 0,90.** Un día vale si su total es ≥ 60 % de la mediana de los 28 días completos anteriores. Con 0,90 se rechazaba el 17,4 % de los días completos de la historia (domingos y festivos), y `08_snapshot.py summary` podía reportar el último día completo con un día de error, que es justo la diferencia entre las hipótesis A y B del retraso. Evidencia en `data_notes.md` §3.1. Los snapshots crudos no cambian; solo cambia cómo se leen.
- **Comprobación cruzada con `Gene` para `DemaCome` (provisional).** Además del umbral anterior, `DemaCome` debe ser ≥ 95 % del total de `Gene` del mismo día. Cubre un posible día a medio llenar que el umbral de 0,60 dejaría pasar. Se confirma o se descarta con los snapshots del 11 al 19 de noviembre de 2026.
- **El backtest simula el corte de disponibilidad.** La historia descargada está completa, pero en producción al emitir en `D` solo hay datos hasta `D − DATA_LAG_DAYS`. Sin simularlo habría fuga temporal en los baselines.
- **Paquete instalable con `uv_build`.** Se agregó `[build-system]` a `pyproject.toml` y el layout `src/demand_forecast/`. Se eligió `uv_build` porque viene con `uv` y no necesita configuración para el layout `src/`. No es una dependencia de ejecución.

## 2026-11-10: respuestas del usuario a la Fase 0

- **Variable objetivo: `DemaCome`, entidad `Sistema`.** La justificación está en `data_notes.md` §1.
- **Cliente de datos: REST directo con `requests` en lugar de `pydataxm`.** Esto cambia la fuente prevista en PLAN.md §1 y en el ítem `xm_client.py` de la Fase 1. Motivos (detallados en `data_notes.md` §5):
  - `pydataxm` falla la verificación TLS en Windows.
  - Se cae cuando la API responde un 400 en texto plano.
  - No tiene timeout.
  - Lanza peticiones en paralelo sin control.
  - Altera el proceso entero al importarse.

  `pydataxm` sigue en `pyproject.toml` solo porque lo usan los scripts de `explore/`. Se quitará en la Fase 1, cuando exista `xm_client.py`.
- **`Gene` (`Sistema`, horaria) entra en el MVP como feature adelantada.** Está completa 2 días antes que `DemaCome` (`data_notes.md` §6). Su ingesta y validación siguen las mismas reglas que las de la serie objetivo.
- **Commits:** el usuario crea el repositorio y hace los commits a mano. Claude solo propone los comandos y los mensajes.
- **Mensajes de commit en español** (cambia PLAN.md §3, que decía inglés). Se mantienen los prefijos de Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`). El código y los nombres siguen en inglés.

## 2026-11-10 — Fase 0

- **`uv` instalado con `pip install --user uv` (0.13.7).** No estaba instalado en la máquina. Se eligió pip en lugar del script de instalación remoto para no ejecutar código descargado. El ejecutable quedó en `%APPDATA%\Python\Python312\Scripts`, que no está en el PATH. Mientras tanto se puede usar `python -m uv ...` o agregar esa carpeta al PATH.
- **Python fijado en 3.11 con `.python-version`.** `uv init` había creado el entorno con el Python 3.12 del sistema, porque `requires-python = ">=3.11"` lo admite. Se fijó 3.11 para cumplir la sección 3 (uv descargó CPython 3.11.18).
- **Exploración en `explore/`.** Los scripts de la Fase 0 llaman al REST de XM directamente (`explore/xm_rest.py`) y no forman parte del paquete. Sirven como evidencia reproducible de `docs/data_notes.md`. Sus salidas (`explore/output/`) están en `.gitignore`.
- **TLS en Windows.** `pydataxm` (con `aiohttp`) falla la verificación TLS en esta máquina porque falta la raíz GlobalSign Root R46 en el almacén de Windows. Durante la exploración se usó `SSL_CERT_FILE=<bundle de certifi>`. Cómo resolverlo en `xm_client.py` depende de la pregunta 2 de `docs/data_notes.md` §7.
- **pandas 3.0.3.** `uv` resolvió pandas 3 y `pydataxm` 0.3.19 funcionó con él. Las columnas de texto ahora tienen dtype `str`, así que conviene tenerlo en cuenta en los tests y en Pandera.
