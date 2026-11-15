# Decision log

Cambios respecto a PLAN.md y decisiones tomadas por el camino. Las más recientes van primero.

## 2026-11-15: Fase 5, README y pulido

- **La fase se hizo antes de que la Fase 4 cumpla su criterio de aceptación**, por decisión del usuario. PLAN.md §0 pide no empezar una fase sin cerrar la anterior; aquí lo único pendiente de la Fase 4 es esperar tres días de corridas, y el README no depende de eso.
- **README en español.** Lo decidió el usuario; la recomendación de PLAN.md §8 era inglés con un resumen en español. El panel y la API siguen en inglés.
- **Sin licencia por ahora.** Lo decidió el usuario; la recomendación era MIT. Sin archivo de licencia, el código es público pero nadie tiene permiso de reutilizarlo. La casilla de PLAN.md queda abierta.
- **Autor: HoseStr**, el usuario de GitHub.
- **El README solo muestra el backtest.** El error real no se copia: está en el panel y en `/metrics`, y un número pegado en el README quedaría viejo al día siguiente.
- **La tabla del README deja de reproducirse cuando la ventana avanza.** `backtest` evalúa siempre los últimos 12 meses completos, así que hacia el 5 de diciembre de 2026 pasará a la ventana 2025-12-01 a 2026-11-30. El README lo dice. Cuando pase, hay que decidir si se actualiza la tabla (y con qué modelo) o si se agrega una opción para fijar la ventana.
- **"Qué aprendí y qué haría distinto" la redactó Claude** con lo que consta en este documento. Habla en primera persona del usuario, que debe revisarla.
- **El README dice que el proyecto se desarrolló con Claude Code.** El repo ya lo muestra (`CLAUDE.md`, `PLAN.md` y este documento); el usuario puede quitar la frase.
- **Revisión en limpio:** clon del commit `e9fd087` en la misma máquina Windows. `uv sync` tardó 25 segundos con la caché de `uv` llena, y `backtest --model lightgbm` 1 minuto 40 y dio la misma tabla (1,96 %). `pytest` pasó con 311 tests y 7 saltados (los de drift, sin el grupo `monitoring`); `forecast` y `serve` respondieron. No se probó en Linux ni con la caché vacía, y no lo siguió una persona ajena, que es lo que pide el criterio de aceptación.
- **El diagrama de arquitectura es Mermaid**, que GitHub renderiza. No se vio renderizado antes del push.

## 2026-11-15: Fase 4, automatización y monitoreo

### Bitácora de pronósticos

- **CSV y no Parquet: `data/predictions/<año>.csv`.** Cambia PLAN.md §3 solo para este archivo, con aprobación del usuario. En texto plano el historial de GitHub muestra, línea por línea, que cada pronóstico se escribió antes de su día y que ninguna línea vieja cambió. Pesa unos 500 KB al año.
- **Solo se agregan líneas** (`serving/log.py`). Una fila por hora, con el momento de emisión, la versión del modelo y el último día que vio de cada serie. `append_forecast` abre el archivo en modo de agregar y rechaza:
  - un día que ya está en la bitácora;
  - un día anterior al último registrado (no se rellenan huecos después);
  - un pronóstico emitido cuando su día ya había empezado.
- **Un día sin corrida queda vacío.** Rellenarlo después sería escribir un pronóstico que nunca se emitió.
- **Segunda comprobación en el workflow:** antes de commitear, `daily.yml` falla si `git diff` muestra alguna línea borrada o cambiada en `data/predictions/`.
- **`.gitattributes` fuerza LF** en la bitácora y en la tabla de errores, que se escriben desde Linux.

### Evaluación y alerta

- **`data/metrics/daily.csv` es derivado y se reconstruye entero en cada corrida** (`monitoring/evaluate.py`), a partir de la bitácora y de la demanda procesada. PLAN.md decía "agregar"; así el error siempre corresponde a los datos más recientes de XM, que revisa días ya publicados hasta un 0,04 %. Solo entran los días con sus 24 horas publicadas.
- **Alerta:** si el MAPE de los últimos 7 días evaluados supera 1,5 veces el del backtest del modelo vigente (hoy 2,94 %), el workflow abre un issue con la etiqueta `forecast-alert`, o comenta el que ya esté abierto. Con menos de 7 días evaluados no hay alerta.
- **El primer día evaluado llega unos 6 días después del primer pronóstico**, por el retraso de XM, y la primera alerta posible unos 13 días después.

### Job diario

- **Un solo comando, `daily`** (`pipeline.py`): ingesta, reconstrucción de los errores y registro del pronóstico de mañana. Correrlo dos veces el mismo día no cambia la bitácora.
- **Falla sin registrar un pronóstico** si XM no responde (no se escribe nada), si no hay modelo registrado o si los datos están demasiado viejos.
- **Tolerancia de atraso: 7 días sobre el retraso habitual** (`MAX_EXTRA_LAG_DAYS`), es decir, demanda de hasta D−12 y generación de hasta D−10. Medido en la ventana de selección con `explore/10_stale_data.py`, simulando que XM lleva varios días sin publicar (MAPE en %):

  | Días de atraso | Demanda hasta | Ingenuo estacional | Promedio de 4 semanas | LightGBM |
  |---|---|---|---|---|
  | 0 | D−5 | 3,40 | 3,43 | 2,09 |
  | 1 | D−6 | 3,40 | 3,43 | 2,13 |
  | 2 | D−7 | 3,91 | 3,69 | 2,30 |
  | 3 | D−8 | 3,91 | 3,69 | 2,32 |
  | 5 | D−10 | 3,91 | 3,69 | 2,38 |
  | 7 | D−12 | 3,91 | 3,69 | 2,46 |
  | 10 | D−15 | 4,20 | 3,72 | 2,62 |

  Con una semana de atraso el modelo (2,46 %) sigue por debajo del mejor baseline con datos al día (3,40 %). Siete días cubren un atraso por fin de semana y festivos como Semana Santa; más allá, el job falla y avisa. El número lo propuso Claude con esta tabla; el usuario puede cambiarlo.
- **Hora del cron: 20:17 UTC (15:17 en Bogotá).** Deja cuatro horas después de las 11:00, que es la única medición diurna de publicación de XM, y casi nueve antes de la medianoche. Si GitHub arranca la corrida después de la medianoche de Bogotá, ese día objetivo se queda sin pronóstico.

### API

- **La API ya no calcula: sirve la bitácora.** `/forecast` devuelve el último pronóstico registrado tal como se emitió, y 503 mientras la bitácora esté vacía. `/metrics` agrega el error real (`live`) al backtest. `/health` dice el último día pronosticado y el último evaluado. Así no puede haber dos versiones del pronóstico de un día.
- **La imagen incluye `data/` sin `raw/`** (datos procesados, bitácora y errores). El servicio muestra lo que había al construirse, así que el job diario pide a Render un despliegue nuevo con un deploy hook guardado en el secreto `RENDER_DEPLOY_HOOK_URL`. Sin ese secreto el paso avisa y sigue.
- **Los commits del job no disparan CI** (GitHub no lanza workflows por pushes hechos con `GITHUB_TOKEN`), así que `autoDeployTrigger: checksPass` no basta para redesplegar; por eso el deploy hook.
- **El paso de CI que consulta `/forecast` acepta 200 o 503**, porque la imagen se construye con la bitácora del repo, que empieza vacía.

### Drift

- **Evidently 0.7.23 en el grupo `monitoring`**, aprobado por el usuario. Funciona con pandas 3. Agrega 35 paquetes al lock; solo lo instalan el job diario y CI, nunca la imagen de la API.
- **Qué se compara:** las features de los últimos 28 pronósticos registrados, reconstruidas con los cortes escritos en la bitácora (lo que vio el modelo), contra las del periodo de entrenamiento del modelo vigente.
- **Solo las 10 features de carga.** Las de calendario (hora, día de la semana, día del año, festivos) y los conteos de la ventana difieren por construcción entre unas semanas y diez años, y saldrían siempre con drift.
- **El umbral por defecto de Evidently no sirve aquí.** Con 0,1 de distancia de Wasserstein normalizada, el 92,8 % de las ventanas de 28 días del propio entrenamiento saldría con drift (`explore/11_drift_threshold.py`). Una alarma que siempre suena no avisa de nada.
- **Umbral calibrado por feature:** una feature tiene drift cuando su distancia al entrenamiento supera la del 99 % de las ventanas del mismo tamaño dentro del entrenamiento (`DRIFT_QUANTILE`). Se calcula en cada corrida y tarda unos 50 segundos. Un test comprueba que la distancia calculada es la misma que reporta Evidently.
- **Comprobado con datos reales:** en un periodo normal (28 días hasta el 2026-11-15) ninguna feature queda marcada con 28 días registrados.
- **No hay informe de drift con menos de 7 días en la bitácora** (`DRIFT_MIN_DAYS`). La primera corrida en GitHub, con un solo día registrado (el festivo del 16 de noviembre), marcó `profile_shape` con 0,83 contra un umbral de 0,81. Era una falsa alarma de la calibración: las ventanas de referencia se toman cada 7 días, así que las de menos de una semana empiezan todas el mismo día de la semana (miércoles, con el modelo actual) y el umbral solo conoce ese día. Medido en los dos últimos años del entrenamiento (`explore/12_short_drift_windows.py`):

  | Ventana | Ventanas normales con alguna feature marcada | Peor día de inicio |
  |---|---|---|
  | 1 día | 8,8 % | domingo, 37 % |
  | 2 días | 19,8 % | sábado, 96 % |
  | 3 días | 6,5 % | sábado, 21 % |
  | 7 días | 1,7 % | 3 % |
  | 28 días | 2,4 % | 4 % |

  Desde 7 días la ventana cubre todos los días de la semana y la tasa no depende de cuándo empieza. La prueba anterior con un solo día registrado (un domingo, sin features marcadas) no bastaba para ver el problema. Mientras faltan días, el panel dice cuántos se necesitan.
- El paso de drift del workflow no detiene lo demás: si falla, el panel se publica igual.

### Panel

- **Una página estática** (`monitoring/report.py` y `monitoring/panel.html`), sin librerías externas: los datos van como JSON dentro del HTML y un script propio dibuja dos gráficas en SVG. Se publica en GitHub Pages como artefacto del workflow; `site/` no se commitea.
- **Contenido:** cuatro cifras (MAPE real, MAPE de los últimos 7 días evaluados con su estado, MAPE del backtest y último pronóstico), pronóstico contra real por hora de los últimos 21 días registrados, error de cada día evaluado con su media acumulada, tabla de días, resumen del drift y enlaces al código, la API y la bitácora.
- **En inglés**, como la API, porque es lo que ve un reclutador. Lo decidió Claude; cambiarlo es editar los textos de `panel.html`.
- **Estados vacíos:** la página se ve bien sin pronósticos, con un solo pronóstico y sin días evaluados, que es como estará los primeros 6 días.
- Colores validados para daltonismo y contraste en modo claro y oscuro. La vista previa se revisó con capturas en ambos modos.

### Reentrenamiento

- **`retrain.yml`, domingos a las 12:17 UTC**, con el comando `retrain` (`models/promote.py`).
- **Comparar un modelo recién entrenado con el vigente en las últimas semanas tendría fuga:** el candidato ya vio esos días. PLAN.md lo pedía así; se cambió por una comparación fuera de muestra.
- **Qué se compara:** en los días publicados después del entrenamiento del modelo vigente, el vigente congelado contra modelos reentrenados cada 7 días con lo publicado en su momento, que es lo que habría producido una rutina de reentrenamiento. Ninguno de los dos lados vio los días que pronostica.
- **Se promueve solo con 28 días de evidencia o más** (`RETRAIN_MIN_DAYS`) y si los reentrenados tuvieron menor MAPE. Entonces se entrena con todo lo publicado y se registra como vigente. Con el primer modelo, del 2026-11-13, la primera promoción posible es hacia mediados de diciembre.
- **La decisión se guarda siempre** en `models/registry.json`, en la lista `decisions`: ventana, días, MAPE de cada lado, si se promovió y por qué.
- El reentrenamiento del workflow no usa MLflow, que es solo de desarrollo.

### Pendiente de comprobar

- **Primera corrida en GitHub, el 2026-11-15:** CI pasó en `09025ed` (tests en Linux e imagen de Docker, que no se habían podido probar en local por falta de memoria). El job diario, lanzado a mano a las 12:04 de Bogotá, registró el pronóstico del 16 de noviembre, publicó el panel en https://hosestr.github.io/energuia/ y redesplegó la API, que ya sirve ese pronóstico.
- **Lo que aún no se ha visto correr:** el cron (la única corrida fue manual), la evaluación contra el dato real (el 16 se publica hacia el 21 de noviembre), `retrain.yml` (primer domingo: 22 de noviembre) y la apertura del issue de alerta.
- **Workflows programados:** según la documentación de GitHub, en un repo público se desactivan tras 60 días sin actividad. El job commitea a diario, lo que debería contar como actividad; que los commits del bot cuenten no se confirmó en la documentación oficial.
- **Criterio de aceptación:** tres días seguidos de ejecución automática. Empieza a contar con la primera corrida.

## 2026-11-14: Fase 3, servicio y CI

### Emisión del pronóstico

- **Una sola función emite el pronóstico: `serving/forecast.py::issue_forecast`.** La usan la API y el comando `forecast`, y la usará el job diario de la Fase 4. El día de emisión es la fecha de Bogotá del reloj, venga en la zona horaria que venga (el servidor y CI corren en UTC), y el día objetivo es el siguiente.
- **En vivo la historia se usa entera, sin `available_history`.** Las tablas procesadas solo contienen días ya publicados, así que el corte es lo que hay. La función rechaza una historia que contenga el día de emisión o uno posterior: en vivo es imposible, y si pasa es que el reloj está mal o se está repitiendo un día pasado sin cortar los datos. Para repetir un día pasado hay que cortar antes con `history_at`.
- **El pronóstico dice hasta qué día vio cada serie** (`demand_last_day` y `generation_last_day`), pero **no hay un límite de antigüedad**: con datos de hace un mes el servicio sigue respondiendo. Fijar ese límite exige medir cuánto empeora el error con cortes más viejos, en la ventana de selección. Queda para la Fase 4, junto con la regla de que el job diario falle si no hay datos nuevos.

### API

- **Tres rutas:** `GET /health`, `GET /forecast` y `GET /metrics`, más la documentación automática en `/docs`. Los esquemas están en `serving/schemas.py`.
- **`/forecast` calcula el pronóstico en cada petición** con el modelo vigente y los datos que tiene el servicio. `issued_at` es el momento de ese cálculo. No hay caché: son 24 filas y el modelo es determinista, así que dos peticiones del mismo día devuelven los mismos valores. Cuando exista la bitácora de la Fase 4, la API debe servir lo que diga la bitácora, para que nunca haya dos versiones del pronóstico de un día.
- **`/metrics` devuelve por ahora el backtest del modelo vigente**, tal como está en `models/registry.json`, con los dos baselines. PLAN.md pide el "error reciente", que no existe hasta que la Fase 4 compare pronósticos emitidos con datos reales.
- **Todo en kWh**, la unidad de XM: `demand_kwh` y `mae_kwh`. Las horas llevan el desfase de Bogotá (`-05:00`).
- **El servicio no arranca sin modelo registrado o sin datos procesados.** Si el pronóstico no se puede calcular con los datos que hay, `/forecast` responde 503 con el motivo.
- **Comando `serve`** (`python -m demand_forecast.cli serve`) en lugar de llamar a `uvicorn` a mano. Escucha en `127.0.0.1:8000`; el contenedor usa `--host 0.0.0.0`, y el puerto se puede fijar con la variable `PORT`, que es como lo pasan las plataformas de despliegue.

### Imagen de Docker

- **Dos etapas.** La primera instala con `uv` las dependencias de ejecución del lock (sin el grupo `dev`: ni `mlflow`, ni `pytest`, ni `ruff`). La segunda es `python:3.11-slim-trixie` con ese entorno, `data/processed/` y `models/`. Corre con un usuario sin privilegios (uid 1000).
- **Debian trixie y no bookworm:** la imagen de `uv` 0.13.7 ya no se publica para bookworm.
- **Los datos y el modelo van dentro de la imagen**, así que el servicio desplegado está tan al día como su última construcción. Cómo se actualiza a diario es una decisión de la Fase 4 y depende de la plataforma.
- **Tamaño: 218 MB comprimida y 930 MB desplegada.** Casi todo es el entorno (535 MB): `pyarrow` 170 MB, `scipy` 143 MB (lo exige `lightgbm`), `pandas` 76 MB y `numpy` 73 MB. Los datos son 8 MB y el modelo 1 MB. Bajar de ahí exige dejar Parquet o `pandas`, y no vale la pena.
- **`.dockerignore` es una lista de lo permitido**, no de lo excluido: un `.env` o un archivo nuevo en la raíz no pueden entrar a la imagen por descuido.
- **El pronóstico del contenedor es idéntico al de Windows**, hasta el último decimal, para el 2026-11-15. LightGBM no lo garantiza entre sistemas operativos; con este modelo y estos datos se cumple.

### CI

- **`ci.yml` corre en cada push y en cada PR**, con dos trabajos: uno con `ruff check`, `ruff format --check` y `pytest` sobre el entorno del lock, y otro que construye la imagen, arranca el contenedor y consulta las tres rutas.
- **Versiones exactas de las acciones** (`actions/checkout@v7.0.1` y `astral-sh/setup-uv@v10.3.3`) y de `uv` (0.13.7, la misma de la máquina local y de la imagen). La imagen se construye con `docker build` a secas, sin acciones de Docker.
- **Probado en Linux antes del primer push:** `ruff` y los 233 tests pasan en un contenedor Debian trixie con el entorno del lock. El primer push (commit `d12a556`, 2026-11-14) dejó el workflow en verde en GitHub.
- **El contenedor usa 173 MB de memoria** con el servicio arrancado y las tres rutas consultadas.

### Despliegue

- **Plataforma: Render, capa gratuita.** La eligió el usuario el 2026-11-14. Hugging Face Spaces se descartó porque hay reportes de que Docker pasó a ser de pago en cuentas gratuitas nuevas, y Cloud Run porque exige cuenta de facturación y más configuración.
- **Condiciones leídas ese día en la documentación de Render:** 512 MB de RAM y 0,1 CPU, 750 horas de instancia al mes por workspace, y el servicio se duerme tras 15 minutos sin tráfico y tarda cerca de un minuto en despertar. El contenedor usa 173 MB, así que cabe. Al crear la cuenta y el servicio, el 2026-11-14, Render no pidió tarjeta. La documentación no dice cuántos minutos de construcción incluye el plan gratuito (**por verificar**).
- **`render.yaml` en la raíz** describe el servicio: imagen construida desde el `Dockerfile`, región `virginia` (la más cercana a Colombia; no se puede cambiar después), comprobación de salud en `/health` y despliegue solo cuando CI pasa en el commit (`autoDeployTrigger: checksPass`).
- **El puerto lo pone Render** en la variable `PORT` (10000 por defecto), que el comando `serve` ya lee.
- **Desplegado el 2026-11-14 en https://energuia-api.onrender.com.** Ese día las tres rutas respondieron 200 y `/forecast` devolvió para el 2026-11-15 los mismos valores que el comando `forecast` en local.
- **La raíz redirige a `/docs`.** Sin eso, quien abre la dirección a secas ve un 404.
- **Consecuencia del sueño de 15 minutos:** la primera petición después de un rato sin tráfico tarda cerca de un minuto. El panel de GitHub Pages de la Fase 4 no depende de la API, así que no le afecta.

### Dependencias

- **`fastapi` y `uvicorn` como dependencias de ejecución**, aprobadas por el usuario. Antes estaban instaladas solo porque las arrastra `mlflow`.
- **`httpx2` en el grupo `dev`, en lugar de `httpx`.** `TestClient` necesita un cliente HTTP. La fase empezó con `httpx`, pero Starlette 1.10 lo marca como obsoleto para `TestClient` y busca primero `httpx2`, y FastAPI lo documenta igual. El usuario aprobó el cambio el 2026-11-14. Solo afecta a los tests: no entra en la imagen, nada más en el lock pedía `httpx` y el código no cambió. Según PyPI lo mantiene Pydantic (`github.com/pydantic/httpx2`), con el primer release en mayo de 2026; su código no se revisó.

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
