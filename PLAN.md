# PLAN.md — Pronóstico de demanda eléctrica de Colombia (día siguiente)

> Documento de trabajo para Claude Code. Léelo completo antes de escribir código.

## 0. Cómo trabajar con este plan

- Avanza **fase por fase, en orden**. No empieces una fase hasta que la anterior cumpla sus criterios de aceptación.
- Al terminar cada fase: corre los tests, marca las casillas de este archivo, haz commit y **detente a resumirle al usuario** qué quedó hecho, qué decidiste y qué dudas hay.
- Si la realidad contradice el plan (la API devuelve otra cosa, una librería no funciona), **gana la realidad**: anota el cambio en `docs/decisions.md` y sigue. No fuerces el código para que encaje con una suposición de este documento.
- Todo lo marcado como **(por verificar)** es una suposición no comprobada. Compruébala antes de construir encima.
- No agregues herramientas fuera de la sección 3 sin preguntar. El objetivo es un sistema pequeño que funcione de verdad, no un catálogo de tecnologías.

## 1. Contexto y objetivo

Proyecto de portafolio para postulaciones como ML/AI Engineer. Lo que debe demostrar es **ingeniería de ML de punta a punta** (datos, entrenamiento, despliegue, automatización, monitoreo). La precisión del modelo importa menos que la calidad y honestidad del sistema.

**Objetivo del sistema:** cada día, de forma automática, pronosticar la demanda horaria de energía del Sistema Interconectado Nacional (SIN) de Colombia para el día siguiente (24 valores), y dejar evidencia pública de predicción contra dato real acumulada en el tiempo.

**Fuente de datos:** API pública de XM (operador del mercado eléctrico), vía la librería `pydataxm`.

**Resultado que verá un reclutador:**
1. Un README que en menos de un minuto muestre el problema, la arquitectura, la tabla de resultados y los enlaces en vivo.
2. Un panel público (GitHub Pages) con pronóstico contra real y el error acumulado.
3. Una API desplegada que devuelve el pronóstico vigente.
4. Un historial de GitHub Actions corriendo a diario sin intervención.

## 2. Fuera de alcance

- Deep learning, Kubernetes, Airflow, feature stores, bases de datos administradas.
- Servicios de pago. Todo debe correr en capas gratuitas.
- Frontend elaborado. El panel es HTML estático generado por un script.
- Variables climáticas en el MVP (quedan como extensión, sección 9).

## 3. Decisiones ya tomadas

| Tema | Decisión | Motivo |
|---|---|---|
| Lenguaje | Python 3.11 | `pydataxm` exige >= 3.10 |
| Entorno y dependencias | `uv` con `pyproject.toml` | Reproducible y rápido |
| Comandos | CLI con Typer: `uv run python -m demand_forecast.cli <comando>` | Debe funcionar igual en Windows (PowerShell) y en Linux (CI). **No usar Makefile** |
| Datos | Parquet versionado dentro del repo (`data/`) | La serie es pequeña (24 valores por día); evita infraestructura |
| Validación de datos | Pandera | Esquema explícito y testeable |
| Modelo | LightGBM | Estándar para series tabulares, rápido en CPU |
| Experimentos | MLflow con almacenamiento local (`mlruns/` ignorado por git) | Seguimiento durante el desarrollo |
| Registro de modelos | Archivo del modelo + `models/registry.json` en el repo | En CI no hay servidor MLflow persistente; esto es simple y auditable |
| API | FastAPI + Docker | |
| Automatización | GitHub Actions (cron) | Gratis en repos públicos |
| Monitoreo | Script propio de error + Evidently para drift | |
| Panel | HTML estático en GitHub Pages | Cero infraestructura |
| Calidad | pytest, ruff (lint y formato), pre-commit | |
| Idioma | Código y nombres en inglés. Mensajes de commit en español, con prefijos de Conventional Commits | Estándar de la industria; el idioma de los commits lo decidió el usuario (ver `docs/decisions.md`) |

## 4. Definición del problema

- **Variable objetivo:** demanda horaria nacional del SIN.
- **Horizonte:** día siguiente completo. En el día `D` se emite el pronóstico de las 24 horas de `D+1`.
- **Zona horaria:** `America/Bogota` (sin horario de verano). Guardar timestamps como inicio de hora.
- **Retraso de publicación:** XM publica el dato real con cierto retraso `L` (por verificar en la Fase 0). Las features solo pueden usar datos **ya publicados en el momento de emitir** el pronóstico. Parametrizar como `DATA_LAG_DAYS` en la configuración; el rezago mínimo utilizable es `horizonte + L`.
- **Métricas:** MAE y MAPE, globales y desglosadas por hora del día y por tipo de día (laboral, sábado, domingo o festivo).
- **Evaluación:** backtesting walk-forward con ventana expansiva sobre los últimos 12 meses, reentrenando mensualmente. **Prohibido cualquier partición aleatoria.**
- **Baselines obligatorios:**
  1. Ingenuo estacional: misma hora del mismo día de la semana anterior disponible.
  2. Promedio de las últimas 4 semanas para esa hora y día de la semana.
- **Meta:** el modelo debe reducir el MAPE del mejor baseline al menos un 15 % relativo. Si no lo logra, se documenta por qué; **no se ajusta la evaluación para que se vea mejor**.

## 5. Estructura del repositorio

```
energy-demand-forecast-co/
├── CLAUDE.md                  # Reglas permanentes (sale de la sección 7)
├── PLAN.md                    # Este archivo
├── README.md
├── pyproject.toml
├── Dockerfile
├── .pre-commit-config.yaml
├── .github/workflows/
│   ├── ci.yml                 # lint + tests en cada push y PR
│   ├── daily.yml              # ingesta, evaluación, pronóstico, panel
│   └── retrain.yml            # reentrenamiento semanal con promoción condicionada
├── src/demand_forecast/
│   ├── cli.py
│   ├── config.py
│   ├── data/                  # xm_client.py, ingest.py, schema.py
│   ├── features/              # build.py
│   ├── models/                # baselines.py, train.py, backtest.py, registry.py
│   ├── serving/               # app.py
│   └── monitoring/            # evaluate.py, drift.py, report.py
├── tests/
├── data/
│   ├── raw/                   # tal como llega de XM
│   ├── processed/             # serie horaria limpia
│   ├── predictions/           # bitácora de pronósticos (solo se agrega)
│   └── metrics/               # error diario
├── models/                    # modelo vigente + registry.json
├── docs/                      # decisions.md, data_notes.md
└── site/                      # panel generado (no editar a mano)
```

## 6. Fases

### Fase 0 — Reconocimiento de la API (sin arquitectura todavía)

Objetivo: reemplazar suposiciones por hechos antes de diseñar nada.

- [x] Crear el proyecto con `uv` e instalar `pydataxm`.
- [x] Listar las métricas con `ReadDB().get_collections()` y ubicar las de demanda. Candidatas: `DemaReal` y `DemaCome` con entidad `Sistema` **(por verificar)**. Elegir una y justificarlo.
- [x] Descargar una semana y documentar: columnas, unidades, convención de horas (¿columnas `Hour01` a `Hour24`?), tipos y valores faltantes.
- [ ] Medir el retraso de publicación `L`: última fecha disponible frente a la fecha de hoy. Repetir en dos días distintos si es posible. *(1 de 2 mediciones: 2026-11-10. Falta la segunda con `explore/08_snapshot.py`.)*
- [x] Determinar desde qué fecha hay historia y descargar una muestra antigua para confirmar que el formato no cambió.
- [x] Escribir los hallazgos en `docs/data_notes.md`.

Datos confirmados leyendo el código de `pydataxm` 0.3.19 (no hace falta reverificarlos):
- Firma: `ReadDB().request_data(coleccion, metrica, start_date, end_date, filtros=None)`, con fechas `YYYY-MM-DD` o `datetime.date`.
- El constructor `ReadDB()` **hace una llamada de red** para traer el inventario de métricas. Instanciarlo una sola vez y simularlo (mock) en los tests.
- Ante una métrica inexistente **imprime un mensaje y devuelve un DataFrame vacío** en vez de lanzar excepción. Validar siempre que el resultado no esté vacío.
- Al importarse ejecuta `warnings.filterwarnings("ignore")` y `nest_asyncio.apply()`, que afectan a todo el proceso. Aislar su uso en `data/xm_client.py`.
- Parte las consultas por mes internamente; los datos horarios admiten como máximo 30 días por petición.
- Si la librería da problemas, la alternativa es llamar al REST directamente: `POST https://servapibi.xm.com.co/hourly` con cuerpo `{"MetricId", "StartDate", "EndDate", "Entity", "Filter"}`.

**Aceptación:** `docs/data_notes.md` responde las cinco preguntas anteriores con evidencia. Detenerse y presentarlo al usuario.

### Fase 1 — Datos y baselines

> **Ajustes por la Fase 0** (ver `docs/data_notes.md` y `docs/decisions.md`): el cliente usa el **REST directo con `requests`** y no `pydataxm`; se ingieren **dos series**, `DemaCome` (objetivo) y `Gene` (feature adelantada), ambas con entidad `Sistema`; los días recientes llegan **parciales sin marca**. Antes de escribir código hay que convertir el proyecto en paquete: agregar `[build-system]` a `pyproject.toml` y crear el layout `src/demand_forecast/` para que funcione `uv run python -m demand_forecast.cli`.

- [ ] `data/xm_client.py`: cliente REST fino (`POST https://servapibi.xm.com.co/hourly`) con reintentos, timeout y errores explícitos. Debe:
  - partir los rangos en ventanas de 31 días como máximo;
  - distinguir el 400 "No existen datos" (resultado vacío) del 400 "Rango de consulta supera..." (error de uso);
  - convertir las horas de texto a número;
  - hacer las peticiones de forma secuencial;
  - no empezar ninguna ventana antes de 2000-01-01.
- [ ] Quitar `pydataxm` de `pyproject.toml` cuando `xm_client.py` lo reemplace. Antes, reescribir `explore/01_inventory.py` y `explore/02_recent_sample.py`, que todavía lo importan, o anotar que dependen de él.
- [ ] `data/ingest.py`: descarga histórica completa y descarga incremental **idempotente** (correrla dos veces no duplica filas). En cada corrida debe **volver a descargar y sobrescribir los últimos N días** (configurable), porque los días parciales se completan después.
- [ ] **Detección de días incompletos** con un umbral configurable: un día vale si su total es ≥ 90 % de la mediana de los 28 días completos anteriores. Los días incompletos se guardan marcados (o en `raw/`), pero no entran a `processed/` como verdad. Test con los días reales 2026-11-06 y 2026-11-07 como fixture.
- [ ] Transformar de formato ancho (una columna por hora) a largo: `timestamp`, `demand`. `HourNN` corresponde a `Date + (NN−1) h` en `America/Bogota`.
- [ ] `data/schema.py`: esquema Pandera (sin duplicados, frecuencia horaria completa, valores positivos, rango plausible).
- [ ] `config.py`: `DATA_LAG_DAYS = 5` (provisional, pendiente de las mediciones del usuario del 11 al 19 de noviembre de 2026), las métricas, la URL, el umbral de completitud y la ventana de re-descarga.
- [ ] Política de faltantes y atípicos documentada en `docs/decisions.md`. Marcar, no borrar en silencio (por ejemplo, el apagón del 2007-04-26 y los festivos; ver `data_notes.md` §4).
- [ ] `models/baselines.py` y `models/backtest.py` con la evaluación walk-forward de la sección 4.
- [ ] Un único notebook exploratorio en `notebooks/` con estacionalidad diaria, semanal, anual y efecto de festivos.
- [ ] Tests: transformación ancho a largo, idempotencia de la ingesta, esquema, backtesting sin solapamiento entre train y test.

**Aceptación:** `uv run python -m demand_forecast.cli backtest --model seasonal-naive` imprime la tabla de métricas de ambos baselines, y los tests pasan.

### Fase 2 — Modelo y experimentos

- [ ] `features/build.py`: rezagos (respetando `horizonte + L`), medias y desviaciones móviles, hora, día de la semana, mes, festivos de Colombia (paquete `holidays`), víspera y día posterior a festivo.
- [ ] **Test anti-fuga obligatorio:** para un timestamp objetivo, ninguna feature usa datos posteriores al momento de emisión. Construirlo antes que las features.
- [ ] `models/train.py`: LightGBM con registro en MLflow de parámetros, métricas y versión de los datos.
- [ ] Estrategia multi-horizonte: empezar con un solo modelo que use la hora objetivo como feature. Comparar con 24 modelos solo si sobra tiempo.
- [ ] Ajuste de hiperparámetros acotado (búsqueda pequeña, validación temporal).
- [ ] Análisis de errores: ¿dónde falla? (festivos, fin de año, horas pico). Guardarlo en `docs/`.
- [ ] `models/registry.py`: guardar modelo con metadatos (fecha, rango de datos, métricas, hash del commit) en `models/registry.json`.

**Aceptación:** el modelo cumple la meta de la sección 4 en backtesting, o existe una explicación documentada de por qué no. Hay una tabla comparativa modelo contra baselines lista para el README.

### Fase 3 — Servicio y CI

- [ ] `serving/app.py`: `GET /health`, `GET /forecast` (pronóstico vigente con versión del modelo y momento de emisión), `GET /metrics` (error reciente).
- [ ] Esquemas Pydantic y tests de la API con `TestClient`.
- [ ] `Dockerfile` multi-etapa, imagen pequeña, usuario no root.
- [ ] `ci.yml`: ruff, pytest y construcción de la imagen en cada push y PR.
- [ ] Despliegue en capa gratuita. **Preguntar al usuario** antes de elegir (candidatos: Hugging Face Spaces con Docker, Render, Cloud Run) y verificar las condiciones vigentes de cada una.

**Aceptación:** CI en verde, la imagen corre localmente con `docker run` y hay una URL pública que responde en `/forecast`.

### Fase 4 — Automatización y monitoreo

- [ ] `daily.yml` (cron diario; recordar que el cron de GitHub usa UTC y puede retrasarse):
  1. Ingesta incremental de datos reales.
  2. Evaluar los pronósticos pasados cuyos datos reales ya llegaron; agregar a `data/metrics/`.
  3. Emitir el pronóstico del siguiente día objetivo y agregarlo a `data/predictions/` con versión del modelo y momento de emisión.
  4. Generar reporte de drift con Evidently (features recientes contra el periodo de entrenamiento).
  5. Regenerar el panel y publicarlo en GitHub Pages.
  6. Commit automático de datos y métricas (requiere `permissions: contents: write`).
- [ ] **La bitácora de pronósticos es inmutable:** solo se agrega, nunca se reescribe un pronóstico ya emitido. Es la prueba de honestidad del proyecto.
- [ ] Alerta: si el MAPE móvil de 7 días supera 1,5 veces el MAPE del backtesting, abrir un issue automáticamente.
- [ ] `retrain.yml` (semanal): entrenar candidato, compararlo con el modelo vigente en las últimas semanas y **promoverlo solo si es mejor**; registrar la decisión en `models/registry.json` en ambos casos.
- [ ] El job diario debe fallar con mensaje claro si XM no responde o no hay datos nuevos, sin corromper nada.
- [ ] Verificar si GitHub desactiva los workflows programados por inactividad del repo y cómo evitarlo **(por verificar)**.

**Aceptación:** tres días consecutivos de ejecución automática sin intervención, con el panel actualizado y la bitácora creciendo.

### Fase 5 — README y pulido

- [ ] README con este orden: qué problema resuelve (2 líneas), enlaces en vivo, diagrama de arquitectura, tabla de resultados, cómo reproducir en 3 comandos, decisiones de diseño, limitaciones conocidas.
- [ ] Sección "Qué aprendí / qué haría distinto".
- [ ] Insignias de CI y del job diario.
- [ ] Licencia y `CONTRIBUTING` mínimo.
- [ ] Revisión final: clonar el repo en limpio y seguir el README al pie de la letra.

**Aceptación:** una persona ajena al proyecto puede reproducir el backtesting siguiendo solo el README.

## 7. Reglas permanentes (copiar a `CLAUDE.md` al crear el repo)

- **Sin fuga temporal.** Toda feature, partición o estadística se calcula solo con datos disponibles en el momento de emisión. Ante la duda, escribir un test.
- **Honestidad en las métricas.** No cambiar el periodo de evaluación, las métricas ni los baselines para mejorar un número. Los resultados malos se documentan.
- **Tests primero en lo crítico:** transformaciones de datos, features y backtesting.
- **Nada de red en los tests.** Simular siempre a XM.
- **Commits pequeños** con Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- **Sin secretos en el repo.** Usar GitHub Secrets y un `.env` ignorado por git.
- **Configuración centralizada** en `config.py`; nada de valores mágicos dispersos.
- **Compatibilidad Windows y Linux:** usar `pathlib`, no depender de comandos de shell específicos.
- **Type hints y docstrings breves** en funciones públicas. Sin comentarios que repitan el código.
- **Preguntar antes de** agregar una dependencia nueva, cambiar una decisión de la sección 3 o crear cuentas en servicios externos.

## 8. Preguntas abiertas para el usuario

Hacerlas cuando toque, no todas al inicio:

1. Nombre definitivo del repositorio (propuesta: `energy-demand-forecast-co`).
2. Idioma del README (recomendación: inglés, con un resumen en español).
3. Plataforma de despliegue de la API (Fase 3).
4. Licencia (recomendación: MIT).

## 9. Extensiones posteriores al MVP

Solo cuando las cinco fases estén completas y el sistema lleve semanas corriendo:

- Temperatura como variable exógena (por ejemplo, Open-Meteo) y medición de cuánto mejora.
- Pronóstico probabilístico (regresión cuantílica) con evaluación de cobertura de los intervalos.
- Pronóstico por región o por agente.
- Servidor MLflow alojado en vez del registro en archivo.
- Entrada de blog o hilo explicando el caso de drift más interesante que haya detectado el sistema.

## 10. Primer paso

Ejecuta la **Fase 0** completa y preséntale al usuario `docs/data_notes.md` antes de crear la estructura de la sección 5.
