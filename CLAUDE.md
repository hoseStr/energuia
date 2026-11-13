# CLAUDE.md

Reglas permanentes del proyecto (copiadas de PLAN.md §7). El plan por fases está en `PLAN.md`.

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

## Forma de trabajo con el usuario

- **Git lo maneja el usuario.** No ejecutar `git init`, `git add` ni `git commit`. Al cerrar cada fase, entregar un bloque con los comandos listos para copiar, agrupados en commits pequeños.
- **Mensajes de commit en español**, con los prefijos de Conventional Commits en inglés (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Conversación y documentos en `docs/` en español; código, identificadores y docstrings en inglés.
- `uv` no está en el PATH de la máquina del usuario: usar `python -m uv ...` (o `.venv/Scripts/python.exe` directamente). Con el entorno virtual activado (`(demand-forecast)` en el prompt), `python -m uv` falla con "No module named uv" porque `uv` está instalado en el Python 3.12 del sistema; en ese caso usar `python <script>` a secas.

## Estado y hechos clave (detalles en `docs/data_notes.md` y `docs/decisions.md`)

- Fase 0 terminada el 2026-11-10, Fase 1 el 2026-11-11 y Fase 2 el 2026-11-13 (210 tests). Lo siguiente es la Fase 3. De la Fase 2 queda una casilla abierta: fijar los retrasos con los snapshots del 14 al 17 de noviembre; si cambian, hay que volver a correr `backtest` y `train` y actualizar los resultados.
- Resultado de la Fase 2 en la ventana 2025-11-01 a 2026-10-31: LightGBM MAPE 1,96 % (meta ≤ 2,58 %). Detalle en `docs/error_analysis.md`.
- El modelo pronostica la demanda relativa al perfil: la media hora a hora de los días del mismo tipo que el objetivo en los 28 días publicados hasta el corte. Todas las features de carga son razones, nunca kWh.
- Cualquier decisión de modelado (features, hiperparámetros) se toma en la ventana de selección, los 12 meses anteriores a la de evaluación (`cli tune`). La ventana de evaluación no se usa para ajustar nada.
- Comandos de la Fase 2: `python -m demand_forecast.cli tune` (27 candidatos, unos 25 minutos) y `python -m demand_forecast.cli train` (backtest, entrenamiento con todo lo publicado y registro en `models/registry.json`). `backtest --model lightgbm` desglosa el error del modelo; tarda unos 30 segundos porque siempre corre el modelo y los baselines.
- `train` guarda en el registro el commit y si había cambios sin commitear: correrlo después de commitear el código.
- `lightgbm` (dependencia de ejecución) y `mlflow` (grupo `dev`) ya están instalados y probados. MLflow 3 guarda las corridas en `mlflow.db` (SQLite) y solo los artefactos en `mlruns/`; ambos están en `.gitignore`. Al importarse, `mlflow` imprime un aviso para agentes que se silencia con `MLFLOW_DISABLE_AGENT_HINT=1`.
- Comandos, desde la raíz del repo: `python -m demand_forecast.cli ingest` (re-descarga los últimos 31 días; `--full` pide toda la historia, unos 13 minutos) y `python -m demand_forecast.cli backtest --model seasonal-naive`.
- Baselines en la ventana 2025-11-01 a 2026-10-31: ingenuo estacional MAPE 3,40 %, promedio de 4 semanas 3,03 %.
- Datos en `data/raw/<serie>/<año>.parquet` (tal como llega de XM, con días parciales) y `data/processed/<serie>/<año>.parquet` (horario, solo días completos, sin huecos). Series: `demand` y `generation`.
- En un backtest, toda lectura del pasado pasa por `models.backtest.available_history` (corte en `D − DATA_LAG_DAYS` para la demanda y en `D − GENERATION_LAG_DAYS` para `Gene`). Las features se definen respecto a ese corte, no con rezagos fijos.
- `uv sync` a secas desinstala el grupo `notebook`; usar `uv sync --group notebook`. `jupyter execute` en Windows necesita `PYTHONUTF8=1`.
- Un día está completo si su total es ≥ 60 % de la mediana de los 28 días completos anteriores (no 90 %: muchos domingos y festivos completos quedan entre 0,73 y 0,90). Para `DemaCome` se exige además que sea ≥ 95 % del total de `Gene` del mismo día (en 26 años nunca rechaza un día completo).
- Variable objetivo: `DemaCome` / `Sistema` (kWh por hora). Feature adelantada: `Gene` / `Sistema`.
- Cliente de datos: REST directo con `requests` (`POST https://servapibi.xm.com.co/hourly`), no `pydataxm`.
- Los días recientes llegan parciales sin ninguna marca. `DemaCome` está completa hasta ~D−5 y `Gene` hasta ~D−3, con tres mediciones en días hábiles (10, 11 y 13 de noviembre de 2026; la del 13 a las 11:02, las otras hacia las 22:00; no hay snapshot del 12). Los días ya completos se revisan después, hasta un 0,04 %. El usuario toma snapshots hasta el 19 de noviembre de 2026 con `explore/08_snapshot.py`; los del 14 al 17 (fin de semana y festivo) deciden si XM también publica en días no hábiles. Revisar `python -m uv run python explore/08_snapshot.py summary` antes de fijar `DATA_LAG_DAYS` y `GENERATION_LAG_DAYS` en la Fase 2.
- En esta máquina Windows, el almacén de certificados no tiene la raíz GlobalSign R46. `requests` funciona porque usa `certifi`; `aiohttp` y `ssl` por defecto fallan.
