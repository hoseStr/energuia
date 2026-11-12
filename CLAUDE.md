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

- Fase 0 terminada el 2026-11-10. Lo siguiente es la Fase 1 (ver los ajustes al inicio de esa sección en `PLAN.md`). El proyecto ya es un paquete instalable (`src/demand_forecast/`, backend `uv_build`); todavía no hay `tests/`.
- Un día está completo si su total es ≥ 60 % de la mediana de los 28 días completos anteriores (no 90 %: muchos domingos y festivos completos quedan entre 0,73 y 0,90). Para `DemaCome` se agrega, de forma provisional, que sea ≥ 95 % del total de `Gene` del mismo día.
- Variable objetivo: `DemaCome` / `Sistema` (kWh por hora). Feature adelantada: `Gene` / `Sistema`.
- Cliente de datos: REST directo con `requests` (`POST https://servapibi.xm.com.co/hourly`), no `pydataxm`.
- Los días recientes llegan parciales sin ninguna marca. `DemaCome` está completa hasta ~D−5 y `Gene` hasta ~D−3, con dos mediciones en días hábiles (10 y 11 de noviembre de 2026). Los días ya completos se revisan después, hasta un 0,04 %. El usuario toma snapshots hasta el 19 de noviembre de 2026 con `explore/08_snapshot.py`; los del 14 al 17 (fin de semana y festivo) deciden si XM también publica en días no hábiles. Revisar `python -m uv run python explore/08_snapshot.py summary` antes de fijar `DATA_LAG_DAYS` en la Fase 2.
- En esta máquina Windows, el almacén de certificados no tiene la raíz GlobalSign R46. `requests` funciona porque usa `certifi`; `aiohttp` y `ssl` por defecto fallan.
