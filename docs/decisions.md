# Decision log

Cambios respecto a PLAN.md y decisiones tomadas por el camino. Las más recientes van primero.

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
