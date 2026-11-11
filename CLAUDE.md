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
