# Monitoreo

- El panel está en español y en inglés (`?lang=es` y `?lang=en`): los textos de ambos idiomas van como JSON en `monitoring/panel.html`, y todo texto nuevo debe agregarse en los dos (hay un test).
- No hay informe de drift con menos de 7 días en la bitácora (`DRIFT_MIN_DAYS`): las ventanas más cortas dan falsas alarmas (`explore/12_short_drift_windows.py`, unos 12 minutos).
