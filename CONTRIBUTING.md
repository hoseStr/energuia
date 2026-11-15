# Cómo contribuir

Es un proyecto personal de portafolio, pero los reportes de errores y las propuestas son bienvenidos.

## Reportar un problema

Abre un [issue](https://github.com/hoseStr/energuia/issues) con lo que esperabas, lo que pasó y, si aplica, el comando y su salida.
Los issues con la etiqueta `forecast-alert` los abre el job diario cuando el error reciente sube; no hace falta reportarlos.

## Proponer un cambio

1. Instala el entorno completo y los hooks:

   ```
   uv sync --group monitoring
   uv run pre-commit install
   ```

2. Haz el cambio con sus tests. Antes de abrir el pull request deben pasar:

   ```
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest
   ```

3. Abre el pull request contra `main`. CI corre lo mismo y construye la imagen de Docker.

## Reglas del proyecto

- **Sin fuga temporal:** nada se calcula con datos que no estaban publicados al emitir el pronóstico. Ante la duda, un test.
- **Las métricas no se retocan:** no se cambian el periodo de evaluación, las métricas ni los baselines para mejorar un número.
- **Los tests no usan la red:** XM siempre se simula.
- **La bitácora no se edita:** `data/predictions/` y `data/metrics/` los escribe solo el job diario.
- **Configuración en `config.py`**, sin valores sueltos por el código.
- **Commits pequeños** con prefijos de Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`) y el mensaje en español.
- Código, nombres y docstrings en inglés; documentos en español.
