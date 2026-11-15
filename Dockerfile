# syntax=docker/dockerfile:1

# --- Build: resolve the locked runtime dependencies into a virtual environment ---
FROM ghcr.io/astral-sh/uv:0.13.7-python3.11-trixie-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first: this layer is rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# --- Runtime: the environment, the data (forecast log and errors included) and the models ---
FROM python:3.11-slim-trixie

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY data ./data
COPY models ./models

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

# The port comes from PORT when the platform sets it; 8000 otherwise.
CMD ["python", "-m", "demand_forecast.cli", "serve", "--host", "0.0.0.0"]
