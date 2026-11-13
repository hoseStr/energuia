"""Experiment tracking with a local MLflow store. Only used while developing and retraining."""

from collections.abc import Mapping

from demand_forecast import config


def log_run(
    name: str,
    *,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    tags: Mapping[str, object],
) -> None:
    """Record one run: what was tried, on which data, and how it scored."""
    # Development dependency: it is not installed where the model is only served.
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({key: str(value) for key, value in tags.items()})
        mlflow.log_params(dict(params))
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
