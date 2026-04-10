import time
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow


def setup_mlflow(experiment_name: str = "air-paradis") -> None:
    """
    Configure MLflow tracking (SQLite backend + local artifacts).
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)


def track_run(
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Optional[Dict[str, str]] = None,
) -> str:
    """
    Fonction standardisée de tracking MLflow :
    - params
    - metrics
    - artifacts (ex: ROC curve, loss curve)
    """
    artifacts = artifacts or {}

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        for artifact_name, file_path in artifacts.items():
            if file_path and Path(file_path).exists():
                mlflow.log_artifact(file_path, artifact_path=artifact_name)

        return run.info.run_id


class Timer:
    """
    Helper pour mesurer le temps d'entraînement / prédiction.
    """
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.dt = time.perf_counter() - self.t0
