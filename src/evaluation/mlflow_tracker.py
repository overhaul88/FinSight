"""Optional MLflow tracking for FinSight."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import socket
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from src.config import settings


class FinSightTracker:
    """MLflow wrapper that degrades gracefully when MLflow is unavailable."""

    def __init__(self) -> None:
        self._mlflow = self._load_mlflow()
        if self._mlflow is not None:
            self._configure_mlflow()

    @staticmethod
    def _load_mlflow():
        try:
            import mlflow  # type: ignore
        except ImportError:
            return None
        return mlflow

    def _configure_mlflow(self) -> None:
        if self._mlflow is None:
            return

        tracking_uri = settings.mlflow_tracking_uri
        if not self._tracking_uri_reachable(tracking_uri):
            self._mlflow = None
            return

        try:
            self._mlflow.set_tracking_uri(tracking_uri)
            self._mlflow.set_experiment(settings.mlflow_experiment_name)
        except Exception:
            self._mlflow = None

    @staticmethod
    def _tracking_uri_reachable(tracking_uri: str) -> bool:
        parsed = urlparse(tracking_uri)
        if parsed.scheme not in {"http", "https"}:
            return True

        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    @contextmanager
    def ragas_run(self, run_name: str | None = None, tags: Optional[Dict[str, Any]] = None):
        if self._mlflow is None:
            yield None
            return

        run_name = run_name or f"ragas-eval-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        with self._mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run

    def log_ragas_scores(self, scores: Dict[str, float], params: Optional[Dict[str, Any]] = None) -> None:
        if self._mlflow is None:
            return
        self._mlflow.log_metrics(scores)
        if params:
            self._mlflow.log_params({key: str(value) for key, value in params.items()})

        thresholds = {
            "faithfulness": 0.80,
            "answer_relevancy": 0.75,
            "context_precision": 0.70,
            "context_recall": 0.65,
        }
        for metric, threshold in thresholds.items():
            if metric in scores:
                self._mlflow.log_metric(f"{metric}_threshold_passed", int(scores[metric] >= threshold))

    def log_ingestion_metrics(self, metrics: Dict[str, Any]) -> None:
        if self._mlflow is None:
            return
        numeric = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        other = {key: str(value) for key, value in metrics.items() if not isinstance(value, (int, float))}
        if numeric:
            self._mlflow.log_metrics(numeric)
        if other:
            self._mlflow.log_params(other)
