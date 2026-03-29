import threading
from typing import NamedTuple

from sklearn.pipeline import Pipeline

from ml_service.mlflow_utils import load_model


class ModelLoadError(Exception):
    """Raised when a model cannot be loaded or activated."""


class ModelData(NamedTuple):
    model: Pipeline | None
    run_id: str | None
    error: str | None


class Model:
    """
    Thread-safe container for the currently active model.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.data = ModelData(model=None, run_id=None, error=None)

    def get(self) -> ModelData:
        with self.lock:
            return self.data

    def set(self, run_id: str) -> None:
        if not run_id or not run_id.strip():
            raise ModelLoadError('run_id must be a non-empty string')

        normalized_run_id = run_id.strip()

        try:
            model = load_model(run_id=normalized_run_id)
        except Exception as exc:
            raise ModelLoadError(
                f'Failed to load model from MLflow for run_id={normalized_run_id}: {exc}'
            ) from exc

        with self.lock:
            self.data = ModelData(model=model, run_id=normalized_run_id, error=None)

    def set_error(self, error_message: str) -> None:
        with self.lock:
            self.data = ModelData(
                model=self.data.model,
                run_id=self.data.run_id,
                error=error_message,
            )

    @property
    def features(self) -> list[str]:
        with self.lock:
            if self.data.model is None:
                return []

            if not hasattr(self.data.model, 'feature_names_in_'):
                raise ModelLoadError('Loaded model does not expose feature_names_in_')

            return list(self.data.model.feature_names_in_)

    @property
    def model_type(self) -> str | None:
        with self.lock:
            if self.data.model is None:
                return None

            model = self.data.model

            if hasattr(model, 'steps') and model.steps:
                return model.steps[-1][1].__class__.__name__

            return model.__class__.__name__