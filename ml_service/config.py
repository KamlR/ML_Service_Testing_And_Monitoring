import os

MODEL_ARTIFACT_PATH = 'model'


def tracking_uri() -> str:
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not tracking_uri:
        raise RuntimeError('Please set MLFLOW_TRACKING_URI')
    return tracking_uri


def default_run_id() -> str:
    """
    Returns model URI for startup.
    """

    default_run_id = os.getenv('DEFAULT_RUN_ID')
    if not default_run_id:
        raise RuntimeError('Set DEFAULT_RUN_ID to load model on startup')
    return default_run_id


def evidently_url() -> str:
    return os.getenv('EVIDENTLY_URL', 'http://158.160.2.37:8000/')


def evidently_project_id() -> str | None:
    return os.getenv('EVIDENTLY_PROJECT_ID')


def drift_batch_size() -> int:
    return int(os.getenv('DRIFT_BATCH_SIZE', '50'))


def drift_check_interval_seconds() -> int:
    return int(os.getenv('DRIFT_CHECK_INTERVAL_SECONDS', '60'))