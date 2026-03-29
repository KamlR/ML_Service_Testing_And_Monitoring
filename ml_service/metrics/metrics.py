import os
import time
from typing import Any

import psutil
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


# Гистограммные бакеты подобраны так, чтобы потом в Grafana/Prometheus
# можно было считать histogram_quantile(0.75/0.90/0.95/0.99/0.999, ...)
# для latency-метрик.
LATENCY_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    10.0,
)

NUMERIC_VALUE_BUCKETS = (
    0,
    1,
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    80,
    100,
    500,
    1_000,
    10_000,
    100_000,
    1_000_000,
)

PROCESS = psutil.Process(os.getpid())
SERVICE_START_TIME = time.time()


HTTP_REQUESTS_TOTAL = Counter(
    'ml_service_http_requests_total',
    'Total number of HTTP requests',
    ['method', 'path', 'status_code'],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    'ml_service_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'path'],
    buckets=LATENCY_BUCKETS,
)

HTTP_REQUEST_EXCEPTIONS_TOTAL = Counter(
    'ml_service_http_request_exceptions_total',
    'Total number of unhandled HTTP exceptions',
    ['method', 'path', 'exception_type'],
)

SERVICE_UPTIME_SECONDS = Gauge(
    'ml_service_uptime_seconds',
    'Service uptime in seconds',
)

PROCESS_CPU_PERCENT = Gauge(
    'ml_service_process_cpu_percent',
    'Current process CPU usage percent',
)

PROCESS_MEMORY_RSS_BYTES = Gauge(
    'ml_service_process_memory_rss_bytes',
    'Current process RSS memory in bytes',
)

PROCESS_OPEN_FDS = Gauge(
    'ml_service_process_open_fds',
    'Current number of open file descriptors',
)

PREDICT_REQUESTS_TOTAL = Counter(
    'ml_service_predict_requests_total',
    'Total number of predict requests',
)

PREDICT_ERRORS_TOTAL = Counter(
    'ml_service_predict_errors_total',
    'Total number of predict errors',
    ['error_type'],
)

PREPROCESSING_DURATION_SECONDS = Histogram(
    'ml_service_preprocessing_duration_seconds',
    'Time spent on request preprocessing in seconds',
    buckets=LATENCY_BUCKETS,
)

MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    'ml_service_model_inference_duration_seconds',
    'Time spent on model inference in seconds',
    buckets=LATENCY_BUCKETS,
)

MODEL_PREDICTION_PROBABILITY = Histogram(
    'ml_service_model_prediction_probability',
    'Distribution of model positive-class probabilities',
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

MODEL_PREDICTIONS_TOTAL = Counter(
    'ml_service_model_predictions_total',
    'Total number of model predictions by class',
    ['prediction'],
)

FEATURE_MISSING_TOTAL = Counter(
    'ml_service_feature_missing_total',
    'Number of requests with missing required feature',
    ['feature'],
)

FEATURE_NUMERIC_VALUE = Histogram(
    'ml_service_feature_numeric_value',
    'Distribution of numeric feature values',
    ['feature'],
    buckets=NUMERIC_VALUE_BUCKETS,
)

FEATURE_CATEGORICAL_VALUE_TOTAL = Counter(
    'ml_service_feature_categorical_value_total',
    'Number of observed categorical feature values',
    ['feature', 'value'],
)

MODEL_UPDATES_TOTAL = Counter(
    'ml_service_model_updates_total',
    'Total number of successful model updates',
)

MODEL_UPDATE_ERRORS_TOTAL = Counter(
    'ml_service_model_update_errors_total',
    'Total number of failed model update attempts',
)

MODEL_LAST_UPDATE_TIMESTAMP = Gauge(
    'ml_service_model_last_update_timestamp_seconds',
    'Unix timestamp of the last successful model update',
)

ACTIVE_MODEL_INFO = Gauge(
    'ml_service_active_model_info',
    'Currently active model information',
    ['run_id', 'model_type'],
)

ACTIVE_MODEL_FEATURE_INFO = Gauge(
    'ml_service_active_model_feature_info',
    'Features required by the currently active model',
    ['run_id', 'feature'],
)


NUMERIC_FEATURES = {
    'age',
    'fnlwgt',
    'education.num',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
}

CATEGORICAL_FEATURES = {
    'workclass',
    'education',
    'marital.status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native.country',
}


def metrics_response() -> tuple[bytes, str]:
    update_runtime_metrics()
    payload = generate_latest()
    return payload, CONTENT_TYPE_LATEST


def update_runtime_metrics() -> None:
    SERVICE_UPTIME_SECONDS.set(time.time() - SERVICE_START_TIME)

    try:
        PROCESS_CPU_PERCENT.set(PROCESS.cpu_percent(interval=None))
    except Exception:
        pass

    try:
        PROCESS_MEMORY_RSS_BYTES.set(PROCESS.memory_info().rss)
    except Exception:
        pass

    try:
        PROCESS_OPEN_FDS.set(PROCESS.num_fds())
    except Exception:
        pass


def observe_preprocessing_duration(duration_seconds: float) -> None:
    PREPROCESSING_DURATION_SECONDS.observe(duration_seconds)


def observe_inference_duration(duration_seconds: float) -> None:
    MODEL_INFERENCE_DURATION_SECONDS.observe(duration_seconds)


def observe_prediction(probability: float, prediction: int) -> None:
    MODEL_PREDICTION_PROBABILITY.observe(probability)
    MODEL_PREDICTIONS_TOTAL.labels(prediction=str(prediction)).inc()


def observe_input_features(row: dict[str, Any]) -> None:
    for feature_name, value in row.items():
        if value is None:
            continue

        if feature_name in NUMERIC_FEATURES:
            try:
                FEATURE_NUMERIC_VALUE.labels(feature=feature_name).observe(float(value))
            except (TypeError, ValueError):
                continue
            continue

        if feature_name in CATEGORICAL_FEATURES:
            FEATURE_CATEGORICAL_VALUE_TOTAL.labels(
                feature=feature_name,
                value=str(value),
            ).inc()


def observe_missing_features(features: list[str]) -> None:
    for feature_name in features:
        FEATURE_MISSING_TOTAL.labels(feature=feature_name).inc()


def observe_predict_error(error_type: str) -> None:
    PREDICT_ERRORS_TOTAL.labels(error_type=error_type).inc()


def observe_model_update_success(run_id: str, model_type: str, features: list[str]) -> None:
    MODEL_UPDATES_TOTAL.inc()
    MODEL_LAST_UPDATE_TIMESTAMP.set(time.time())
    set_active_model_info(run_id=run_id, model_type=model_type, features=features)


def observe_model_update_error() -> None:
    MODEL_UPDATE_ERRORS_TOTAL.inc()


def set_active_model_info(run_id: str, model_type: str, features: list[str]) -> None:
    ACTIVE_MODEL_INFO.clear()
    ACTIVE_MODEL_FEATURE_INFO.clear()

    ACTIVE_MODEL_INFO.labels(run_id=run_id, model_type=model_type).set(1)

    for feature_name in features:
        ACTIVE_MODEL_FEATURE_INFO.labels(run_id=run_id, feature=feature_name).set(1)