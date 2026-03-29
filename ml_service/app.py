from typing import Any
from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, HTTPException, Response

from ml_service import config
from ml_service.drift import DriftMonitor
from ml_service.features import to_dataframe
from ml_service.metrics.metrics import (
    PREDICT_REQUESTS_TOTAL,
    metrics_response,
    observe_inference_duration,
    observe_input_features,
    observe_missing_features,
    observe_model_update_error,
    observe_model_update_success,
    observe_predict_error,
    observe_prediction,
    observe_preprocessing_duration,
    set_active_model_info,
)
from ml_service.middleware import PrometheusMiddleware
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model, ModelLoadError
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)

logger = logging.getLogger(__name__)

MODEL = Model()
DRIFT_MONITOR = DriftMonitor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Tries to load the initial model from MLflow on startup.
    If loading fails, the service still starts and reports degraded state.
    Also starts Evidently drift monitoring background task.
    """
    configure_mlflow()
    run_id = config.default_run_id()

    try:
        MODEL.set(run_id=run_id)
        set_active_model_info(
            run_id=MODEL.get().run_id or run_id,
            model_type=MODEL.model_type or 'unknown',
            features=MODEL.features,
        )
        logger.info('Initial model loaded successfully: run_id=%s', run_id)
    except ModelLoadError as exc:
        logger.exception('Failed to load initial model: run_id=%s', run_id)
        MODEL.set_error(str(exc))

    DRIFT_MONITOR.start()

    try:
        yield
    finally:
        await DRIFT_MONITOR.stop()


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)
    app.add_middleware(PrometheusMiddleware)

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = MODEL.get()

        if model_state.model is None:
            return {
                'status': 'degraded',
                'run_id': model_state.run_id,
                'detail': model_state.error or 'Model is not loaded',
            }

        return {
            'status': 'ok',
            'run_id': model_state.run_id,
            'detail': None,
        }

    @app.get('/metrics')
    def metrics() -> Response:
        payload, content_type = metrics_response()
        return Response(content=payload, media_type=content_type)

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        PREDICT_REQUESTS_TOTAL.inc()

        model_state = MODEL.get()
        model = model_state.model

        if model is None:
            observe_predict_error('model_not_loaded')
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        preprocess_start = time.perf_counter()
        try:
            df = to_dataframe(request, needed_columns=MODEL.features)
            observe_preprocessing_duration(time.perf_counter() - preprocess_start)
        except ValueError as exc:
            observe_preprocessing_duration(time.perf_counter() - preprocess_start)

            message = str(exc)
            if 'Missing required features' in message:
                missing_part = message.split(':', maxsplit=1)[-1].strip()
                missing_part = missing_part.strip('[]')
                if missing_part:
                    missing_features = [
                        item.strip().strip("'").strip('"')
                        for item in missing_part.split(',')
                        if item.strip()
                    ]
                    observe_missing_features(missing_features)

            observe_predict_error('preprocessing_error')
            raise HTTPException(status_code=422, detail=message) from exc

        observe_input_features(df.iloc[0].to_dict())

        inference_start = time.perf_counter()
        try:
            probabilities = model.predict_proba(df)
            observe_inference_duration(time.perf_counter() - inference_start)

            probability = float(probabilities[0][1])
            prediction = int(probability >= 0.5)

            observe_prediction(probability=probability, prediction=prediction)
        except Exception as exc:
            observe_inference_duration(time.perf_counter() - inference_start)
            observe_predict_error('inference_error')
            logger.exception('Inference failed for run_id=%s', model_state.run_id)
            raise HTTPException(status_code=500, detail='Model inference failed') from exc

        DRIFT_MONITOR.add_event(
            features=df.iloc[0].to_dict(),
            prediction=prediction,
            probability=probability,
        )

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id

        try:
            MODEL.set(run_id=run_id)
            observe_model_update_success(
                run_id=MODEL.get().run_id or run_id,
                model_type=MODEL.model_type or 'unknown',
                features=MODEL.features,
            )
        except ModelLoadError as exc:
            observe_model_update_error()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()