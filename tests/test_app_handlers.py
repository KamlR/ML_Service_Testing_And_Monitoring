from fastapi.testclient import TestClient

from ml_service import app as app_module
from ml_service.model import ModelData, ModelLoadError


def test_predict_handler_returns_prediction(valid_payload, dummy_model):
    test_app = app_module.create_app()
    old_data = app_module.MODEL.data

    try:
        app_module.MODEL.data = ModelData(model=dummy_model, run_id='run-1', error=None)

        response = TestClient(test_app).post('/predict', json=valid_payload)

        assert response.status_code == 200
        body = response.json()
        assert body['prediction'] == 1
        assert abs(body['probability'] - 0.8) < 1e-9
    finally:
        app_module.MODEL.data = old_data


def test_predict_handler_returns_422_for_missing_feature(valid_payload, dummy_model):
    payload = dict(valid_payload)
    payload['age'] = None

    test_app = app_module.create_app()
    old_data = app_module.MODEL.data

    try:
        app_module.MODEL.data = ModelData(model=dummy_model, run_id='run-1', error=None)

        response = TestClient(test_app).post('/predict', json=payload)

        assert response.status_code == 422
        assert 'Missing required features' in response.json()['detail']
    finally:
        app_module.MODEL.data = old_data


def test_predict_handler_returns_503_when_model_not_loaded(valid_payload):
    test_app = app_module.create_app()
    old_data = app_module.MODEL.data

    try:
        app_module.MODEL.data = ModelData(model=None, run_id=None, error='not loaded')

        response = TestClient(test_app).post('/predict', json=valid_payload)

        assert response.status_code == 503
        assert response.json()['detail'] == 'Model is not loaded yet'
    finally:
        app_module.MODEL.data = old_data


def test_predict_handler_returns_request_validation_error():
    test_app = app_module.create_app()

    payload = {
        'age': 'not-an-int',
        'education.num': 13,
        'hours.per.week': 40,
    }

    response = TestClient(test_app).post('/predict', json=payload)

    assert response.status_code == 422


def test_update_model_returns_400_for_invalid_run_id(monkeypatch):
    test_app = app_module.create_app()

    def fake_set(run_id: str):
        raise ModelLoadError('bad run id')

    monkeypatch.setattr(app_module.MODEL, 'set', fake_set)

    response = TestClient(test_app).post('/updateModel', json={'run_id': 'bad-id'})

    assert response.status_code == 400
    assert response.json()['detail'] == 'bad run id'


def test_health_returns_degraded_when_model_is_not_loaded():
    test_app = app_module.create_app()
    old_data = app_module.MODEL.data

    try:
        app_module.MODEL.data = ModelData(model=None, run_id=None, error='startup failed')

        response = TestClient(test_app).get('/health')

        assert response.status_code == 200
        assert response.json()['status'] == 'degraded'
        assert response.json()['detail'] == 'startup failed'
    finally:
        app_module.MODEL.data = old_data