from unittest.mock import patch

import pytest

from ml_service.model import Model, ModelLoadError


def test_model_set_loads_model_successfully(dummy_model):
    model_container = Model()

    with patch('ml_service.model.load_model', return_value=dummy_model) as mocked_load:
        model_container.set('run-123')

    state = model_container.get()

    mocked_load.assert_called_once_with(run_id='run-123')
    assert state.model is dummy_model
    assert state.run_id == 'run-123'
    assert state.error is None
    assert model_container.features == ['age', 'education.num', 'hours.per.week']


def test_model_set_raises_on_empty_run_id():
    model_container = Model()

    with pytest.raises(ModelLoadError, match='run_id must be a non-empty string'):
        model_container.set('   ')


def test_model_set_raises_when_loader_fails():
    model_container = Model()

    with patch('ml_service.model.load_model', side_effect=Exception('mlflow error')):
        with pytest.raises(ModelLoadError, match='Failed to load model from MLflow'):
            model_container.set('bad-run-id')


def test_loaded_model_can_run_inference(dummy_model):
    result = dummy_model.predict_proba(None)

    assert result.shape == (1, 2)
    assert float(result[0][1]) == 0.8