import numpy as np
import pytest

from ml_service.model import ModelData


class DummyModel:
    def __init__(self, probability: float = 0.8, feature_names=None):
        self._probability = probability
        self.feature_names_in_ = np.array(
            feature_names or ['age', 'education.num', 'hours.per.week'],
            dtype=object,
        )

    def predict_proba(self, df):
        return np.array([[1 - self._probability, self._probability]])


@pytest.fixture
def valid_payload() -> dict:
    return {
        'age': 40,
        'workclass': 'Private',
        'fnlwgt': 100000,
        'education': 'Bachelors',
        'education.num': 13,
        'marital.status': 'Never-married',
        'occupation': 'Tech-support',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital.gain': 0,
        'capital.loss': 0,
        'hours.per.week': 40,
        'native.country': 'United-States',
    }


@pytest.fixture
def dummy_model() -> DummyModel:
    return DummyModel()


@pytest.fixture
def loaded_model_data(dummy_model) -> ModelData:
    return ModelData(model=dummy_model, run_id='test-run-id', error=None)