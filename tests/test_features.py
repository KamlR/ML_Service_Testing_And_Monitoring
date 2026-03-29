import pytest

from ml_service.features import to_dataframe
from ml_service.schemas import PredictRequest


def make_request(payload: dict) -> PredictRequest:
    return PredictRequest(**payload)


def test_to_dataframe_uses_only_needed_columns(valid_payload):
    req = make_request(valid_payload)

    df = to_dataframe(req, needed_columns=['age', 'education.num', 'hours.per.week'])

    assert list(df.columns) == ['age', 'education.num', 'hours.per.week']
    assert df.iloc[0]['age'] == 40
    assert df.iloc[0]['education.num'] == 13
    assert df.iloc[0]['hours.per.week'] == 40


def test_to_dataframe_raises_on_missing_required_feature(valid_payload):
    payload = dict(valid_payload)
    payload['age'] = None
    req = make_request(payload)

    with pytest.raises(ValueError, match='Missing required features'):
        to_dataframe(req, needed_columns=['age', 'education.num'])


def test_to_dataframe_raises_on_unsupported_feature(valid_payload):
    req = make_request(valid_payload)

    with pytest.raises(ValueError, match='unsupported feature columns'):
        to_dataframe(req, needed_columns=['age', 'unknown_feature'])


def test_to_dataframe_returns_all_default_feature_columns(valid_payload):
    req = make_request(valid_payload)

    df = to_dataframe(req)

    assert 'age' in df.columns
    assert 'education.num' in df.columns
    assert 'hours.per.week' in df.columns
    assert len(df.columns) == 14