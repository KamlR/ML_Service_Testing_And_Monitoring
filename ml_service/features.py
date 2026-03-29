import pandas as pd

from ml_service.schemas import PredictRequest


FEATURE_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education.num',
    'marital.status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
    'native.country',
]


def to_dataframe(req: PredictRequest, needed_columns: list[str] = None) -> pd.DataFrame:
    columns = [
        column for column in needed_columns if column in FEATURE_COLUMNS
    ] if needed_columns is not None else FEATURE_COLUMNS

    if not columns:
        raise ValueError('No valid feature columns were provided for the active model')

    unsupported_columns = []
    if needed_columns is not None:
        unsupported_columns = [column for column in needed_columns if column not in FEATURE_COLUMNS]

    if unsupported_columns:
        raise ValueError(
            f'Active model expects unsupported feature columns: {unsupported_columns}'
        )

    missing_columns = [
        column
        for column in columns
        if getattr(req, column.replace('.', '_'), None) is None
    ]

    if missing_columns:
        raise ValueError(
            f'Missing required features for active model: {missing_columns}'
        )

    row = [getattr(req, column.replace('.', '_')) for column in columns]
    return pd.DataFrame([row], columns=columns)