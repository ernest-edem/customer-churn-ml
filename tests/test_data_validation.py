import pandas as pd
import pytest

from ml_system.data import validate_dataset
from ml_system.exceptions.errors import DataError


def test_validate_dataset_success():
    dataframe = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "churn": [0, 1, 0],
        }
    )

    validate_dataset(dataframe, "churn")


def test_validate_dataset_rejects_empty_dataset():
    dataframe = pd.DataFrame()

    with pytest.raises(DataError, match="Dataset is empty"):
        validate_dataset(dataframe, "churn")


def test_validate_dataset_rejects_missing_target():
    dataframe = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "churn": [0, 1, 0],
        }
    )

    with pytest.raises(DataError, match="Target column not found"):
        validate_dataset(dataframe, "customer_status")


def test_validate_dataset_rejects_empty_target_name():
    dataframe = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "churn": [0, 1, 0],
        }
    )

    with pytest.raises(DataError, match="Target column cannot be empty"):
        validate_dataset(dataframe, "   ")


def test_validate_dataset_rejects_all_missing_target():
    dataframe = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "churn": [None, None, None],
        }
    )

    with pytest.raises(
        DataError,
        match="Target column contains no valid observations",
    ):
        validate_dataset(dataframe, "churn")