import pandas as pd
import pytest

from ml_system.data import profile_dataset
from ml_system.exceptions.errors import DataError


def test_profile_dataset():
    dataframe = pd.DataFrame(
        {
            "age": [25, 35, 45, 35],
            "monthly_charges": [50.0, 75.0, None, 75.0],
            "contract": ["Month-to-month", "One year", "Two year", "One year"],
            "churn": [0, 1, 0, 1],
        }
    )

    profile = profile_dataset(dataframe, "churn")

    assert profile["rows"] == 4
    assert profile["columns"] == 4

    assert profile["column_names"] == [
        "age",
        "monthly_charges",
        "contract",
        "churn",
    ]

    assert profile["data_types"]["age"] == "int64"
    assert profile["data_types"]["contract"] in {"object", "str"}

    assert profile["missing_values"] == {
        "monthly_charges": 1,
    }

    assert profile["duplicate_rows"] == 1

    assert profile["numerical_columns"] == [
        "age",
        "monthly_charges",
        "churn",
    ]

    assert profile["categorical_columns"] == [
        "contract",
    ]

    assert profile["target_column"] == "churn"

    assert profile["target_distribution"] == {
        "0": 2,
        "1": 2,
    }


def test_profile_dataset_counts_duplicate_rows():
    dataframe = pd.DataFrame(
        {
            "age": [25, 35, 25],
            "churn": [0, 1, 0],
        }
    )

    profile = profile_dataset(dataframe, "churn")

    assert profile["duplicate_rows"] == 1


def test_profile_dataset_rejects_empty_dataset():
    dataframe = pd.DataFrame()

    with pytest.raises(DataError, match="Cannot profile an empty dataset"):
        profile_dataset(dataframe, "churn")


def test_profile_dataset_rejects_missing_target():
    dataframe = pd.DataFrame(
        {
            "age": [25, 35],
            "churn": [0, 1],
        }
    )

    with pytest.raises(DataError, match="Target column not found"):
        profile_dataset(dataframe, "customer_status")