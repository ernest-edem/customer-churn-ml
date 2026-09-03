import pandas as pd
import pytest

from ml_system.config.schemas import SplitSettings
from ml_system.data.splitter import split_dataset
from ml_system.exceptions.errors import DataError


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [21, 25, 31, 35, 42, 48, 53, 61, 67, 72],
            "MonthlyCharges": [
                20.0,
                25.0,
                31.0,
                35.0,
                42.0,
                48.0,
                53.0,
                61.0,
                67.0,
                72.0,
            ],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "Two year",
                "One year",
            ],
            "Churn": [
                "No",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
            ],
        }
    )


def test_split_dataset_returns_expected_shapes(sample_dataset):
    settings = SplitSettings(
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    X_train, X_test, y_train, y_test = split_dataset(
        sample_dataset,
        "Churn",
        settings,
    )

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_split_dataset_removes_target_from_features(sample_dataset):
    settings = SplitSettings(
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    X_train, X_test, _, _ = split_dataset(
        sample_dataset,
        "Churn",
        settings,
    )

    assert "Churn" not in X_train.columns
    assert "Churn" not in X_test.columns
    assert list(X_train.columns) == [
        "Age",
        "MonthlyCharges",
        "Contract",
    ]


def test_split_dataset_is_reproducible(sample_dataset):
    settings = SplitSettings(
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    first = split_dataset(
        sample_dataset,
        "Churn",
        settings,
    )

    second = split_dataset(
        sample_dataset,
        "Churn",
        settings,
    )

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_series_equal(first[2], second[2])
    pd.testing.assert_series_equal(first[3], second[3])


def test_split_dataset_preserves_stratification(sample_dataset):
    settings = SplitSettings(
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    _, _, y_train, y_test = split_dataset(
        sample_dataset,
        "Churn",
        settings,
    )

    assert y_train.value_counts().to_dict() == {
        "No": 5,
        "Yes": 3,
    }

    assert y_test.value_counts().to_dict() == {
        "No": 1,
        "Yes": 1,
    }


def test_split_dataset_without_stratification(sample_dataset):
    settings = SplitSettings(
        test_size=0.2,
        random_state=42,
        stratify=False,
    )

    X_train, X_test, y_train, y_test = split_dataset(
        sample_dataset,
        "Churn",
        settings,
    )

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_split_dataset_rejects_missing_target(sample_dataset):
    settings = SplitSettings()

    with pytest.raises(DataError, match="Target column not found"):
        split_dataset(
            sample_dataset,
            "MissingTarget",
            settings,
        )


def test_split_dataset_rejects_invalid_test_size(sample_dataset):
    settings = SplitSettings(
        test_size=1.0,
        random_state=42,
        stratify=True,
    )

    with pytest.raises(
        DataError,
        match="test_size must be between 0 and 1",
    ):
        split_dataset(
            sample_dataset,
            "Churn",
            settings,
        )


def test_split_dataset_rejects_missing_target_values(sample_dataset):
    dataset = sample_dataset.copy()
    dataset.loc[0, "Churn"] = None

    settings = SplitSettings(
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    with pytest.raises(
        DataError,
        match="Target column contains missing values",
    ):
        split_dataset(
            dataset,
            "Churn",
            settings,
        )


def test_split_dataset_rejects_single_class_target(sample_dataset):
    dataset = sample_dataset.copy()
    dataset["Churn"] = "No"

    settings = SplitSettings(
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    with pytest.raises(
        DataError,
        match="at least two classes",
    ):
        split_dataset(
            dataset,
            "Churn",
            settings,
        )