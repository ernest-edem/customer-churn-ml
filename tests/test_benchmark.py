from pathlib import Path

import pandas as pd
import pytest

from ml_system.benchmark import (
    MODEL_CONFIGURATIONS,
    benchmark_models,
    cross_validate_models,
)

CONFIG_PATH = Path("config/config.yaml")


def test_benchmark_models_returns_dataframe():
    result = benchmark_models(CONFIG_PATH)

    assert isinstance(result, pd.DataFrame)


def test_cross_validate_models_returns_dataframe():
    result = cross_validate_models(CONFIG_PATH)

    assert isinstance(result, pd.DataFrame)


def test_cross_validate_models_contains_all_models():
    result = cross_validate_models(CONFIG_PATH)

    expected_models = {
        model_settings.name
        for model_settings in MODEL_CONFIGURATIONS
    }

    assert set(result["model"]) == expected_models


def test_cross_validate_models_contains_expected_columns():
    result = cross_validate_models(CONFIG_PATH)

    expected_columns = {
        "model",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
    }

    assert set(result.columns) == expected_columns


def test_cross_validate_models_default_five_folds():
    result = cross_validate_models(CONFIG_PATH)

    assert len(result) == len(MODEL_CONFIGURATIONS)

    for column in result.columns:
        if column != "model":
            assert pd.api.types.is_numeric_dtype(result[column])


def test_cross_validate_models_rejects_invalid_n_splits():
    with pytest.raises(
        ValueError,
        match="n_splits must be at least 2",
    ):
        cross_validate_models(
            config_path=CONFIG_PATH,
            n_splits=1,
        )


def test_cross_validate_models_returns_valid_metric_values():
    result = cross_validate_models(CONFIG_PATH)

    metric_columns = [
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]

    for column in metric_columns:
        assert result[column].notna().all()

    mean_columns = [
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
    ]

    for column in mean_columns:
        assert result[column].between(0.0, 1.0).all()

    std_columns = [
        "accuracy_std",
        "precision_std",
        "recall_std",
        "f1_std",
        "roc_auc_std",
    ]

    for column in std_columns:
        assert (result[column] >= 0.0).all()


def test_cross_validate_models_sorted_by_roc_auc():
    result = cross_validate_models(CONFIG_PATH)

    roc_auc_values = result["roc_auc_mean"].tolist()

    assert roc_auc_values == sorted(
        roc_auc_values,
        reverse=True,
    )