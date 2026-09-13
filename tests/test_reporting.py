from pathlib import Path

import pandas as pd
import pytest

from ml_system.exceptions.errors import MLSystemError
from ml_system.reporting import (
    save_benchmark_report,
    save_cross_validation_report,
    save_metrics_report,
)


def test_save_metrics_report_creates_json_file(tmp_path: Path):
    metrics = {
        "accuracy": 0.79,
        "precision": 0.78,
        "recall": 0.79,
        "f1": 0.78,
        "roc_auc": 0.83,
    }

    output_path = tmp_path / "metrics.json"

    result = save_metrics_report(
        metrics=metrics,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()

    saved_metrics = pd.read_json(
        output_path,
        typ="series",
    ).to_dict()

    assert saved_metrics.keys() == metrics.keys()

    for key, value in metrics.items():
        assert saved_metrics[key] == pytest.approx(value)


def test_save_metrics_report_rejects_non_dict():
    with pytest.raises(
        MLSystemError,
        match="Metrics must be provided as a dictionary.",
    ):
        save_metrics_report(
            metrics="invalid",
            output_path="reports/metrics/test.json",
        )


def test_save_benchmark_report_creates_csv_file(tmp_path: Path):
    benchmark = pd.DataFrame(
        {
            "model": [
                "gradient_boosting",
                "logistic_regression",
            ],
            "accuracy": [
                0.79,
                0.78,
            ],
            "roc_auc": [
                0.83,
                0.82,
            ],
        }
    )

    output_path = tmp_path / "benchmark.csv"

    result = save_benchmark_report(
        benchmark=benchmark,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()

    saved_benchmark = pd.read_csv(output_path)

    pd.testing.assert_frame_equal(
        saved_benchmark,
        benchmark,
    )


def test_save_benchmark_report_rejects_non_dataframe():
    with pytest.raises(
        MLSystemError,
        match="Benchmark results must be a pandas DataFrame.",
    ):
        save_benchmark_report(
            benchmark="invalid",
            output_path="reports/metrics/test.csv",
        )


def test_save_benchmark_report_rejects_empty_dataframe():
    with pytest.raises(
        MLSystemError,
        match="Benchmark results cannot be empty.",
    ):
        save_benchmark_report(
            benchmark=pd.DataFrame(),
            output_path="reports/metrics/test.csv",
        )


def test_save_cross_validation_report_creates_csv_file(
    tmp_path: Path,
):
    cross_validation = pd.DataFrame(
        {
            "model": [
                "gradient_boosting",
                "logistic_regression",
            ],
            "accuracy_mean": [
                0.80,
                0.79,
            ],
            "accuracy_std": [
                0.006,
                0.005,
            ],
            "roc_auc_mean": [
                0.84,
                0.83,
            ],
            "roc_auc_std": [
                0.006,
                0.005,
            ],
        }
    )

    output_path = tmp_path / "cross_validation.csv"

    result = save_cross_validation_report(
        cross_validation=cross_validation,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()

    saved_cross_validation = pd.read_csv(output_path)

    pd.testing.assert_frame_equal(
        saved_cross_validation,
        cross_validation,
    )


def test_save_cross_validation_report_rejects_non_dataframe():
    with pytest.raises(
        MLSystemError,
        match="Cross-validation results must be a pandas DataFrame.",
    ):
        save_cross_validation_report(
            cross_validation="invalid",
            output_path="reports/metrics/test.csv",
        )


def test_save_cross_validation_report_rejects_empty_dataframe():
    with pytest.raises(
        MLSystemError,
        match="Cross-validation results cannot be empty.",
    ):
        save_cross_validation_report(
            cross_validation=pd.DataFrame(),
            output_path="reports/metrics/test.csv",
        )