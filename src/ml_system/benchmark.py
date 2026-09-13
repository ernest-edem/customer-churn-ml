from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from ml_system.config.loader import load_settings
from ml_system.config.schemas import ModelSettings
from ml_system.data import load_dataset, split_dataset
from ml_system.evaluation import evaluate_model
from ml_system.models import build_model
from ml_system.reporting import (
    save_benchmark_report,
    save_cross_validation_report,
)
from ml_system.training import build_training_pipeline, train_model


logger = logging.getLogger("ml_system")


@dataclass(frozen=True)
class BenchmarkResult:
    """Metrics produced for one benchmarked model."""

    model_name: str
    metrics: dict[str, Any]


MODEL_CONFIGURATIONS = (
    ModelSettings(
        name="logistic_regression",
        parameters={
            "max_iter": 1000,
            "random_state": 42,
        },
    ),
    ModelSettings(
        name="decision_tree",
        parameters={
            "random_state": 42,
        },
    ),
    ModelSettings(
        name="random_forest",
        parameters={
            "n_estimators": 200,
            "random_state": 42,
        },
    ),
    ModelSettings(
        name="gradient_boosting",
        parameters={
            "random_state": 42,
        },
    ),
)


def benchmark_models(
    config_path: str | Path,
) -> pd.DataFrame:
    """
    Train and evaluate all required classification models.

    Returns
    -------
    pandas.DataFrame
        Comparable model performance metrics from the holdout test set.
    """
    settings = load_settings(config_path)

    dataframe = load_dataset(settings.data.path)

    X_train, X_test, y_train, y_test = split_dataset(
        dataframe=dataframe,
        target_column=settings.data.target_column,
        settings=settings.split,
    )

    training_data = X_train.copy()
    training_data[settings.data.target_column] = y_train

    test_data = X_test.copy()
    test_data[settings.data.target_column] = y_test

    results: list[BenchmarkResult] = []

    for model_settings in MODEL_CONFIGURATIONS:
        logger.info(
            "Benchmarking model: %s",
            model_settings.name,
        )

        model = build_model(model_settings)

        pipeline = build_training_pipeline(
            dataframe=training_data,
            target_column=settings.data.target_column,
            preprocessing_settings=settings.preprocessing,
            feature_selection_settings=settings.features.selection,
            model=model,
        )

        trained_pipeline = train_model(
            pipeline=pipeline,
            training_data=training_data,
            target_column=settings.data.target_column,
        )

        metrics = evaluate_model(
            pipeline=trained_pipeline,
            test_data=test_data,
            target_column=settings.data.target_column,
            metrics=settings.evaluation.metrics,
        )

        results.append(
            BenchmarkResult(
                model_name=model_settings.name,
                metrics=metrics,
            )
        )

    benchmark_dataframe = pd.DataFrame(
        [
            {
                "model": result.model_name,
                "accuracy": result.metrics["accuracy"],
                "precision": result.metrics["precision"],
                "recall": result.metrics["recall"],
                "f1": result.metrics["f1"],
                "roc_auc": result.metrics["roc_auc"],
            }
            for result in results
        ]
    )

    return benchmark_dataframe.sort_values(
        by="roc_auc",
        ascending=False,
    ).reset_index(drop=True)


def cross_validate_models(
    config_path: str | Path,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Perform stratified k-fold cross-validation for all required models.

    Cross-validation is performed only on the training portion of the
    original holdout split. The final holdout test set therefore remains
    untouched and available for final model evaluation.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    n_splits:
        Number of stratified cross-validation folds.

    Returns
    -------
    pandas.DataFrame
        Mean and standard deviation for each evaluation metric.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    settings = load_settings(config_path)

    dataframe = load_dataset(settings.data.path)

    X_train, _, y_train, _ = split_dataset(
        dataframe=dataframe,
        target_column=settings.data.target_column,
        settings=settings.split,
    )

    training_data = X_train.copy()
    training_data[settings.data.target_column] = y_train

    cross_validator = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=settings.split.random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc",
    }

    results: list[dict[str, Any]] = []

    for model_settings in MODEL_CONFIGURATIONS:
        logger.info(
            "Cross-validating model: %s using %d folds",
            model_settings.name,
            n_splits,
        )

        model = build_model(model_settings)

        pipeline = build_training_pipeline(
            dataframe=training_data,
            target_column=settings.data.target_column,
            preprocessing_settings=settings.preprocessing,
            feature_selection_settings=settings.features.selection,
            model=model,
        )

        # cross_validate clones the pipeline and fits each clone separately
        # within each fold. This keeps preprocessing and model fitting inside
        # the cross-validation process.
        cv_results = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=cross_validator,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
        )

        results.append(
            {
                "model": model_settings.name,
                "accuracy_mean": cv_results["test_accuracy"].mean(),
                "accuracy_std": cv_results["test_accuracy"].std(),
                "precision_mean": cv_results["test_precision"].mean(),
                "precision_std": cv_results["test_precision"].std(),
                "recall_mean": cv_results["test_recall"].mean(),
                "recall_std": cv_results["test_recall"].std(),
                "f1_mean": cv_results["test_f1"].mean(),
                "f1_std": cv_results["test_f1"].std(),
                "roc_auc_mean": cv_results["test_roc_auc"].mean(),
                "roc_auc_std": cv_results["test_roc_auc"].std(),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            by="roc_auc_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    config_path = "config/config.yaml"

    benchmark = benchmark_models(config_path)

    benchmark_report_path = Path(
        "reports/metrics/model_benchmark.csv"
    )

    save_benchmark_report(
        benchmark=benchmark,
        output_path=benchmark_report_path,
    )

    print("\nModel Benchmark Results")
    print("=" * 70)
    print(benchmark.to_string(index=False))

    cross_validation = cross_validate_models(
        config_path=config_path,
        n_splits=5,
    )

    cross_validation_report_path = Path(
        "reports/metrics/cross_validation.csv"
    )

    save_cross_validation_report(
        cross_validation=cross_validation,
        output_path=cross_validation_report_path,
    )

    print("\n5-Fold Stratified Cross-Validation Results")
    print("=" * 70)
    print(cross_validation.to_string(index=False))