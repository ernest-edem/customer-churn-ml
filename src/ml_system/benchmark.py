from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml_system.config.loader import load_settings
from ml_system.config.schemas import ModelSettings
from ml_system.data import load_dataset, split_dataset
from ml_system.evaluation import evaluate_model
from ml_system.models import build_model
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
        Comparable model performance metrics.
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


if __name__ == "__main__":
    benchmark = benchmark_models("config/config.yaml")

    print("\nModel Benchmark Results")
    print("=" * 70)
    print(benchmark.to_string(index=False))