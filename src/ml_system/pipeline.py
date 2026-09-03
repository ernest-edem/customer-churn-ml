from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from ml_system.config.schemas import Settings
from ml_system.data import (
    load_dataset,
    split_dataset,
    validate_dataset,
)
from ml_system.evaluation import evaluate_model
from ml_system.exceptions.errors import (
    DataError,
    EvaluationError,
    MLSystemError,
    ModelError,
)
from ml_system.models import build_model
from ml_system.persistence import save_model
from ml_system.training import (
    build_training_pipeline,
    train_model,
)

from ml_system.reporting import save_metrics_report

logger = logging.getLogger("ml_system")

@dataclass
class TrainingResult:
    """Result produced by the complete ML training workflow."""

    model: Pipeline
    metrics: dict[str, Any]
    model_path: Path
    metrics_path: Path


def run_training_pipeline(
    settings: Settings,
) -> TrainingResult:
    """
    Execute the complete machine learning training workflow.

    The workflow is controlled by the supplied configuration and
    connects dataset loading, validation, splitting, preprocessing,
    feature selection, model creation, training, evaluation, and
    model persistence.

    Parameters
    ----------
    settings:
        Validated application settings.

    Returns
    -------
    TrainingResult
        Trained model, evaluation results, and persisted model path.

    Raises
    ------
    MLSystemError
        If any stage of the training workflow fails.
    """

    logger.info(
        "Starting ML training workflow: project=%s",
        settings.project.name,
    )

    try:
        dataframe = load_dataset(settings.data.path)

        validate_dataset(
            dataframe=dataframe,
            target_column=settings.data.target_column,
        )

        X_train, X_test, y_train, y_test = split_dataset(
            dataframe=dataframe,
            target_column=settings.data.target_column,
            settings=settings.split,
        )

        training_data = X_train.copy()
        training_data[settings.data.target_column] = y_train

        test_data = X_test.copy()
        test_data[settings.data.target_column] = y_test

        model = build_model(settings.model)

        training_pipeline = build_training_pipeline(
            dataframe=training_data,
            target_column=settings.data.target_column,
            preprocessing_settings=settings.preprocessing,
            feature_selection_settings=(
                settings.features.selection
            ),
            model=model,
        )

        trained_pipeline = train_model(
            pipeline=training_pipeline,
            training_data=training_data,
            target_column=settings.data.target_column,
        )

        metrics = evaluate_model(
            pipeline=trained_pipeline,
            test_data=test_data,
            target_column=settings.data.target_column,
            metrics=settings.evaluation.metrics,
        )

        model_path = Path(settings.persistence.model_path)

        save_model(
            model=trained_pipeline,
            path=model_path,
        )

        metrics_path = Path("reports/metrics/production_metrics.json")

        save_metrics_report(
        metrics=metrics,
        output_path=metrics_path,
    )

    except (
        DataError,
        ModelError,
        EvaluationError,
    ):
        logger.exception(
            "ML training workflow failed."
        )
        raise

    except MLSystemError:
        logger.exception(
            "ML training workflow failed."
        )
        raise

    except (OSError, ValueError, TypeError) as exc:
        logger.exception(
            "Unexpected error during ML training workflow."
        )
        raise MLSystemError(
            f"ML training workflow failed: {exc}"
        ) from exc

    logger.info(
        "ML training workflow completed successfully: model=%s",
        model_path,
    )

    return TrainingResult(
        model=trained_pipeline,
        metrics=metrics,
        model_path=model_path,
        metrics_path=metrics_path,
)