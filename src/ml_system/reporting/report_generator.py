from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ml_system.exceptions.errors import MLSystemError


logger = logging.getLogger("ml_system")


def save_metrics_report(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save model evaluation metrics as a JSON report.
    """
    if not isinstance(metrics, dict):
        raise MLSystemError(
            "Metrics must be provided as a dictionary."
        )

    path = Path(output_path)

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                metrics,
                file,
                indent=4,
            )

    except (OSError, TypeError, ValueError) as exc:
        raise MLSystemError(
            f"Failed to save metrics report: {path}"
        ) from exc

    logger.info(
        "Metrics report saved successfully: %s",
        path,
    )

    return path


def save_benchmark_report(
    benchmark: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save model benchmark results as a CSV report.
    """
    if not isinstance(benchmark, pd.DataFrame):
        raise MLSystemError(
            "Benchmark results must be a pandas DataFrame."
        )

    if benchmark.empty:
        raise MLSystemError(
            "Benchmark results cannot be empty."
        )

    path = Path(output_path)

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        benchmark.to_csv(
            path,
            index=False,
        )

    except (OSError, ValueError) as exc:
        raise MLSystemError(
            f"Failed to save benchmark report: {path}"
        ) from exc

    logger.info(
        "Benchmark report saved successfully: %s",
        path,
    )

    return path


def save_cross_validation_report(
    cross_validation: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save cross-validation results as a CSV report.
    """
    if not isinstance(cross_validation, pd.DataFrame):
        raise MLSystemError(
            "Cross-validation results must be a pandas DataFrame."
        )

    if cross_validation.empty:
        raise MLSystemError(
            "Cross-validation results cannot be empty."
        )

    path = Path(output_path)

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cross_validation.to_csv(
            path,
            index=False,
        )

    except (OSError, ValueError) as exc:
        raise MLSystemError(
            f"Failed to save cross-validation report: {path}"
        ) from exc

    logger.info(
        "Cross-validation report saved successfully: %s",
        path,
    )

    return path