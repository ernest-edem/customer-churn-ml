from __future__ import annotations

import logging

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from ml_system.config.schemas import FeatureSelectionSettings
from ml_system.exceptions.errors import DataError


logger = logging.getLogger("ml_system")


SUPPORTED_METHODS = {
    "mutual_information",
}


def build_feature_selector(
    settings: FeatureSelectionSettings,
    feature_count: int,
) -> SelectKBest | None:
    """
    Build a configurable feature-selection transformer.

    Parameters
    ----------
    settings:
        Feature-selection configuration.
    feature_count:
        Number of features available after preprocessing.

    Returns
    -------
    SelectKBest | None
        Configured feature selector, or None when feature selection
        is disabled.

    Raises
    ------
    DataError
        If the feature-selection configuration is invalid.
    """

    if feature_count <= 0:
        raise DataError(
            "Feature count must be greater than zero."
        )

    if not settings.enabled:
        logger.info("Feature selection disabled.")

        return None

    method = settings.method.strip().lower()

    if method not in SUPPORTED_METHODS:
        supported = ", ".join(sorted(SUPPORTED_METHODS))

        raise DataError(
            f"Unsupported feature selection method '{settings.method}'. "
            f"Supported methods: {supported}"
        )

    if settings.top_k <= 0:
        raise DataError(
            "Feature selection top_k must be greater than zero."
        )

    if settings.top_k > feature_count:
        logger.warning(
            "Configured top_k=%d exceeds available features=%d. "
            "Using all available features.",
            settings.top_k,
            feature_count,
        )

    top_k = min(settings.top_k, feature_count)

    if method == "mutual_information":
        selector = SelectKBest(
            score_func=mutual_info_classif,
            k=top_k,
        )

    else:
        raise DataError(
            f"Unsupported feature selection method: {settings.method}"
        )

    logger.info(
        "Feature selector created: method=%s, top_k=%d",
        method,
        top_k,
    )

    return selector