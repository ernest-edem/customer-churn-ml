import numpy as np
import pytest

from ml_system.config.schemas import FeatureSelectionSettings
from ml_system.exceptions.errors import DataError
from ml_system.features import build_feature_selector


def test_build_mutual_information_selector():
    settings = FeatureSelectionSettings(
        enabled=True,
        method="mutual_information",
        top_k=3,
    )

    selector = build_feature_selector(
        settings,
        feature_count=5,
    )

    assert selector is not None
    assert selector.k == 3


def test_feature_selector_selects_top_k_features():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    settings = FeatureSelectionSettings(
        enabled=True,
        method="mutual_information",
        top_k=3,
    )

    selector = build_feature_selector(
        settings,
        feature_count=X.shape[1],
    )

    assert selector is not None

    selector.fit(X, y)

    transformed = selector.transform(X)

    assert transformed.shape == (100, 3)
    assert selector.get_support().sum() == 3


def test_feature_selector_disabled():
    settings = FeatureSelectionSettings(
        enabled=False,
        method="mutual_information",
        top_k=3,
    )

    selector = build_feature_selector(
        settings,
        feature_count=5,
    )

    assert selector is None


def test_feature_selector_caps_top_k_to_available_features():
    settings = FeatureSelectionSettings(
        enabled=True,
        method="mutual_information",
        top_k=10,
    )

    selector = build_feature_selector(
        settings,
        feature_count=4,
    )

    assert selector is not None
    assert selector.k == 4


def test_feature_selector_rejects_invalid_method():
    settings = FeatureSelectionSettings(
        enabled=True,
        method="chi_square",
        top_k=3,
    )

    with pytest.raises(
        DataError,
        match="Unsupported feature selection method",
    ):
        build_feature_selector(
            settings,
            feature_count=5,
        )


def test_feature_selector_rejects_invalid_top_k():
    settings = FeatureSelectionSettings(
        enabled=True,
        method="mutual_information",
        top_k=0,
    )

    with pytest.raises(
        DataError,
        match="top_k must be greater than zero",
    ):
        build_feature_selector(
            settings,
            feature_count=5,
        )


def test_feature_selector_rejects_invalid_feature_count():
    settings = FeatureSelectionSettings(
        enabled=True,
        method="mutual_information",
        top_k=3,
    )

    with pytest.raises(
        DataError,
        match="Feature count must be greater than zero",
    ):
        build_feature_selector(
            settings,
            feature_count=0,
        )