from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from ml_system.config.loader import validate_settings
from ml_system.config.schemas import (
    DataSettings,
    EncodingSettings,
    EvaluationSettings,
    FeatureSelectionSettings,
    FeatureSettings,
    MissingValueSettings,
    ModelSettings,
    PersistenceSettings,
    PreprocessingSettings,
    ProjectSettings,
    ScalingSettings,
    Settings,
    SplitSettings,
    TaskSettings,
)
from ml_system.data.splitter import split_dataset
from ml_system.data.validator import validate_dataset
from ml_system.exceptions.errors import (
    ConfigurationError,
    DataError,
    EvaluationError,
    ModelError,
)
from ml_system.evaluation import evaluate_model
from ml_system.features import build_feature_selector
from ml_system.models import build_model
from ml_system.persistence import load_model, save_model
from ml_system.preprocessing import build_preprocessing_pipeline
from ml_system.training import (
    build_training_pipeline,
    train_model,
)


@pytest.fixture
def classification_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [21, 25, 31, 35, 42, 48, 53, 61],
            "MonthlyCharges": [
                20.0,
                25.0,
                31.0,
                35.0,
                42.0,
                48.0,
                53.0,
                61.0,
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
            ],
        }
    )


@pytest.fixture
def preprocessing_settings() -> PreprocessingSettings:
    return PreprocessingSettings(
        missing_values=MissingValueSettings(
            strategy="median",
        ),
        categorical_encoding=EncodingSettings(
            strategy="onehot",
        ),
        numerical_scaling=ScalingSettings(
            strategy="standard",
        ),
    )


@pytest.fixture
def feature_selection_settings() -> FeatureSelectionSettings:
    return FeatureSelectionSettings(
        enabled=True,
        method="mutual_information",
        top_k=2,
    )


@pytest.fixture
def evaluation_settings() -> EvaluationSettings:
    return EvaluationSettings(
        metrics=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        project=ProjectSettings(
            name="customer_churn_ml",
            version="1.0",
        ),
        data=DataSettings(
            path=str(tmp_path / "customer_churn.csv"),
            target_column="Churn",
        ),
        task=TaskSettings(
            type="classification",
        ),
        split=SplitSettings(
            test_size=0.20,
            random_state=42,
            stratify=True,
        ),
        preprocessing=PreprocessingSettings(
            missing_values=MissingValueSettings(
                strategy="median",
            ),
            categorical_encoding=EncodingSettings(
                strategy="onehot",
            ),
            numerical_scaling=ScalingSettings(
                strategy="standard",
            ),
        ),
        features=FeatureSettings(
            selection=FeatureSelectionSettings(
                enabled=True,
                method="mutual_information",
                top_k=2,
            ),
        ),
        model=ModelSettings(
            name="random_forest",
            parameters={
                "n_estimators": 10,
                "random_state": 42,
            },
        ),
        evaluation=EvaluationSettings(
            metrics=["accuracy"],
        ),
        persistence=PersistenceSettings(
            model_path=str(
                tmp_path / "models" / "model.joblib"
            ),
        ),
    )


# ==========================================================
# CONFIGURATION
# ==========================================================


def test_validate_settings_rejects_empty_dataset_path(
    settings: Settings,
):
    settings.data.path = ""

    with pytest.raises(
        ConfigurationError,
        match="Dataset path cannot be empty",
    ):
        validate_settings(settings)


def test_validate_settings_rejects_empty_target_column(
    settings: Settings,
):
    settings.data.target_column = ""

    with pytest.raises(
        ConfigurationError,
        match="Target column cannot be empty",
    ):
        validate_settings(settings)


def test_validate_settings_rejects_invalid_test_size(
    settings: Settings,
):
    settings.split.test_size = 1.0

    with pytest.raises(
        ConfigurationError,
        match="test_size must be between 0 and 1",
    ):
        validate_settings(settings)


def test_validate_settings_rejects_empty_model_name(
    settings: Settings,
):
    settings.model.name = ""

    with pytest.raises(
        ConfigurationError,
        match="Model name cannot be empty",
    ):
        validate_settings(settings)


def test_validate_settings_rejects_empty_metrics(
    settings: Settings,
):
    settings.evaluation.metrics = []

    with pytest.raises(
        ConfigurationError,
        match="At least one evaluation metric",
    ):
        validate_settings(settings)


# ==========================================================
# DATA VALIDATION
# ==========================================================


def test_validate_dataset_rejects_non_dataframe():
    with pytest.raises(
        DataError,
        match="Dataset must be a pandas DataFrame",
    ):
        validate_dataset(
            dataframe="invalid",
            target_column="Churn",
        )


def test_validate_dataset_rejects_empty_dataframe():
    with pytest.raises(
        DataError,
        match="Dataset is empty",
    ):
        validate_dataset(
            dataframe=pd.DataFrame(),
            target_column="Churn",
        )


def test_validate_dataset_rejects_missing_target(
    classification_dataset: pd.DataFrame,
):
    with pytest.raises(
        DataError,
        match="Target column not found",
    ):
        validate_dataset(
            dataframe=classification_dataset,
            target_column="MissingTarget",
        )


def test_validate_dataset_rejects_target_with_no_valid_values():
    dataframe = pd.DataFrame(
        {
            "Age": [21, 25, 30],
            "Churn": [None, None, None],
        }
    )

    with pytest.raises(
        DataError,
        match="Target column contains no valid observations",
    ):
        validate_dataset(
            dataframe=dataframe,
            target_column="Churn",
        )


# ==========================================================
# DATA SPLITTING
# ==========================================================


def test_split_dataset_rejects_missing_target(
    classification_dataset: pd.DataFrame,
):
    settings = SplitSettings()

    with pytest.raises(
        DataError,
        match="Target column not found",
    ):
        split_dataset(
            dataframe=classification_dataset,
            target_column="MissingTarget",
            settings=settings,
        )


def test_split_dataset_rejects_missing_target_values(
    classification_dataset: pd.DataFrame,
):
    dataframe = classification_dataset.copy()
    dataframe.loc[0, "Churn"] = None

    settings = SplitSettings()

    with pytest.raises(
        DataError,
        match="Target column contains missing values",
    ):
        split_dataset(
            dataframe=dataframe,
            target_column="Churn",
            settings=settings,
        )


def test_split_dataset_rejects_single_class_target(
    classification_dataset: pd.DataFrame,
):
    dataframe = classification_dataset.copy()
    dataframe["Churn"] = "No"

    settings = SplitSettings()

    with pytest.raises(
        DataError,
        match="at least two classes",
    ):
        split_dataset(
            dataframe=dataframe,
            target_column="Churn",
            settings=settings,
        )


def test_split_dataset_rejects_invalid_test_size(
    classification_dataset: pd.DataFrame,
):
    settings = SplitSettings(
        test_size=1.0,
    )

    with pytest.raises(
        DataError,
        match="test_size must be between 0 and 1",
    ):
        split_dataset(
            dataframe=classification_dataset,
            target_column="Churn",
            settings=settings,
        )


def test_split_dataset_can_disable_stratification(
    classification_dataset: pd.DataFrame,
):
    settings = SplitSettings(
        test_size=0.25,
        random_state=42,
        stratify=False,
    )

    X_train, X_test, y_train, y_test = split_dataset(
        dataframe=classification_dataset,
        target_column="Churn",
        settings=settings,
    )

    assert len(X_train) == 6
    assert len(X_test) == 2
    assert len(y_train) == 6
    assert len(y_test) == 2


# ==========================================================
# PREPROCESSING
# ==========================================================


def test_preprocessing_rejects_missing_target(
    classification_dataset: pd.DataFrame,
    preprocessing_settings: PreprocessingSettings,
):
    with pytest.raises(
        DataError,
        match="Target column not found",
    ):
        build_preprocessing_pipeline(
            dataframe=classification_dataset,
            target_column="MissingTarget",
            settings=preprocessing_settings,
        )


def test_preprocessing_rejects_invalid_encoding_strategy(
    classification_dataset: pd.DataFrame,
    preprocessing_settings: PreprocessingSettings,
):
    preprocessing_settings.categorical_encoding.strategy = "label"

    with pytest.raises(
        DataError,
        match="Unsupported categorical encoding strategy",
    ):
        build_preprocessing_pipeline(
            dataframe=classification_dataset,
            target_column="Churn",
            settings=preprocessing_settings,
        )


def test_preprocessing_rejects_invalid_scaling_strategy(
    classification_dataset: pd.DataFrame,
    preprocessing_settings: PreprocessingSettings,
):
    preprocessing_settings.numerical_scaling.strategy = "minmax"

    with pytest.raises(
        DataError,
        match="Unsupported numerical scaling strategy",
    ):
        build_preprocessing_pipeline(
            dataframe=classification_dataset,
            target_column="Churn",
            settings=preprocessing_settings,
        )


def test_preprocessing_rejects_invalid_missing_value_strategy(
    classification_dataset: pd.DataFrame,
    preprocessing_settings: PreprocessingSettings,
):
    preprocessing_settings.missing_values.strategy = "invalid"

    with pytest.raises(
        DataError,
        match="Unsupported numerical missing-value strategy",
    ):
        build_preprocessing_pipeline(
            dataframe=classification_dataset,
            target_column="Churn",
            settings=preprocessing_settings,
        )


# ==========================================================
# FEATURE SELECTION
# ==========================================================


def test_feature_selection_can_be_disabled():
    settings = FeatureSelectionSettings(
        enabled=False,
    )

    selector = build_feature_selector(
        settings=settings,
        feature_count=5,
    )

    assert selector is None


def test_feature_selection_rejects_invalid_method():
    settings = FeatureSelectionSettings(
        enabled=True,
        method="invalid_method",
        top_k=2,
    )

    with pytest.raises(
        DataError,
        match="Unsupported feature selection method",
    ):
        build_feature_selector(
            settings=settings,
            feature_count=5,
        )


def test_feature_selection_rejects_invalid_top_k():
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
            settings=settings,
            feature_count=5,
        )


def test_feature_selection_limits_top_k_to_available_features():
    settings = FeatureSelectionSettings(
        enabled=True,
        method="mutual_information",
        top_k=10,
    )

    selector = build_feature_selector(
        settings=settings,
        feature_count=3,
    )

    assert selector is not None
    assert selector.k == 3


# ==========================================================
# MODEL FACTORY
# ==========================================================


def test_model_factory_rejects_empty_model_name():
    settings = ModelSettings(
        name="",
    )

    with pytest.raises(
        ModelError,
        match="Model name cannot be empty",
    ):
        build_model(settings)


def test_model_factory_rejects_unknown_model():
    settings = ModelSettings(
        name="unsupported_model",
    )

    with pytest.raises(
        ModelError,
        match="Unsupported model",
    ):
        build_model(settings)


def test_model_factory_builds_all_required_models():
    model_names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ]

    for model_name in model_names:
        model = build_model(
            ModelSettings(
                name=model_name,
            )
        )

        assert model is not None


# ==========================================================
# TRAINING
# ==========================================================


def test_training_pipeline_rejects_invalid_model(
    classification_dataset: pd.DataFrame,
    preprocessing_settings: PreprocessingSettings,
    feature_selection_settings: FeatureSelectionSettings,
):
    with pytest.raises(
        ModelError,
        match="valid scikit-learn estimator",
    ):
        build_training_pipeline(
            dataframe=classification_dataset,
            target_column="Churn",
            preprocessing_settings=preprocessing_settings,
            feature_selection_settings=feature_selection_settings,
            model="invalid",
        )


def test_training_rejects_missing_target(
    classification_dataset: pd.DataFrame,
    preprocessing_settings: PreprocessingSettings,
    feature_selection_settings: FeatureSelectionSettings,
):
    model = LogisticRegression(
        max_iter=1000,
    )

    with pytest.raises(
        DataError,
        match="Target column not found",
    ):
        build_training_pipeline(
            dataframe=classification_dataset,
            target_column="MissingTarget",
            preprocessing_settings=preprocessing_settings,
            feature_selection_settings=feature_selection_settings,
            model=model,
        )


def test_training_rejects_single_class_target(
    classification_dataset: pd.DataFrame,
    preprocessing_settings: PreprocessingSettings,
    feature_selection_settings: FeatureSelectionSettings,
):
    model = LogisticRegression(
        max_iter=1000,
    )

    training_data = classification_dataset.copy()
    training_data["Churn"] = "No"

    pipeline = build_training_pipeline(
        dataframe=classification_dataset,
        target_column="Churn",
        preprocessing_settings=preprocessing_settings,
        feature_selection_settings=feature_selection_settings,
        model=model,
    )

    with pytest.raises(
        DataError,
        match="at least two classes",
    ):
        train_model(
            pipeline=pipeline,
            training_data=training_data,
            target_column="Churn",
        )


# ==========================================================
# EVALUATION
# ==========================================================


def test_evaluation_rejects_empty_metrics(
    classification_dataset: pd.DataFrame,
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    with pytest.raises(
        EvaluationError,
        match="At least one evaluation metric",
    ):
        evaluate_model(
            pipeline=pipeline,
            test_data=classification_dataset,
            target_column="Churn",
            metrics=[],
        )


def test_evaluation_rejects_unsupported_metric(
    classification_dataset: pd.DataFrame,
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    with pytest.raises(
        EvaluationError,
        match="Unsupported evaluation metric",
    ):
        evaluate_model(
            pipeline=pipeline,
            test_data=classification_dataset,
            target_column="Churn",
            metrics=["invalid_metric"],
        )


def test_evaluation_rejects_missing_target(
    classification_dataset: pd.DataFrame,
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    with pytest.raises(
        DataError,
        match="Target column not found",
    ):
        evaluate_model(
            pipeline=pipeline,
            test_data=classification_dataset,
            target_column="MissingTarget",
            metrics=["accuracy"],
        )


# ==========================================================
# PERSISTENCE
# ==========================================================


def test_save_model_rejects_invalid_extension(
    tmp_path: Path,
):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    with pytest.raises(
        ModelError,
        match=r"\.joblib extension",
    ):
        save_model(
            model=model,
            path=tmp_path / "model.pkl",
        )


def test_load_model_rejects_missing_file(
    tmp_path: Path,
):
    with pytest.raises(
        ModelError,
        match="Model file not found",
    ):
        load_model(
            tmp_path / "missing.joblib"
        )


def test_load_model_rejects_invalid_extension(
    tmp_path: Path,
):
    model_path = tmp_path / "model.pkl"
    model_path.write_text(
        "invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        ModelError,
        match=r"\.joblib extension",
    ):
        load_model(model_path)


def test_save_model_rejects_non_pipeline_object(
    tmp_path: Path,
):
    with pytest.raises(
        ModelError,
        match="Only scikit-learn Pipeline",
    ):
        save_model(
            model=LogisticRegression(),
            path=tmp_path / "model.joblib",
        )