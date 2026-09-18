from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_option("client.showSidebarNavigation", False)

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPORTS_DIR = Path("reports")
METRICS_DIR = REPORTS_DIR / "metrics"

HOLDOUT_METRICS_PATH = METRICS_DIR / "model_metrics.csv"
CROSS_VALIDATION_PATH = METRICS_DIR / "cross_validation.csv"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def render_html(markup: str) -> None:
    """Render trusted HTML markup."""
    st.markdown(markup, unsafe_allow_html=True)


def load_metrics(path: Path) -> pd.DataFrame | None:
    """Load a metrics CSV file when it exists."""
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def get_metric_value(
    metrics: pd.DataFrame,
    metric_name: str,
    model_name: str | None = None,
) -> float | None:
    """Return a metric value from a metrics dataframe."""
    if metrics.empty:
        return None

    metric_column = None

    for column in metrics.columns:
        if column.lower() in {
            "metric",
            "metrics",
            "metric_name",
        }:
            metric_column = column
            break

    if metric_column is None:
        return None

    rows = metrics[
        metrics[metric_column].astype(str).str.lower()
        == metric_name.lower()
    ]

    if model_name is not None:
        model_columns = [
            column
            for column in metrics.columns
            if column.lower() in {"model", "model_name"}
        ]

        if model_columns:
            rows = rows[
                rows[model_columns[0]].astype(str).str.lower()
                == model_name.lower()
            ]

    if rows.empty:
        return None

    value_columns = [
        column
        for column in rows.columns
        if column != metric_column
        and column.lower() not in {"model", "model_name"}
    ]

    if not value_columns:
        return None

    try:
        return float(rows.iloc[0][value_columns[0]])
    except (TypeError, ValueError):
        return None


def format_percentage(value: float | None) -> str:
    """Format a metric as a percentage."""
    if value is None:
        return "N/A"

    if abs(value) <= 1:
        value *= 100

    return f"{value:.2f}%"


# ---------------------------------------------------------------------------
# Custom navigation
# ---------------------------------------------------------------------------

def render_navigation() -> None:
    """Render the application navigation bar."""

    render_html('<div class="custom-navigation">')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(
            "Overview",
            key="nav_overview",
            use_container_width=True,
        ):
            st.switch_page("app.py")

    with col2:
        if st.button(
            "Model Performance",
            key="nav_model",
            use_container_width=True,
        ):
            st.switch_page("pages/3_Model_Performance.py")

    with col3:
        if st.button(
            "Prediction",
            key="nav_prediction",
            use_container_width=True,
        ):
            st.switch_page("pages/2_Prediction.py")

    with col4:
        if st.button(
            "About",
            key="nav_about",
            use_container_width=True,
        ):
            st.switch_page("pages/4_About.py")

    render_html("</div>")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

render_html(
    """
    <style>
        .stApp {
            background: #f8fafc;
        }

        header[data-testid="stHeader"] {
            background: transparent;
            box-shadow: none;
        }

        header[data-testid="stHeader"] [data-testid="stToolbar"] {
            visibility: visible;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .custom-navigation {
            margin-bottom: 1.5rem;
        }

        .custom-navigation button {
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            color: #334155;
            font-weight: 600;
            min-height: 42px;
        }

        .custom-navigation button:hover {
            border-color: #94a3b8;
            color: #0f172a;
        }

        .page-title {
            margin-top: 0.5rem;
            margin-bottom: 0.25rem;
            color: #0f172a;
            font-size: 2.2rem;
            font-weight: 700;
        }

        .page-subtitle {
            margin-bottom: 2rem;
            color: #64748b;
            font-size: 1rem;
        }

        .metric-card {
            padding: 1.25rem;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }

        .metric-label {
            margin-bottom: 0.35rem;
            color: #64748b;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .metric-value {
            color: #0f172a;
            font-size: 1.65rem;
            font-weight: 700;
        }

        .section-title {
            margin-top: 2rem;
            margin-bottom: 0.75rem;
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 700;
        }

        .info-card {
            padding: 1.25rem;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            background: #ffffff;
        }

        .info-card h4 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            color: #0f172a;
        }

        .info-card p {
            margin-bottom: 0;
            color: #475569;
            line-height: 1.6;
        }

        .disclaimer {
            margin-top: 2rem;
            padding: 1rem 1.25rem;
            border-left: 4px solid #64748b;
            border-radius: 8px;
            background: #f1f5f9;
            color: #475569;
            line-height: 1.6;
        }
    </style>
    """
)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the Model Performance page."""

    render_navigation()

    render_html(
        """
        <div class="page-title">Model Performance</div>
        <div class="page-subtitle">
            Evaluation results for the customer churn machine learning system.
        </div>
        """
    )

    holdout_metrics = load_metrics(HOLDOUT_METRICS_PATH)
    cross_validation_metrics = load_metrics(CROSS_VALIDATION_PATH)

    # -----------------------------------------------------------------------
    # Holdout evaluation
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Holdout Test Performance</div>',
        unsafe_allow_html=True,
    )

    if holdout_metrics is None:
        st.warning(
            "Holdout metrics could not be loaded. "
            f"Expected file: `{HOLDOUT_METRICS_PATH}`"
        )
    else:
        accuracy = get_metric_value(
            holdout_metrics,
            "accuracy",
            "GradientBoosting",
        )

        precision = get_metric_value(
            holdout_metrics,
            "precision_weighted",
            "GradientBoosting",
        )

        recall = get_metric_value(
            holdout_metrics,
            "recall_weighted",
            "GradientBoosting",
        )

        f1 = get_metric_value(
            holdout_metrics,
            "f1_weighted",
            "GradientBoosting",
        )

        roc_auc = get_metric_value(
            holdout_metrics,
            "roc_auc",
            "GradientBoosting",
        )

        metric_columns = st.columns(5)

        metric_values = [
            ("Accuracy", accuracy),
            ("Precision", precision),
            ("Recall", recall),
            ("F1 Score", f1),
            ("ROC-AUC", roc_auc),
        ]

        for column, (label, value) in zip(
            metric_columns,
            metric_values,
        ):
            with column:
                render_html(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">
                            {format_percentage(value)}
                        </div>
                    </div>
                    """
                )

    # -----------------------------------------------------------------------
    # Cross-validation results
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">5-Fold Cross-Validation</div>',
        unsafe_allow_html=True,
    )

    if cross_validation_metrics is None:
        st.info(
            "Cross-validation results are not available yet. "
            f"Expected file: `{CROSS_VALIDATION_PATH}`"
        )
    else:
        st.dataframe(
            cross_validation_metrics,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------------------------
    # Model selection
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Selection</div>',
        unsafe_allow_html=True,
    )

    selection_col1, selection_col2 = st.columns(2)

    with selection_col1:
        render_html(
            """
            <div class="info-card">
                <h4>Selected Model</h4>
                <p>
                    Gradient Boosting was selected for the final persisted
                    pipeline based on the holdout ROC-AUC comparison.
                </p>
            </div>
            """
        )

    with selection_col2:
        render_html(
            """
            <div class="info-card">
                <h4>Evaluation Strategy</h4>
                <p>
                    The final evaluation uses a stratified holdout test set.
                    Five-fold stratified cross-validation is used on the
                    training split to assess model stability without replacing
                    the holdout evaluation.
                </p>
            </div>
            """
        )

    # -----------------------------------------------------------------------
    # Methodology
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Evaluation Methodology</div>',
        unsafe_allow_html=True,
    )

    methodology_col1, methodology_col2 = st.columns(2)

    with methodology_col1:
        render_html(
            """
            <div class="info-card">
                <h4>Data Split</h4>
                <p>
                    The processed dataset is divided into training and holdout
                    test sets using an 80/20 stratified split with a fixed
                    random state of 42.
                </p>
            </div>
            """
        )

    with methodology_col2:
        render_html(
            """
            <div class="info-card">
                <h4>Cross-Validation</h4>
                <p>
                    Stratified five-fold cross-validation is applied only to
                    the training data. Accuracy, weighted precision, weighted
                    recall, weighted F1, and ROC-AUC are reported.
                </p>
            </div>
            """
        )

    # -----------------------------------------------------------------------
    # Pipeline configuration
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Pipeline Configuration</div>',
        unsafe_allow_html=True,
    )

    pipeline_col1, pipeline_col2, pipeline_col3 = st.columns(3)

    with pipeline_col1:
        render_html(
            """
            <div class="info-card">
                <h4>Feature Selection</h4>
                <p>
                    Mutual Information SelectKBest is used to select the top
                    10 predictive features.
                </p>
            </div>
            """
        )

    with pipeline_col2:
        render_html(
            """
            <div class="info-card">
                <h4>Preprocessing</h4>
                <p>
                    Numerical features use imputation and standard scaling.
                    Categorical features use imputation and one-hot encoding.
                </p>
            </div>
            """
        )

    with pipeline_col3:
        render_html(
            """
            <div class="info-card">
                <h4>Final Model</h4>
                <p>
                    The persisted model uses Gradient Boosting with the
                    configured training pipeline.
                </p>
            </div>
            """
        )

    # -----------------------------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------------------------

    render_html(
        """
        <div class="disclaimer">
            <strong>Note:</strong>
            These metrics describe performance on the available dataset and
            evaluation splits. They should not be interpreted as a guarantee
            of performance on unseen real-world customers.
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()