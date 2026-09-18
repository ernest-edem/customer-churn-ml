from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Prediction Results",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------

MODEL_PATH = Path("models/model.joblib")

PAGE_ASSESSMENT = "assessment"
PAGE_RESULTS = "results"
PAGE_MODEL = "model"


# ---------------------------------------------------------------------------
# Shared navigation
# ---------------------------------------------------------------------------

def render_navigation(active_page: str) -> None:
    """Render the shared application header and page navigation."""

    st.markdown(
        """
        <style>
        /* ---------------------------------------------------------------
           Design tokens
        --------------------------------------------------------------- */

        :root {
            --surface: #ffffff;
            --surface-muted: #f8fafc;
            --border: #e2e8f0;
            --border-strong: #cbd5e1;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
            --text-tertiary: #94a3b8;
            --accent: #2563eb;
            --accent-soft: #eff6ff;
            --radius-lg: 20px;
            --radius-md: 14px;
            --radius-sm: 10px;
            --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.04);
            --shadow-md:
                0 1px 2px rgba(15, 23, 42, 0.04),
                0 10px 30px rgba(15, 23, 42, 0.06);
            --shadow-lg:
                0 2px 4px rgba(15, 23, 42, 0.04),
                0 18px 48px rgba(15, 23, 42, 0.10);
        }

        /* ---------------------------------------------------------------
           Sidebar removal (all Streamlit versions)
        --------------------------------------------------------------- */

        section[data-testid="stSidebar"],
        div[data-testid="stSidebar"],
        div[data-testid="stSidebarNav"],
        div[data-testid="stSidebarNavItems"],
        div[data-testid="stSidebarContent"],
        div[data-testid="stSidebarCollapsedControl"],
        div[data-testid="collapsedControl"],
        button[data-testid="stSidebarCollapseButton"],
        button[kind="header"] {
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
            visibility: hidden !important;
        }

        div[data-testid="stAppViewContainer"] > section:first-child {
            display: none !important;
        }

        div[data-testid="stAppViewContainer"] {
            margin-left: 0 !important;
            padding-left: 0 !important;
        }

        /* ---------------------------------------------------------------
           Streamlit chrome
           Keep the top-right toolbar and three-dot menu available while
           removing the visible default header decoration.
        --------------------------------------------------------------- */

        header[data-testid="stHeader"] {
            background: transparent;
            box-shadow: none;
            height: 0;
        }

        header[data-testid="stHeader"] > div {
            background: transparent;
        }

        header[data-testid="stHeader"] [data-testid="stToolbar"] {
            visibility: visible;
            right: 0.75rem;
        }

        footer {
            visibility: hidden;
            height: 0;
        }

        /* ---------------------------------------------------------------
           Main application container
        --------------------------------------------------------------- */

        .stApp {
            background:
                radial-gradient(
                    1200px 600px at 50% -10%,
                    #f1f5f9 0%,
                    transparent 60%
                ),
                #ffffff;
        }

        .block-container {
            width: 100%;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
            padding-top: 1.75rem;
            padding-bottom: 3.5rem;
        }

        /* ---------------------------------------------------------------
           Application header
        --------------------------------------------------------------- */

        .custom-navigation {
            width: 100%;
            min-height: 72px;
            display: flex;
            align-items: center;
        }

        .app-header {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            white-space: nowrap;
            padding: 0.35rem 0;
        }

        .brand {
            color: var(--text-primary);
            font-size: 1.32rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.05;
        }

        .brand-accent {
            color: var(--accent);
        }

        .brand-subtitle {
            color: var(--text-secondary);
            font-size: 0.78rem;
            font-weight: 500;
            line-height: 1.25;
            margin-top: 0.3rem;
            letter-spacing: 0.005em;
        }

        .navigation-divider {
            height: 34px;
            width: 1px;
            background: var(--border);
            margin: 0 0.35rem;
        }

        /* ---------------------------------------------------------------
           Navigation buttons
        --------------------------------------------------------------- */

        .custom-navigation button,
        div[data-testid="stHorizontalBlock"] button {
            font-weight: 600;
            border-radius: var(--radius-sm);
            min-height: 40px;
            transition:
                background-color 160ms ease,
                border-color 160ms ease,
                box-shadow 160ms ease,
                transform 120ms ease;
        }

        div[data-testid="stHorizontalBlock"] button:hover {
            transform: translateY(-1px);
        }

        div[data-testid="stHorizontalBlock"] button:active {
            transform: translateY(0);
        }

        button[kind="secondary"] {
            border-color: var(--border) !important;
            color: var(--text-secondary) !important;
            background: var(--surface) !important;
        }

        button[kind="secondary"]:hover {
            border-color: var(--accent) !important;
            color: var(--accent) !important;
            background: var(--accent-soft) !important;
        }

        button[kind="primary"] {
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.22);
        }

        button[kind="primary"]:hover {
            box-shadow: 0 10px 26px rgba(37, 99, 235, 0.30);
        }

        /* Active navigation button */
        .nav-active + div button,
        button.nav-active {
            font-weight: 700;
        }

        hr,
        div[data-testid="stDivider"] hr {
            margin: 0.5rem 0 2rem;
            border-color: var(--border);
        }

        /* ---------------------------------------------------------------
           Dashboard styling
        --------------------------------------------------------------- */

        .page-eyebrow {
            display: inline-block;
            color: var(--accent);
            background: var(--accent-soft);
            border: 1px solid rgba(37, 99, 235, 0.16);
            border-radius: 999px;
            padding: 0.35rem 0.85rem;
            font-size: 0.7rem;
            font-weight: 750;
            letter-spacing: 0.1em;
            line-height: 1.2;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }

        .page-title {
            color: var(--text-primary);
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.14;
            margin: 0;
        }

        .page-description {
            color: var(--text-secondary);
            font-size: 1.02rem;
            line-height: 1.7;
            margin-top: 0.85rem;
            max-width: 720px;
        }

        .result-card {
            height: 100%;
            box-sizing: border-box;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1.6rem 1.75rem;
            margin-bottom: 0.75rem;
            background: var(--surface);
            box-shadow: var(--shadow-md);
            transition:
                transform 200ms ease,
                box-shadow 200ms ease,
                border-color 200ms ease;
        }

        .result-card:hover {
            transform: translateY(-3px);
            border-color: var(--border-strong);
            box-shadow: var(--shadow-lg);
        }

        .result-label {
            color: var(--text-secondary);
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            line-height: 1.3;
        }

        .result-value {
            color: var(--text-primary);
            font-size: 1.62rem;
            font-weight: 750;
            letter-spacing: -0.025em;
            line-height: 1.25;
            margin-top: 0.55rem;
        }

        .result-description {
            color: var(--text-secondary);
            font-size: 0.9rem;
            line-height: 1.65;
            margin-top: 0.6rem;
        }

        .probability-value {
            color: var(--text-primary);
            font-size: 2.9rem;
            font-weight: 800;
            letter-spacing: -0.045em;
            line-height: 1.05;
            margin-top: 0.45rem;
        }

        .section-title {
            color: var(--text-primary);
            font-size: 1.28rem;
            font-weight: 750;
            letter-spacing: -0.022em;
            line-height: 1.3;
            margin: 1.25rem 0 0.75rem;
        }

        .section-description {
            color: var(--text-secondary);
            font-size: 0.93rem;
            line-height: 1.65;
            margin-bottom: 1.15rem;
            max-width: 780px;
        }

        .assessment-note {
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: var(--radius-md);
            background: var(--surface-muted);
            padding: 1.35rem 1.5rem;
            color: #334155;
            font-size: 0.93rem;
            line-height: 1.75;
        }

        .disclaimer {
            border: 1px solid #fde68a;
            border-radius: var(--radius-md);
            background: #fffbeb;
            color: #78350f;
            padding: 1.15rem 1.35rem;
            font-size: 0.87rem;
            line-height: 1.7;
        }

        .empty-state {
            border: 1px dashed var(--border-strong);
            border-radius: var(--radius-lg);
            padding: 4rem 2rem;
            text-align: center;
            background: var(--surface-muted);
        }

        .empty-state-icon {
            font-size: 2.6rem;
            margin-bottom: 0.9rem;
            opacity: 0.85;
        }

        .empty-state-title {
            color: var(--text-primary);
            font-size: 1.35rem;
            font-weight: 750;
            letter-spacing: -0.025em;
            line-height: 1.3;
        }

        .empty-state-description {
            color: var(--text-secondary);
            max-width: 560px;
            margin: 0.75rem auto 0;
            font-size: 0.96rem;
            line-height: 1.7;
        }

        /* ---------------------------------------------------------------
           Metrics and progress
        --------------------------------------------------------------- */

        div[data-testid="stMetric"] {
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            background: var(--surface);
            padding: 0.9rem 1rem;
            margin-bottom: 0.6rem;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--text-secondary);
        }

        .stProgress > div > div {
            border-radius: 999px;
        }

        div[data-testid="stCaptionContainer"] {
            color: var(--text-tertiary);
            margin-top: 0.35rem;
        }

        /* ---------------------------------------------------------------
           Responsive layout
        --------------------------------------------------------------- */

        @media (max-width: 900px) {
            .block-container {
                max-width: 100%;
                padding-left: 1.25rem;
                padding-right: 1.25rem;
            }

            .page-title {
                font-size: 1.95rem;
            }

            .probability-value {
                font-size: 2.4rem;
            }

            .result-card {
                height: auto;
                padding: 1.4rem;
            }
        }

        @media (max-width: 640px) {
            .app-header {
                justify-content: center;
            }

            .brand-subtitle,
            .navigation-divider {
                display: none;
            }

            .custom-navigation {
                min-height: auto;
            }

            .page-title {
                font-size: 1.62rem;
            }

            .page-description {
                font-size: 0.93rem;
            }

            .probability-value {
                font-size: 2.1rem;
            }

            .empty-state {
                padding: 2.75rem 1.25rem;
            }
        }

        /* ---------------------------------------------------------------
           Dark mode support
        --------------------------------------------------------------- */

        @media (prefers-color-scheme: dark) {
            :root {
                --surface: #0f172a;
                --surface-muted: #111827;
                --border: #334155;
                --border-strong: #475569;
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --text-tertiary: #64748b;
                --accent: #60a5fa;
                --accent-soft: rgba(96, 165, 250, 0.12);
                --shadow-md: none;
                --shadow-lg: 0 18px 48px rgba(0, 0, 0, 0.45);
            }

            .stApp {
                background:
                    radial-gradient(
                        1200px 600px at 50% -10%,
                        #0b1220 0%,
                        transparent 60%
                    ),
                    #020617;
            }

            .brand-subtitle,
            .page-description,
            .result-description,
            .section-description,
            .result-label,
            .empty-state-description {
                color: var(--text-secondary);
            }

            .page-title,
            .result-value,
            .section-title,
            .empty-state-title,
            .probability-value,
            .brand {
                color: var(--text-primary);
            }

            .navigation-divider {
                background: var(--border);
            }

            .result-card,
            div[data-testid="stMetric"] {
                border-color: var(--border);
                background: var(--surface);
                box-shadow: none;
            }

            .result-card:hover {
                border-color: var(--border-strong);
                box-shadow: var(--shadow-lg);
            }

            .assessment-note {
                background: var(--surface-muted);
                color: #cbd5e1;
                border-color: var(--border);
                border-left-color: var(--accent);
            }

            .disclaimer {
                border-color: rgba(253, 230, 138, 0.28);
                background: rgba(120, 53, 15, 0.18);
                color: #fcd34d;
            }

            .empty-state {
                border-color: var(--border-strong);
                background: var(--surface-muted);
            }

            .page-eyebrow {
                border-color: rgba(96, 165, 250, 0.22);
            }

            button[kind="secondary"] {
                background: var(--surface) !important;
                border-color: var(--border) !important;
                color: var(--text-secondary) !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="custom-navigation">', unsafe_allow_html=True)

    header_left, header_navigation = st.columns(
        [2.15, 1],
        vertical_alignment="center",
    )

    with header_left:
        st.markdown(
            """
            <div class="app-header">
                <div>
                    <div class="brand">
                        Customer Churn <span class="brand-accent">AI</span>
                    </div>
                    <div class="brand-subtitle">
                        Machine Learning Prediction System
                    </div>
                </div>
                <div class="navigation-divider"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with header_navigation:
        nav_assessment, nav_results, nav_model = st.columns(
            [1, 1, 1],
            vertical_alignment="center",
        )

        with nav_assessment:
            if st.button(
                "Assessment",
                key="nav_assessment",
                use_container_width=True,
                type=(
                    "primary"
                    if active_page == PAGE_ASSESSMENT
                    else "secondary"
                ),
            ):
                st.switch_page("app.py")

        with nav_results:
            if st.button(
                "Results",
                key="nav_results",
                use_container_width=True,
                type=(
                    "primary"
                    if active_page == PAGE_RESULTS
                    else "secondary"
                ),
            ):
                st.switch_page("pages/2_Prediction_Results.py")

        with nav_model:
            if st.button(
                "Model",
                key="nav_model",
                use_container_width=True,
                type=(
                    "primary"
                    if active_page == PAGE_MODEL
                    else "secondary"
                ),
            ):
                st.switch_page("pages/3_Model_Performance.py")

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_percentage(value: float) -> str:
    """Format a decimal probability as a percentage."""

    return f"{value * 100:.1f}%"


def get_prediction_label(prediction_result: Any) -> str:
    """Convert the stored prediction value into a readable label."""

    if isinstance(prediction_result, str):
        normalized = prediction_result.strip().lower()

        if normalized in {"yes", "churn", "1", "true"}:
            return "Likely to Churn"

        if normalized in {"no", "not churn", "0", "false"}:
            return "Likely to Stay"

        return prediction_result

    if isinstance(prediction_result, bool):
        return "Likely to Churn" if prediction_result else "Likely to Stay"

    if isinstance(prediction_result, (int, float)):
        return "Likely to Churn" if int(prediction_result) == 1 else "Likely to Stay"

    return str(prediction_result)


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def render_no_prediction_state() -> None:
    """Render the results page when no prediction exists."""

    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">📊</div>
            <div class="empty-state-title">
                No Prediction Available
            </div>
            <div class="empty-state-description">
                Complete a customer assessment to generate a churn prediction.
                Your prediction result and model assessment will appear here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if st.button(
        "Start Customer Assessment",
        type="primary",
        use_container_width=False,
    ):
        st.switch_page("app.py")


# ---------------------------------------------------------------------------
# Prediction dashboard
# ---------------------------------------------------------------------------

def render_dashboard(
    prediction_result: Any,
    churn_probability: float,
) -> None:
    """Render the complete customer prediction dashboard."""

    prediction_label = get_prediction_label(prediction_result)

    churn_probability = max(0.0, min(1.0, float(churn_probability)))

    if churn_probability >= 0.5:
        risk_description = (
            "The model estimates a higher likelihood that this customer "
            "will churn."
        )
    else:
        risk_description = (
            "The model estimates a lower likelihood that this customer "
            "will churn."
        )

    st.markdown(
        """
        <div class="page-eyebrow">Customer Assessment</div>
        <h1 class="page-title">Prediction Results</h1>
        <div class="page-description">
            Review the customer's predicted churn outcome and the model's
            estimated probability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # -----------------------------------------------------------------------
    # Primary result
    # -----------------------------------------------------------------------

    result_col, probability_col = st.columns(
        [1, 1],
        vertical_alignment="top",
    )

    with result_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Prediction</div>
                <div class="result-value">{prediction_label}</div>
                <div class="result-description">
                    {risk_description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with probability_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Churn Probability</div>
                <div class="probability-value">
                    {format_percentage(churn_probability)}
                </div>
                <div class="result-description">
                    Estimated probability of customer churn.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # -----------------------------------------------------------------------
    # Probability visualization
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Churn Probability</div>',
        unsafe_allow_html=True,
    )

    st.progress(
        churn_probability,
        text=f"Estimated churn probability: {format_percentage(churn_probability)}",
    )

    st.caption(
        "This probability is produced by the trained Gradient Boosting "
        "classification pipeline."
    )

    st.write("")

    # -----------------------------------------------------------------------
    # Model assessment
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Assessment</div>',
        unsafe_allow_html=True,
    )

    assessment_col, metrics_col = st.columns(
        [1.35, 1],
        vertical_alignment="top",
    )

    with assessment_col:
        st.markdown(
            f"""
            <div class="assessment-note">
                <strong>Interpretation</strong><br><br>
                The model classified this customer as
                <strong>{prediction_label.lower()}</strong> with an estimated
                churn probability of
                <strong>{format_percentage(churn_probability)}</strong>.
                <br><br>
                This result should be used as a decision support signal,
                alongside customer history, business context, and other
                relevant information.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metrics_col:
        metric_1, metric_2 = st.columns(2)

        with metric_1:
            st.metric("Accuracy", "78.75%")

        with metric_2:
            st.metric("ROC AUC", "83.44%")

        metric_3, metric_4 = st.columns(2)

        with metric_3:
            st.metric("Precision", "77.81%")

        with metric_4:
            st.metric("Recall", "78.75%")

    st.write("")

    # -----------------------------------------------------------------------
    # Model information
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True,
    )

    model_col_1, model_col_2, model_col_3 = st.columns(3)

    with model_col_1:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">Algorithm</div>
                <div class="result-value">Gradient Boosting</div>
                <div class="result-description">
                    Classification model selected during model benchmarking.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with model_col_2:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">Validation</div>
                <div class="result-value">5-Fold CV</div>
                <div class="result-description">
                    Stratified cross validation was performed on the
                    training split.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with model_col_3:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">ROC AUC</div>
                <div class="result-value">83.44%</div>
                <div class="result-description">
                    Holdout test ROC AUC used for final model evaluation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">What This Result Means</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="assessment-note">
            The prediction represents the output of the trained machine
            learning model for the information provided during the customer
            assessment. A higher churn probability indicates that the model
            identified patterns associated with customers who churned in the
            training data.
            <br><br>
            The prediction is not a guarantee of future customer behaviour.
            It should be interpreted together with business knowledge and
            customer specific information.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # -----------------------------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------------------------

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Important:</strong>
            This application is an educational and portfolio machine learning
            project. Predictions are generated from historical customer data
            and should not be treated as certainty. The system does not
            replace business analysis or professional decision making.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    action_col_1, action_col_2, action_col_3 = st.columns(
        [1, 1, 1],
        vertical_alignment="center",
    )

    with action_col_1:
        if st.button(
            "Start New Assessment",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.pop("prediction_result", None)
            st.session_state.pop("churn_probability", None)
            st.switch_page("app.py")

    with action_col_2:
        if st.button(
            "Return to Assessment",
            use_container_width=True,
        ):
            st.switch_page("app.py")

    with action_col_3:
        if st.button(
            "View Model Performance",
            use_container_width=True,
        ):
            st.switch_page("pages/3_Model_Performance.py")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Prediction Results page."""

    render_navigation(PAGE_RESULTS)

    prediction_result = st.session_state.get("prediction_result")
    churn_probability = st.session_state.get("churn_probability")

    if prediction_result is None or churn_probability is None:
        render_no_prediction_state()
        return

    try:
        probability = float(churn_probability)
    except (TypeError, ValueError):
        st.error(
            "The stored prediction probability is invalid. "
            "Please start a new assessment."
        )
        return

    render_dashboard(
        prediction_result=prediction_result,
        churn_probability=probability,
    )


if __name__ == "__main__":
    main()
