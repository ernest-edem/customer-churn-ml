from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any

import pandas as pd
import streamlit as st

from ml_system.persistence import load_model


# ============================================================================
# Streamlit configuration
# ============================================================================

# Disable Streamlit's automatically generated multipage sidebar navigation.
# Custom navigation is provided by the application header.
st.set_option("client.showSidebarNavigation", False)

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# Application constants
# ============================================================================

MODEL_PATH = Path("models/model.joblib")

PAGE_ASSESSMENT = "assessment"
PAGE_RESULTS = "results"
PAGE_MODEL = "model"

TOTAL_STEPS = 4


# ============================================================================
# HTML helper
# ============================================================================

def render_html(markup: str) -> None:
    """Render dedented HTML safely through Streamlit Markdown."""

    st.markdown(
        dedent(markup).strip(),
        unsafe_allow_html=True,
    )


# ============================================================================
# Global styling
# ============================================================================

def inject_global_styles() -> None:
    """Apply application-wide layout and visual styling."""

    render_html(
        """
        <style>
        /* =================================================================
           Design tokens
        ================================================================= */

        :root {
            --surface: #ffffff;
            --surface-muted: #f8fafc;
            --surface-soft: #f1f5f9;

            --border: #e2e8f0;
            --border-strong: #cbd5e1;

            --text-primary: #0f172a;
            --text-secondary: #64748b;
            --text-tertiary: #94a3b8;

            --accent: #2563eb;
            --accent-hover: #1d4ed8;
            --accent-soft: #eff6ff;

            --success: #15803d;
            --success-soft: #f0fdf4;

            --radius-xl: 22px;
            --radius-lg: 18px;
            --radius-md: 14px;
            --radius-sm: 10px;

            --shadow-sm:
                0 1px 2px rgba(15, 23, 42, 0.04);

            --shadow-md:
                0 1px 2px rgba(15, 23, 42, 0.04),
                0 10px 30px rgba(15, 23, 42, 0.06);

            --shadow-lg:
                0 2px 4px rgba(15, 23, 42, 0.04),
                0 18px 48px rgba(15, 23, 42, 0.10);
        }


        /* =================================================================
           Streamlit chrome
        ================================================================= */

        header[data-testid="stHeader"] {
            background: transparent;
            box-shadow: none;
        }

        header[data-testid="stHeader"] > div {
            background: transparent;
        }

        /*
           Keep the Streamlit toolbar available.

           This preserves the three-dot menu for settings, appearance,
           recording, and other Streamlit controls.
        */
        header[data-testid="stHeader"] [data-testid="stToolbar"] {
            visibility: visible;
            right: 0.75rem;
        }

        footer {
            visibility: hidden;
            height: 0;
        }


        /* =================================================================
           Application background
        ================================================================= */

        .stApp {
            background:
                radial-gradient(
                    1100px 520px at 50% -8%,
                    #f1f5f9 0%,
                    transparent 62%
                ),
                #ffffff;
        }

        .block-container {
            width: 100%;
            max-width: 1200px;

            margin-left: auto;
            margin-right: auto;

            padding-top: 1.25rem;
            padding-bottom: 3.5rem;
        }


        /* =================================================================
           Shared application navigation
        ================================================================= */

        .custom-navigation {
            width: 100%;
            min-height: 72px;

            display: flex;
            align-items: center;
        }

        .app-header {
            display: flex;
            align-items: center;

            gap: 0.75rem;

            white-space: nowrap;

            padding: 0.4rem 0;
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
            width: 1px;
            height: 34px;

            background: var(--border);

            margin: 0 0.35rem;
        }

        .navigation-wrapper {
            width: 100%;

            display: flex;
            align-items: center;
            justify-content: center;
        }


        /* =================================================================
           Buttons
        ================================================================= */

        .custom-navigation button,
        div[data-testid="stHorizontalBlock"] button {
            min-height: 40px;

            border-radius: var(--radius-sm);

            font-weight: 600;

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
            box-shadow:
                0 6px 18px rgba(37, 99, 235, 0.20);
        }

        button[kind="primary"]:hover {
            box-shadow:
                0 10px 26px rgba(37, 99, 235, 0.28);
        }


        /* =================================================================
           Dividers
        ================================================================= */

        hr,
        div[data-testid="stDivider"] hr {
            margin-top: 0.55rem;
            margin-bottom: 2rem;

            border-color: var(--border);
        }


        /* =================================================================
           Page introduction
        ================================================================= */

        .page-header {
            width: 100%;
            max-width: 860px;

            margin-left: auto;
            margin-right: auto;

            text-align: center;
        }

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

            max-width: 760px;

            margin: 0.85rem auto 0;

            font-size: 1rem;
            line-height: 1.7;
        }


        /* =================================================================
           Assessment overview
        ================================================================= */

        .overview-grid {
            width: 100%;
            max-width: 1080px;

            margin: 2.2rem auto 0;
        }

        .overview-card {
            width: 100%;

            height: 238px;
            min-height: 238px;

            box-sizing: border-box;

            display: flex;
            flex-direction: column;
            justify-content: center;

            border: 1px solid var(--border);
            border-radius: var(--radius-lg);

            background: var(--surface);

            padding: 2rem 2.25rem;

            box-shadow: var(--shadow-md);

            transition:
                transform 200ms ease,
                box-shadow 200ms ease,
                border-color 200ms ease;
        }

        .overview-card:hover {
            transform: translateY(-3px);

            border-color: var(--border-strong);

            box-shadow: var(--shadow-lg);
        }

        .overview-card-title {
            color: var(--text-primary);

            font-size: 1.42rem;
            font-weight: 750;

            letter-spacing: -0.025em;
            line-height: 1.25;

            margin-bottom: 0.8rem;
        }

        .overview-card-text {
            color: var(--text-secondary);

            font-size: 0.96rem;
            line-height: 1.72;

            margin: 0;
        }

        .model-list {
            display: flex;
            flex-direction: column;

            gap: 0.7rem;

            margin-top: 0.1rem;
        }

        .model-item {
            display: flex;
            align-items: center;

            gap: 0.65rem;

            color: var(--text-secondary);

            font-size: 0.96rem;
            line-height: 1.45;
        }

        .model-item::before {
            content: "";

            width: 7px;
            height: 7px;

            flex: 0 0 7px;

            border-radius: 50%;

            background: var(--accent);
        }


        /* =================================================================
           Assessment action section
        ================================================================= */

        .assessment-section {
            width: 100%;
            max-width: 1080px;

            margin: 2.4rem auto 0;
        }

        .assessment-action-card {
            border: 1px solid var(--border);

            border-radius: var(--radius-lg);

            background: var(--surface);

            padding: 1.65rem 1.8rem;

            box-shadow: var(--shadow-sm);
        }

        .section-title {
            color: var(--text-primary);

            font-size: 1.28rem;
            font-weight: 750;

            letter-spacing: -0.022em;
            line-height: 1.3;

            margin: 0;
        }

        .section-description {
            color: var(--text-secondary);

            max-width: 820px;

            margin-top: 0.45rem;

            font-size: 0.92rem;
            line-height: 1.65;
        }

        .assessment-action-description {
            color: var(--text-secondary);

            font-size: 0.9rem;
            line-height: 1.6;

            margin-top: 0.35rem;
        }


        /* =================================================================
           Assessment dialog
        ================================================================= */

        div[role="dialog"] {
            border-radius: var(--radius-xl);
        }

        div[role="dialog"] > div {
            border-radius: var(--radius-xl);
        }

        .step-header {
            text-align: center;

            margin-bottom: 1rem;
        }

        .step-label {
            color: var(--accent);

            font-size: 0.72rem;
            font-weight: 750;

            letter-spacing: 0.09em;
            line-height: 1.2;

            text-transform: uppercase;
        }

        .step-title {
            color: var(--text-primary);

            font-size: 1.48rem;
            font-weight: 750;

            letter-spacing: -0.025em;
            line-height: 1.25;

            margin-top: 0.35rem;
        }

        .step-spacing {
            height: 0.35rem;
        }

        .dialog-section {
            border: 1px solid var(--border);

            border-radius: var(--radius-md);

            background: var(--surface-muted);

            padding: 1rem 1.1rem;

            margin-bottom: 1rem;
        }


        /* =================================================================
           Review cards
        ================================================================= */

        .review-section-title {
            color: var(--text-primary);

            font-size: 0.92rem;
            font-weight: 750;

            letter-spacing: -0.01em;

            margin: 0 0 0.65rem;
        }

        .review-card {
            width: 100%;

            min-height: 78px;

            box-sizing: border-box;

            display: flex;
            flex-direction: column;
            justify-content: center;

            border: 1px solid var(--border);

            border-radius: var(--radius-sm);

            background: var(--surface);

            padding: 0.85rem 1rem;

            margin-bottom: 0.65rem;
        }

        .review-label {
            color: var(--text-tertiary);

            font-size: 0.68rem;
            font-weight: 700;

            letter-spacing: 0.06em;
            line-height: 1.3;

            text-transform: uppercase;
        }

        .review-value {
            color: var(--text-primary);

            font-size: 0.88rem;
            font-weight: 650;

            line-height: 1.4;

            margin-top: 0.25rem;

            overflow-wrap: anywhere;
        }


        /* =================================================================
           Progress
        ================================================================= */

        .stProgress {
            margin-top: 0.4rem;
            margin-bottom: 1rem;
        }

        .stProgress > div > div {
            border-radius: 999px;
        }


        /* =================================================================
           Footer
        ================================================================= */

        .footer-note {
            width: 100%;
            max-width: 820px;

            margin: 2.25rem auto 0;

            color: var(--text-tertiary);

            font-size: 0.77rem;
            line-height: 1.6;

            text-align: center;
        }


        /* =================================================================
           Responsive layout
        ================================================================= */

        @media (max-width: 1000px) {
            .block-container {
                max-width: 100%;

                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }

            .overview-grid,
            .assessment-section {
                max-width: 100%;
            }
        }

        @media (max-width: 900px) {
            .page-title {
                font-size: 1.95rem;
            }

            .overview-card {
                height: auto;
                min-height: 210px;
            }

            .custom-navigation {
                min-height: 68px;
            }
        }

        @media (max-width: 640px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

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
                font-size: 1.65rem;
            }

            .page-description {
                font-size: 0.92rem;
            }

            .overview-card {
                padding: 1.5rem;
            }

            .overview-card-title {
                font-size: 1.25rem;
            }

            .assessment-action-card {
                padding: 1.35rem;
            }

            .step-title {
                font-size: 1.3rem;
            }
        }


        /* =================================================================
           Dark mode
        ================================================================= */

        @media (prefers-color-scheme: dark) {
            :root {
                --surface: #0f172a;
                --surface-muted: #111827;
                --surface-soft: #1e293b;

                --border: #334155;
                --border-strong: #475569;

                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --text-tertiary: #64748b;

                --accent: #60a5fa;
                --accent-hover: #93c5fd;
                --accent-soft: rgba(96, 165, 250, 0.12);

                --shadow-md: none;
                --shadow-lg:
                    0 18px 48px rgba(0, 0, 0, 0.45);
            }

            .stApp {
                background:
                    radial-gradient(
                        1100px 520px at 50% -8%,
                        #0b1220 0%,
                        transparent 62%
                    ),
                    #020617;
            }

            .brand,
            .page-title,
            .section-title,
            .overview-card-title,
            .step-title,
            .review-section-title,
            .review-value {
                color: var(--text-primary);
            }

            .brand-subtitle,
            .page-description,
            .overview-card-text,
            .model-item,
            .section-description,
            .assessment-action-description,
            .review-label,
            .footer-note {
                color: var(--text-secondary);
            }

            .navigation-divider {
                background: var(--border);
            }

            .overview-card,
            .assessment-action-card {
                border-color: var(--border);
                background: var(--surface);
                box-shadow: none;
            }

            .overview-card:hover {
                border-color: var(--border-strong);
                box-shadow: var(--shadow-lg);
            }

            .review-card {
                border-color: var(--border);
                background: var(--surface-muted);
            }

            button[kind="secondary"] {
                background: var(--surface) !important;
                border-color: var(--border) !important;
                color: var(--text-secondary) !important;
            }

            button[kind="secondary"]:hover {
                border-color: var(--accent) !important;
                color: var(--accent) !important;
                background: var(--accent-soft) !important;
            }

            .page-eyebrow {
                border-color: rgba(96, 165, 250, 0.22);
            }
        }
        </style>
        """
    )


# ============================================================================
# Model loading
# ============================================================================

@st.cache_resource
def get_model():
    """Load and cache the persisted machine learning model."""

    return load_model(MODEL_PATH)


# ============================================================================
# Navigation
# ============================================================================

def render_navigation(active_page: str) -> None:
    """Render the shared application navigation."""

    render_html(
        """
        <div class="custom-navigation">
        """
    )

    header_left, header_navigation = st.columns(
        [2.15, 1],
        vertical_alignment="center",
    )

    with header_left:
        render_html(
            """
            <div class="app-header">
                <div>
                    <div class="brand">
                        Customer Churn
                        <span class="brand-accent">AI</span>
                    </div>

                    <div class="brand-subtitle">
                        Machine Learning Prediction System
                    </div>
                </div>

                <div class="navigation-divider"></div>
            </div>
            """
        )

    with header_navigation:
        render_html(
            """
            <div class="navigation-wrapper">
            """
        )

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
                st.switch_page(
                    "pages/2_Prediction_Results.py"
                )

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
                st.switch_page(
                    "pages/3_Model_Performance.py"
                )

        render_html(
            """
            </div>
            """
        )

    render_html(
        """
        </div>
        """
    )

    st.divider()


# ============================================================================
# Assessment state
# ============================================================================

def initialize_form_state() -> None:
    """Initialize assessment state without overwriting existing values."""

    defaults: dict[str, Any] = {
        "assessment_step": 1,

        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",

        "tenure": 1,

        "PhoneService": "Yes",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",

        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",

        "MonthlyCharges": 70.0,
        "TotalCharges": 70.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_assessment() -> None:
    """Reset assessment inputs and prediction state."""

    assessment_keys = [
        "assessment_step",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "prediction_result",
        "churn_probability",
    ]

    for key in assessment_keys:
        st.session_state.pop(key, None)

    initialize_form_state()


# ============================================================================
# Input preparation
# ============================================================================

def build_input_dataframe() -> pd.DataFrame:
    """Build the model input DataFrame using the original model schema."""

    return pd.DataFrame(
        [
            {
                "gender": st.session_state.gender,
                "SeniorCitizen": st.session_state.SeniorCitizen,
                "Partner": st.session_state.Partner,
                "Dependents": st.session_state.Dependents,
                "tenure": st.session_state.tenure,
                "PhoneService": st.session_state.PhoneService,
                "MultipleLines": st.session_state.MultipleLines,
                "InternetService": st.session_state.InternetService,
                "OnlineSecurity": st.session_state.OnlineSecurity,
                "OnlineBackup": st.session_state.OnlineBackup,
                "DeviceProtection": st.session_state.DeviceProtection,
                "TechSupport": st.session_state.TechSupport,
                "StreamingTV": st.session_state.StreamingTV,
                "StreamingMovies": st.session_state.StreamingMovies,
                "Contract": st.session_state.Contract,
                "PaperlessBilling": st.session_state.PaperlessBilling,
                "PaymentMethod": st.session_state.PaymentMethod,
                "MonthlyCharges": st.session_state.MonthlyCharges,
                "TotalCharges": st.session_state.TotalCharges,
            }
        ]
    )


# ============================================================================
# Assessment steps
# ============================================================================

def render_step_header(step_number: int, title: str) -> None:
    """Render a consistent assessment step header."""

    render_html(
        f"""
        <div class="step-header">
            <div class="step-label">
                Step {step_number} of {TOTAL_STEPS}
            </div>

            <div class="step-title">
                {title}
            </div>
        </div>
        """
    )

    st.progress(
        step_number / TOTAL_STEPS,
        text=f"Step {step_number} of {TOTAL_STEPS}",
    )

    render_html(
        """
        <div class="step-spacing"></div>
        """
    )


def render_demographics_step() -> None:
    """Render customer demographic information."""

    render_step_header(1, "Customer Demographics")

    column_1, column_2 = st.columns(
        2,
        gap="large",
    )

    with column_1:
        st.selectbox(
            "Gender",
            options=["Female", "Male"],
            key="gender",
        )

        st.selectbox(
            "Senior Citizen",
            options=[0, 1],
            format_func=lambda value: (
                "No" if value == 0 else "Yes"
            ),
            key="SeniorCitizen",
        )

    with column_2:
        st.selectbox(
            "Partner",
            options=["Yes", "No"],
            key="Partner",
        )

        st.selectbox(
            "Dependents",
            options=["Yes", "No"],
            key="Dependents",
        )

    st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        step=1,
        key="tenure",
        help=(
            "Number of months the customer has been "
            "with the company."
        ),
    )


def render_services_step() -> None:
    """Render customer service information."""

    render_step_header(2, "Services")

    column_1, column_2 = st.columns(
        2,
        gap="large",
    )

    with column_1:
        st.selectbox(
            "Phone Service",
            options=["Yes", "No"],
            key="PhoneService",
        )

        st.selectbox(
            "Multiple Lines",
            options=[
                "No phone service",
                "No",
                "Yes",
            ],
            key="MultipleLines",
        )

        st.selectbox(
            "Internet Service",
            options=[
                "DSL",
                "Fiber optic",
                "No",
            ],
            key="InternetService",
        )

        st.selectbox(
            "Online Security",
            options=[
                "No",
                "Yes",
                "No internet service",
            ],
            key="OnlineSecurity",
        )

        st.selectbox(
            "Online Backup",
            options=[
                "No",
                "Yes",
                "No internet service",
            ],
            key="OnlineBackup",
        )

    with column_2:
        st.selectbox(
            "Device Protection",
            options=[
                "No",
                "Yes",
                "No internet service",
            ],
            key="DeviceProtection",
        )

        st.selectbox(
            "Technical Support",
            options=[
                "No",
                "Yes",
                "No internet service",
            ],
            key="TechSupport",
        )

        st.selectbox(
            "Streaming TV",
            options=[
                "No",
                "Yes",
                "No internet service",
            ],
            key="StreamingTV",
        )

        st.selectbox(
            "Streaming Movies",
            options=[
                "No",
                "Yes",
                "No internet service",
            ],
            key="StreamingMovies",
        )


def render_account_step() -> None:
    """Render customer account and billing information."""

    render_step_header(3, "Account & Billing")

    column_1, column_2 = st.columns(
        2,
        gap="large",
    )

    with column_1:
        st.selectbox(
            "Contract",
            options=[
                "Month-to-month",
                "One year",
                "Two year",
            ],
            key="Contract",
        )

        st.selectbox(
            "Paperless Billing",
            options=["Yes", "No"],
            key="PaperlessBilling",
        )

    with column_2:
        st.selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            key="PaymentMethod",
        )

    st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=1000.0,
        step=0.01,
        format="%.2f",
        key="MonthlyCharges",
    )

    st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        step=0.01,
        format="%.2f",
        key="TotalCharges",
    )


def render_review_step() -> None:
    """Render a final review before prediction."""

    render_step_header(4, "Review Assessment")

    review_sections = [
        (
            "Customer",
            [
                ("Gender", st.session_state.gender),
                (
                    "Senior Citizen",
                    "Yes"
                    if st.session_state.SeniorCitizen
                    else "No",
                ),
                ("Partner", st.session_state.Partner),
                (
                    "Dependents",
                    st.session_state.Dependents,
                ),
                (
                    "Tenure",
                    f"{st.session_state.tenure} months",
                ),
            ],
        ),
        (
            "Services",
            [
                (
                    "Phone Service",
                    st.session_state.PhoneService,
                ),
                (
                    "Multiple Lines",
                    st.session_state.MultipleLines,
                ),
                (
                    "Internet Service",
                    st.session_state.InternetService,
                ),
                (
                    "Online Security",
                    st.session_state.OnlineSecurity,
                ),
                (
                    "Online Backup",
                    st.session_state.OnlineBackup,
                ),
                (
                    "Device Protection",
                    st.session_state.DeviceProtection,
                ),
                (
                    "Technical Support",
                    st.session_state.TechSupport,
                ),
                (
                    "Streaming TV",
                    st.session_state.StreamingTV,
                ),
                (
                    "Streaming Movies",
                    st.session_state.StreamingMovies,
                ),
            ],
        ),
        (
            "Account",
            [
                (
                    "Contract",
                    st.session_state.Contract,
                ),
                (
                    "Paperless Billing",
                    st.session_state.PaperlessBilling,
                ),
                (
                    "Payment Method",
                    st.session_state.PaymentMethod,
                ),
                (
                    "Monthly Charges",
                    f"${st.session_state.MonthlyCharges:.2f}",
                ),
                (
                    "Total Charges",
                    f"${st.session_state.TotalCharges:.2f}",
                ),
            ],
        ),
    ]

    for section_title, values in review_sections:
        render_html(
            f"""
            <div class="review-section-title">
                {section_title}
            </div>
            """
        )

        review_columns = st.columns(
            2,
            gap="medium",
        )

        for index, (label, value) in enumerate(values):
            with review_columns[index % 2]:
                render_html(
                    f"""
                    <div class="review-card">
                        <div class="review-label">
                            {label}
                        </div>

                        <div class="review-value">
                            {value}
                        </div>
                    </div>
                    """
                )


# ============================================================================
# Prediction
# ============================================================================

def generate_prediction() -> bool:
    """Generate and store the customer churn prediction."""

    try:
        model = get_model()
        input_data = build_input_dataframe()

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.session_state.prediction_result = prediction
        st.session_state.churn_probability = float(
            probabilities[1]
        )

        return True

    except Exception as exc:
        st.error(
            "The prediction could not be generated. "
            "Please review the assessment and try again."
        )

        st.exception(exc)

        return False


# ============================================================================
# Assessment dialog
# ============================================================================

@st.dialog("Customer Churn Assessment", width="large")
def render_assessment_dialog() -> None:
    """Render the four-step customer assessment dialog."""

    step = st.session_state.get(
        "assessment_step",
        1,
    )

    if step == 1:
        render_demographics_step()

    elif step == 2:
        render_services_step()

    elif step == 3:
        render_account_step()

    elif step == 4:
        render_review_step()

    render_html(
        """
        <div style="height: 0.5rem;"></div>
        """
    )

    previous_column, spacer_column, next_column = st.columns(
        [1, 1.4, 1],
        vertical_alignment="center",
    )

    with previous_column:
        if step > 1:
            if st.button(
                "Back",
                use_container_width=True,
            ):
                st.session_state.assessment_step = step - 1
                st.rerun()

    with spacer_column:
        st.empty()

    with next_column:
        if step < TOTAL_STEPS:
            if st.button(
                "Continue",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.assessment_step = step + 1
                st.rerun()

        else:
            if st.button(
                "Generate Prediction",
                type="primary",
                use_container_width=True,
            ):
                if generate_prediction():
                    st.session_state.assessment_step = 1

                    st.switch_page(
                        "pages/2_Prediction_Results.py"
                    )


# ============================================================================
# Assessment landing page
# ============================================================================

def render_assessment_page() -> None:
    """Render the centered assessment landing page."""

    render_html(
        """
        <div class="page-header">
            <div class="page-eyebrow">
                Customer Retention Intelligence
            </div>

            <h1 class="page-title">
                Customer Churn Prediction
            </h1>

            <div class="page-description">
                Use the trained machine learning model to estimate whether
                a customer is likely to churn based on their profile,
                services, and account information.
            </div>
        </div>
        """
    )

    # ------------------------------------------------------------------------
    # Overview cards
    # ------------------------------------------------------------------------

    render_html(
        """
        <div class="overview-grid">
        """
    )

    overview_left, overview_right = st.columns(
        [1, 1],
        gap="large",
        vertical_alignment="top",
    )

    with overview_left:
        render_html(
            """
            <div class="overview-card">
                <div class="overview-card-title">
                    Customer Churn Assessment
                </div>

                <p class="overview-card-text">
                    Complete a guided four-step assessment using customer
                    profile, services, and account information. The trained
                    model evaluates the submitted information without
                    retraining.
                </p>
            </div>
            """
        )

    with overview_right:
        render_html(
            """
            <div class="overview-card">
                <div class="overview-card-title">
                    Model
                </div>

                <div class="model-list">
                    <div class="model-item">
                        Gradient Boosting
                    </div>

                    <div class="model-item">
                        Binary Classification
                    </div>

                    <div class="model-item">
                        19 Customer Features
                    </div>
                </div>
            </div>
            """
        )

    render_html(
        """
        </div>
        """
    )

    # ------------------------------------------------------------------------
    # Assessment action
    # ------------------------------------------------------------------------

    render_html(
        """
        <div class="assessment-section">
            <div class="assessment-action-card">
                <div class="section-title">
                    Start a Customer Assessment
                </div>

                <div class="assessment-action-description">
                    The assessment covers customer demographics, services,
                    account information, and a final review before generating
                    the prediction.
                </div>
            </div>
        </div>
        """
    )

    st.write("")

    action_left, action_center, action_right = st.columns(
        [1, 1, 1],
        vertical_alignment="center",
    )

    with action_center:
        if st.button(
            "Start Customer Assessment",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.assessment_step = 1
            render_assessment_dialog()

    # ------------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------------

    render_html(
        """
        <div class="footer-note">
            Predictions are generated from the persisted machine learning
            pipeline. This application is intended for educational and
            portfolio demonstration purposes.
        </div>
        """
    )


# ============================================================================
# Application entry point
# ============================================================================

def main() -> None:
    """Run the Customer Churn AI application."""

    inject_global_styles()
    initialize_form_state()

    render_navigation(PAGE_ASSESSMENT)
    render_assessment_page()


if __name__ == "__main__":
    main()