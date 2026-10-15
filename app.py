"""
Drifto — ML Model Drift Monitoring System
Run with: streamlit run app.py

Educational demonstration:
- Trains a Random Forest classifier on a selected tabular classification dataset.
- Supports built-in scikit-learn datasets or custom CSV uploads with a chosen target column.
- Creates a reproducible simulated production sample with controlled feature shifts.
- Compares reference and simulated feature distributions.
- Evaluates predictions on the simulated sample using its retained labels.

This is a local simulation, not a live production monitoring service.
Feature drift alone does not prove that model performance has degraded.
"""
from __future__ import annotations

# pylint: disable=no-member

import io
from typing import cast

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data import BUILTIN_DATASETS, infer_target_column, load_dataset as load_source_dataset

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - allows app to run when Plotly is absent.
    import altair as alt

    PLOTLY_AVAILABLE = False

    class _AltairFigure:
        def __init__(self, chart):
            """Wrap an Altair chart so it behaves like a Plotly figure."""
            self.chart = chart

        def update_layout(self, **_unused_kwargs):
            """Ignore layout updates because they are not needed for the fallback."""
            del _unused_kwargs
            return self

        def __getattr__(self, name):
            """Delegate unknown attributes to the wrapped Altair chart."""
            return getattr(self.chart, name)

    class _FallbackPX:
        # pylint: disable=disallowed-name
        @staticmethod
        def bar(data_frame, x=None, y=None, color=None, _orientation="v", **_unused_kwargs):
            """Create a bar chart fallback when Plotly is unavailable."""
            del _orientation
            del _unused_kwargs
            enc = {}
            if x is not None:
                enc["x"] = alt.X(x)
            if y is not None:
                enc["y"] = alt.Y(y)
            if color is not None:
                enc["color"] = alt.Color(color)
            chart = alt.Chart(data_frame).mark_bar().encode(**enc)
            return _AltairFigure(chart)

        @staticmethod
        def box(data_frame, x=None, y=None, color=None, **_unused_kwargs):
            """Create a box plot fallback when Plotly is unavailable."""
            del _unused_kwargs
            enc = {}
            if x is not None:
                enc["x"] = alt.X(x)
            if y is not None:
                enc["y"] = alt.Y(y)
            if color is not None:
                enc["color"] = alt.Color(color)
            chart = alt.Chart(data_frame).mark_boxplot(
                extent="min-max").encode(**enc)
            return _AltairFigure(chart)

    px = _FallbackPX()


def render_chart(chart_figure, use_container_width=True):
    """Render a plot using Plotly when available; otherwise fall back to Altair."""
    if PLOTLY_AVAILABLE:
        st.plotly_chart(chart_figure, use_container_width=use_container_width)
        return
    chart = chart_figure.chart if hasattr(
        chart_figure, "chart") else chart_figure
    st.altair_chart(chart, use_container_width=use_container_width)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drifto | ML Model Drift Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: Inter, sans-serif; }
    .stApp { background: #0b1018; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101d2c 0%, #0a1420 100%);
        border-right: 1px solid rgba(148,163,184,.15);
    }
    .hero {
        background: linear-gradient(135deg, #183454 0%, #101c2c 72%);
        border: 1px solid rgba(96,165,250,.25);
        border-radius: 18px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.2rem;
    }
    .eyebrow {
        display:inline-block; margin:0 .35rem .75rem 0; padding:.25rem .65rem;
        color:#7dd3fc; background:rgba(56,189,248,.10);
        border:1px solid rgba(56,189,248,.28); border-radius:7px;
        font-size:.76rem; font-weight:600;
    }
    .hero h1 { color:#eff6ff; margin:0 0 .45rem 0; font-size:2.2rem; }
    .hero p { color:#b6c9df; margin:0; line-height:1.65; }
    .metric {
        background:linear-gradient(145deg,#17283d,#111e30);
        border:1px solid rgba(148,163,184,.16); border-radius:14px;
        padding:1.05rem 1rem; text-align:center; min-height:105px;
    }
    .metric-label { color:#9db4cd; font-size:.75rem; text-transform:uppercase;
                    letter-spacing:.08em; font-weight:600; }
    .metric-value { color:#38bdf8; font-size:1.9rem; font-weight:700; margin-top:.35rem; }
    .section-title {
        color:#dbeafe; font-size:1.2rem; font-weight:700;
        padding-bottom:.5rem; margin:1.2rem 0 .75rem;
        border-bottom:1px solid rgba(56,189,248,.3);
    }
    .note {
        padding:.9rem 1rem; border-radius:10px; line-height:1.55;
        background:rgba(56,189,248,.08); border:1px solid rgba(56,189,248,.24);
        color:#c7e8fb;
    }
    .warning {
        padding:.9rem 1rem; border-radius:10px; line-height:1.55;
        background:rgba(245,158,11,.09); border:1px solid rgba(245,158,11,.32);
        color:#fde68a;
    }
    .success {
        padding:.9rem 1rem; border-radius:10px; line-height:1.55;
        background:rgba(34,197,94,.08); border:1px solid rgba(34,197,94,.28);
        color:#bbf7d0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data and model helpers
# ---------------------------------------------------------------------------


@st.cache_data
def load_selected_dataset(
    source_choice: str,
    selected_dataset_name: str,
    _uploaded_file_name: str | None,
    uploaded_file_bytes: bytes | None,
    target_name: str | None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load either a built-in sklearn dataset or a user-supplied CSV file."""
    if source_choice == "csv":
        if uploaded_file_bytes is None:
            raise ValueError(
                "Please upload a CSV file before running the model.")

        dataframe = pd.read_csv(io.BytesIO(uploaded_file_bytes))
        if target_name is None:
            target_name = infer_target_column(dataframe)
        if target_name is None or target_name not in dataframe.columns:
            raise ValueError(
                "Could not automatically infer a target column. Please select the target column manually."
            )

        selected_target = target_name
        y: pd.Series = dataframe.loc[:, selected_target].copy()
        X: pd.DataFrame = dataframe.drop(columns=[selected_target]).copy()
        feature_names = list(X.columns)
        class_names = [str(value) for value in sorted(pd.unique(y), key=str)]
        return X, y, feature_names, class_names

    resolved_name = (selected_dataset_name or "breast_cancer").lower()
    if resolved_name not in BUILTIN_DATASETS:
        raise ValueError(f"Unsupported dataset '{selected_dataset_name}'.")

    dataset_result = load_source_dataset(
        dataset_name=resolved_name, return_metadata=True)
    if len(dataset_result) != 4:
        raise TypeError(
            "Expected dataset metadata tuple with feature names and class names.")
    typed_dataset = cast(
        tuple[pd.DataFrame, pd.Series, list[str], list[str]],
        dataset_result,
    )
    return typed_dataset


def class_name_map(value_sequence):
    """Return a label mapping for prediction output tables."""
    unique_values = sorted(pd.unique(value_sequence), key=str)
    return {value: str(value) for value in unique_values}


@st.cache_resource
def train_baseline(
    selected_dataset_name: str = "breast_cancer",
    source_name: str = "builtin",
    _uploaded_file_name: str | None = None,
    uploaded_file_bytes: bytes | None = None,
    target_column_name: str | None = None,
    random_seed: int = 42,
):
    X, y, feature_names, class_names = load_selected_dataset(
        source_choice=source_name,
        selected_dataset_name=selected_dataset_name,
        _uploaded_file_name=_uploaded_file_name,
        uploaded_file_bytes=uploaded_file_bytes,
        target_name=target_column_name,
    )
    X_train, reference_X, y_train, reference_y = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=250, random_state=random_seed, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    predictions = model.predict(reference_X)
    metrics = calculate_metrics(reference_y, predictions)
    return (
        model,
        X_train,
        reference_X,
        y_train,
        reference_y,
        predictions,
        metrics,
        feature_names,
        class_names,
    )


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division="warn"),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division="warn"),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division="warn"),
    }


def simulate_production(reference: pd.DataFrame, shift_magnitude: float, random_seed: int):
    """Create reproducible simulated data by shifting selected numeric features."""
    rng = np.random.default_rng(random_seed)
    simulated = reference.copy().reset_index(drop=True)
    columns = list(simulated.columns)
    count = max(1, min(6, len(columns) // 8))
    shifted_columns = columns[:count]

    for feature_name in shifted_columns:
        ref_std = float(reference[feature_name].std(ddof=0))
        if not np.isfinite(ref_std) or ref_std == 0:
            continue
        simulated[feature_name] = simulated[feature_name] + \
            shift_magnitude * ref_std
        simulated[feature_name] = simulated[feature_name] + rng.normal(
            0, 0.12 * ref_std, size=len(simulated)
        )
    return simulated, shifted_columns


def calculate_drift(reference: pd.DataFrame, current: pd.DataFrame, threshold: float):
    rows = []
    for feature_name in reference.columns:
        ref_mean = float(reference[feature_name].mean())
        new_mean = float(current[feature_name].mean())
        ref_std = float(reference[feature_name].std(ddof=0))
        score = abs(new_mean - ref_mean) / ref_std if ref_std > 1e-12 else 0.0
        rows.append({
            "Feature": feature_name,
            "Reference mean": ref_mean,
            "Simulated mean": new_mean,
            "Reference std": ref_std,
            "Mean difference": new_mean - ref_mean,
            "Drift score": score,
            "Drift detected": bool(score >= threshold),
        })
    return pd.DataFrame(rows).sort_values("Drift score", ascending=False).reset_index(drop=True)


with st.sidebar:
    st.markdown("## 🌊 Drifto")
    st.caption("ML Model Drift Monitoring")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Baseline Metrics", "Prediction Distribution",
         "Feature Drift Analysis", "Drift Warnings", "About"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Dataset settings**")
    dataset_source = st.radio(
        "Dataset source",
        ["builtin", "csv"],
        format_func=lambda choice: "Built-in sklearn dataset" if choice == "builtin" else "CSV upload",
        horizontal=False,
    )

    if dataset_source == "builtin":
        dataset_name = st.selectbox(
            "Built-in dataset",
            options=sorted(BUILTIN_DATASETS.keys()),
            index=0,
        )
        uploaded_file = None
        csv_file_bytes = None
        target_column = None
    else:
        dataset_name = "breast_cancer"
        uploaded_file = st.file_uploader("CSV dataset", type=["csv"])
        csv_file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        target_column = None
        if uploaded_file is not None and csv_file_bytes is not None:
            preview = pd.read_csv(io.BytesIO(csv_file_bytes))
            candidates = [
                column for column in preview.columns
                if preview[column].nunique(dropna=True) > 1
                and len(preview[column].unique()) <= min(20, len(preview))
            ]
            if candidates:
                target_column = st.selectbox(
                    "Target column",
                    options=candidates,
                    index=0,
                )
            else:
                st.warning(
                    "No suitable target column was found. Please upload a CSV with a categorical target column.")
                target_column = None

    st.divider()
    st.markdown("**Simulation settings**")
    shift_strength = st.slider(
        "Feature shift strength (× standard deviation)",
        min_value=0.0, max_value=2.5, value=1.0, step=0.25,
        help="Controls the artificial shift applied to selected features.",
    )
    drift_threshold = st.slider(
        "Drift score threshold",
        min_value=0.25, max_value=2.5, value=0.75, step=0.25,
        help="A feature is flagged when |mean difference| / reference standard deviation meets this threshold.",
    )
    seed = st.number_input("Simulation seed", min_value=0,
                           max_value=9999, value=42, step=1)
    st.caption(
        "This dashboard can monitor any tabular classification dataset or CSV upload.")

try:
    (
        baseline_model,
        baseline_X_train,
        X_ref,
        baseline_y_train,
        y_ref,
        ref_predictions,
        baseline_metrics,
        features,
        target_names,
    ) = train_baseline(
        selected_dataset_name=dataset_name,
        source_name=dataset_source,
        _uploaded_file_name=getattr(uploaded_file, "name", None),
        uploaded_file_bytes=csv_file_bytes,
        target_column_name=target_column,
        random_seed=int(seed),
    )
    reference_data = pd.DataFrame(X_ref).reset_index(drop=True).copy()
except ValueError as exc:
    st.error(f"Dataset configuration error: {exc}")
    st.stop()

new_data, intentionally_shifted = simulate_production(
    reference_data, shift_strength, int(seed))
new_predictions = baseline_model.predict(new_data)
new_metrics = calculate_metrics(y_ref, new_predictions)
drift_df = calculate_drift(reference_data, new_data, drift_threshold)
drifted_df = drift_df[drift_df["Drift detected"]]

label_map = class_name_map(y_ref)
reference_pred_series = pd.Series(ref_predictions).map(label_map)
new_pred_series = pd.Series(new_predictions).map(label_map)
reference_distribution = reference_pred_series.value_counts(
).rename_axis("Class").reset_index(name="Count")
new_distribution = new_pred_series.value_counts(
).rename_axis("Class").reset_index(name="Count")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
selected_dataset_label = (
    dataset_name.replace("_", " ").title()
    if dataset_source == "builtin"
    else (uploaded_file.name if uploaded_file is not None else "CSV dataset")
)
st.markdown(
    f"""
    <div class="hero">
      <span class="eyebrow">AI &amp; DATA SCIENCE</span>
      <span class="eyebrow">MODEL MONITORING</span>
      <span class="eyebrow">CONCEPT DRIFT RESEARCH</span>
      <h1>🌊 Drifto</h1>
      <p>An interactive ML monitoring prototype that compares reference data with simulated incoming data,
      highlights feature distribution changes, and tracks model performance using the {selected_dataset_label} dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == "Overview":
    st.markdown('<div class="section-title">System Overview</div>',
                unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    overview_metrics = [
        (m1, "Baseline accuracy", baseline_metrics["accuracy"]),
        (m2, "Simulated accuracy", new_metrics["accuracy"]),
        (m3, "Features flagged", len(drifted_df)),
        (m4, "Features monitored", len(features)),
    ]
    for col, label, value in overview_metrics:
        shown = f"{value:.2%}" if isinstance(
            value, (float, np.floating)) else str(value)
        col.markdown(
            f'<div class="metric"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{shown}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">What Drifto checks</div>',
                unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**Reference vs. simulated data**")
        compare = pd.DataFrame({
            "Dataset": ["Reference", "Simulated"],
            "Rows": [len(reference_data), len(new_data)],
            "Features": [reference_data.shape[1], new_data.shape[1]],
        })
        st.dataframe(compare, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Largest feature shifts**")
        plot_data = drift_df.head(10).sort_values("Drift score")
        fig = px.bar(
            plot_data, x="Drift score", y="Feature", orientation="h",
            color="Drift detected",
            color_discrete_map={True: "#fb7185", False: "#38bdf8"},
            title="Top 10 features by standardized mean shift",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=45, b=10))
        render_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="note"><strong>Interpretation:</strong> Feature drift means that the observed '
        'feature distribution has changed according to this demo’s simple score. It does not, by itself, '
        'prove that the model is inaccurate. The simulated labels are retained so performance can also be measured.</div>',
        unsafe_allow_html=True,
    )

elif page == "Baseline Metrics":
    st.markdown('<div class="section-title">Baseline vs. Simulated Performance</div>',
                unsafe_allow_html=True)
    metric_rows = []
    for key, label in [
        ("accuracy", "Accuracy"), ("precision", "Weighted precision"),
        ("recall", "Weighted recall"), ("f1", "Weighted F1-score")
    ]:
        metric_rows.append({
            "Metric": label,
            "Reference split": baseline_metrics[key],
            "Simulated data": new_metrics[key],
            "Change (percentage points)": (new_metrics[key] - baseline_metrics[key]) * 100,
        })
    metric_table = pd.DataFrame(metric_rows)
    st.dataframe(
        metric_table,
        use_container_width=True, hide_index=True,
    )
    cm = confusion_matrix(y_ref, new_predictions, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=pd.Index(["Actual malignant", "Actual benign"]),
        columns=pd.Index(["Predicted malignant", "Predicted benign"]),
    )
    st.markdown("**Confusion matrix — simulated data**")
    st.dataframe(cm_df, use_container_width=True)
    st.caption("This is a controlled simulation using retained reference labels, not independently collected production outcomes.")

elif page == "Prediction Distribution":
    st.markdown('<div class="section-title">Prediction Distribution</div>',
                unsafe_allow_html=True)
    ref_plot = reference_distribution.assign(Dataset="Reference")
    new_plot = new_distribution.assign(Dataset="Simulated")
    combined = pd.concat([ref_plot, new_plot], ignore_index=True)
    fig = px.bar(
        combined, x="Class", y="Count", color="Dataset", barmode="group",
        color_discrete_map={"Reference": "#38bdf8", "Simulated": "#f472b6"},
        title="Predicted class counts",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    render_chart(fig, use_container_width=True)
    st.markdown(
        '<div class="note">A change in predicted class proportions can be a useful monitoring signal, '
        'but it is not proof of concept drift or model failure. Investigate it alongside feature changes '
        'and performance metrics.</div>', unsafe_allow_html=True
    )

elif page == "Feature Drift Analysis":
    st.markdown('<div class="section-title">Feature Drift Analysis</div>',
                unsafe_allow_html=True)
    st.caption("Score = |simulated mean − reference mean| / reference standard deviation. This is a simple effect-size indicator, not a formal statistical significance test.")
    styled_drift = drift_df.copy()
    styled_drift["Highlight"] = styled_drift["Drift detected"].astype(str)
    st.dataframe(
        styled_drift,
        use_container_width=True, height=520, hide_index=True,
    )
    selected_features = st.multiselect(
        "Compare feature distributions",
        options=list(drift_df["Feature"]),
        default=list(drift_df.head(3)["Feature"]),
    )
    if selected_features:
        selected = pd.concat([
            reference_data[selected_features].melt(
                var_name="Feature", value_name="Value").assign(Dataset="Reference"),
            new_data[selected_features].melt(
                var_name="Feature", value_name="Value").assign(Dataset="Simulated"),
        ], ignore_index=True)
        fig = px.box(selected, x="Feature", y="Value", color="Dataset", points=False,
                     color_discrete_map={
                         "Reference": "#38bdf8", "Simulated": "#f472b6"},
                     title="Feature distributions: reference vs simulated")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        render_chart(fig, use_container_width=True)

elif page == "Drift Warnings":
    st.markdown('<div class="section-title">Drift Warnings</div>',
                unsafe_allow_html=True)
    if len(drifted_df):
        st.markdown(
            f'<div class="warning"><strong>⚠️ {len(drifted_df)} of {len(features)} features flagged.</strong> '
            f'Their standardized mean-shift scores meet or exceed {drift_threshold:.2f}. '
            'This is a monitoring signal requiring investigation, not proof of model failure.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            drifted_df[["Feature", "Mean difference",
                        "Drift score", "Drift detected"]].copy(),
            use_container_width=True, hide_index=True,
        )
    else:
        st.markdown(
            '<div class="success">No features crossed the selected threshold in this simulation. '
            'This does not guarantee that the model is reliable or that no other type of drift exists.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("**Suggested response workflow**")
    st.write(
        "1. Inspect the flagged features and confirm whether the shift is expected.")
    st.write("2. Check model quality when verified labels become available.")
    st.write(
        "3. If degradation is confirmed, investigate the data pipeline and evaluate retraining.")
    st.write("4. Validate any replacement model before deployment. This prototype does not automatically retrain or roll back a model.")

else:  # About
    st.markdown('<div class="section-title">About Drifto</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Project:** Drifto — ML Model Drift Monitoring System

    **Purpose:** Demonstrate a repeatable workflow for comparing reference data with simulated incoming data,
    flagging feature shifts, and checking whether prediction quality changes in a controlled experiment.

    **Current implementation**
    - Random Forest classifier trained on a selected built-in sklearn dataset or CSV upload.
    - Reproducible simulated data with adjustable shifts.
    - Feature-level standardized mean-shift scores and configurable threshold warnings.
    - Performance metrics and prediction-distribution visualizations.
    - Interactive Streamlit interface.

    **Current limitations**
    - The incoming data is simulated, not connected to a live application.
    - The drift score is a simple standardized mean difference, not a formal statistical test.
    - This version does not implement streaming concept-drift detectors, online learning, automatic retraining, or rollback.
    - Results from this educational dataset do not establish production readiness.
    """)
    st.markdown(
        '<div class="note"><strong>Research direction:</strong> compare the baseline score with formal '
        'drift tests and streaming detectors such as ADWIN; then evaluate detection delay, false alarms, '
        'and model performance under controlled drift scenarios.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "Drifto • Educational prototype • Simulated data only • No live production system connected")
