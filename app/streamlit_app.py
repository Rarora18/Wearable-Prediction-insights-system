from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joblib import load

# Ensure the repo root is importable when running via Streamlit.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_generation import SyntheticConfig, generate_synthetic_wearable_timeseries
from src.features import aggregate_daily_features


ARTIFACTS_DIR = Path("artifacts")
DATA_RAW_DIR = Path("data/raw")


def _ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (Path("data/raw")).mkdir(parents=True, exist_ok=True)
    (Path("data/processed")).mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def _load_or_generate_data(
    n_users: int, days_per_user: int, seed: int, freq_minutes: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_path = DATA_RAW_DIR / "wearable_timeseries.parquet"
    labels_path = DATA_RAW_DIR / "daily_labels.parquet"

    if ts_path.exists() and labels_path.exists():
        return pd.read_parquet(ts_path), pd.read_parquet(labels_path)

    ts_df, labels_df = generate_synthetic_wearable_timeseries(
        n_users=n_users,
        days_per_user=days_per_user,
        cfg=SyntheticConfig(seed=seed, freq_minutes=freq_minutes),
    )
    ts_df.to_parquet(ts_path, index=False)
    labels_df.to_parquet(labels_path, index=False)
    return ts_df, labels_df


@st.cache_data(show_spinner=False)
def _build_modeling_df(ts_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    feat_df = aggregate_daily_features(ts_df)
    return feat_df.merge(labels_df, on=["user_id", "date"], how="inner")


def _load_model_and_schema() -> tuple[object | None, list[str] | None]:
    model_path = ARTIFACTS_DIR / "model.joblib"
    schema_path = ARTIFACTS_DIR / "feature_schema.json"
    if not model_path.exists() or not schema_path.exists():
        return None, None
    model = load(model_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, list(schema["feature_names"])


def _badge(label: str, *, kind: str) -> str:
    # kind: "good" | "bad" | "neutral"
    colors = {"good": "#16a34a", "bad": "#dc2626", "neutral": "#334155"}
    bg = colors.get(kind, "#334155")
    return f"<span style='background:{bg}; color:white; padding:0.15rem 0.55rem; border-radius:999px; font-size:0.85rem;'>{label}</span>"


def _prob_gauge(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(prob) * 100.0,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "#0f172a", "width": 3}, "thickness": 0.8, "value": 50},
            },
            title={"text": "P(high sleep quality)"},
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=240)
    return fig


def _recommendations(feature_row: pd.Series, coef_map: dict[str, float]) -> list[str]:
    """
    Simple heuristic: identify strongest negative contributions to high sleep quality.
    """
    contrib = {f: float(feature_row.get(f, 0.0)) * float(coef_map.get(f, 0.0)) for f in coef_map.keys()}
    worst = sorted(contrib.items(), key=lambda x: x[1])[:8]

    recs: list[str] = []
    for feat, _ in worst:
        if feat in {"steps_night", "restlessness_proxy"}:
            recs.append("Reduce night-time movement: keep the room cooler/darker and limit late fluids.")
        elif feat in {"intensity_evening", "circadian_misalignment_proxy"}:
            recs.append("Move intense activity earlier; try a lighter wind-down routine in the evening.")
        elif feat in {"hr_night_mean", "hr_night_std", "hr_night_minus_day"}:
            recs.append("Aim for better recovery: consistent bedtime, avoid alcohol late, manage stress before sleep.")
        elif feat in {"steps_total"}:
            recs.append("Keep steady daytime activity; avoid long sedentary stretches.")
        elif feat.startswith("hrv_") or feat in {"hr_roll_std_mean"}:
            recs.append("Support HRV: prioritize sleep consistency and avoid heavy training very late.")

    # De-duplicate while preserving order
    seen = set()
    out = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out[:5]


st.set_page_config(page_title="Wearable Prediction & Insights System", page_icon="🛌", layout="wide")

st.markdown(
    """
<style>
div[data-testid="stMetricValue"] { font-size: 2.0rem; }
div[data-testid="stMetricLabel"] { font-size: 0.95rem; color: #475569; }
section.main > div { padding-top: 1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Wearable Prediction & Insights System")
st.caption("Sleep quality classification with feature engineering, explainability, and personalized recommendations.")

_ensure_dirs()

with st.sidebar:
    st.header("Controls")
    st.markdown("Adjust the synthetic data generator (ignored if `data/raw/*.parquet` already exists).")
    st.subheader("Data")
    n_users = st.slider("Users", 50, 600, 200, 50)
    days_per_user = st.slider("Days per user", 5, 30, 10, 1)
    col_seed, col_freq = st.columns(2)
    with col_seed:
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=7, step=1)
    with col_freq:
        freq_minutes = st.selectbox("Freq (min)", [1, 5, 10, 15], index=1)

    st.divider()
    st.subheader("Display")
    show_raw_features = st.toggle("Show raw feature row", value=False)
    show_data_preview = st.toggle("Show dataset preview", value=False)
    st.divider()
    st.info("Tip: Train once with `python -m src.train --generate-data` to enable the dashboard model.")

model, feature_names = _load_model_and_schema()

ts_df, labels_df = _load_or_generate_data(int(n_users), int(days_per_user), int(seed), int(freq_minutes))
modeling_df = _build_modeling_df(ts_df, labels_df)

top_left, top_mid, top_right = st.columns([1.25, 1.25, 1.5])

with top_left:
    st.subheader("Selection")
    user_id = st.selectbox("User", sorted(modeling_df["user_id"].unique().tolist()))
    dates = modeling_df.loc[modeling_df["user_id"] == user_id, "date"].sort_values().unique().tolist()
    date = st.selectbox("Date", dates)

row = modeling_df[(modeling_df["user_id"] == user_id) & (modeling_df["date"] == date)].iloc[0]
target = int(row["sleep_quality"])

with top_mid:
    st.subheader("Prediction")
    if model is None or feature_names is None:
        st.warning("Model artifacts not found. Run `python -m src.train --generate-data` to train and save them.")
        st.stop()

    X_row = pd.DataFrame([row[feature_names].to_dict()])
    proba = float(model.predict_proba(X_row)[0, 1])
    pred = int(proba >= 0.5)

    st.plotly_chart(_prob_gauge(proba), use_container_width=True, config={"displayModeBar": False})

    pred_label = "High" if pred == 1 else "Low"
    act_label = "High" if target == 1 else "Low"
    pred_kind = "good" if pred == 1 else "bad"
    act_kind = "good" if target == 1 else "bad"
    st.markdown(
        f"**Predicted**: {_badge(pred_label, kind=pred_kind)} &nbsp;&nbsp; **Actual**: {_badge(act_label, kind=act_kind)}",
        unsafe_allow_html=True,
    )

with top_right:
    st.subheader("Day overview")
    st.caption("Heart rate and activity intensity for the selected user/day.")
    day_ts = ts_df[(ts_df["user_id"] == user_id) & (pd.to_datetime(ts_df["timestamp"]).dt.normalize() == pd.to_datetime(date))]
    day_ts = day_ts.sort_values("timestamp")
    fig = px.line(day_ts, x="timestamp", y="heart_rate", title="Heart rate", template="plotly_white")
    fig.update_traces(line=dict(color="#ef4444", width=2.2))
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=220)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.area(day_ts, x="timestamp", y="activity_intensity", title="Activity intensity", template="plotly_white")
    fig2.update_traces(line=dict(color="#2563eb", width=1.5), fillcolor="rgba(37,99,235,0.25)")
    fig2.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=220)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

met1, met2, met3, met4 = st.columns(4)
met1.metric("Users", int(modeling_df["user_id"].nunique()))
met2.metric("Days", int(len(modeling_df)))
met3.metric("Positive rate", f"{float(modeling_df['sleep_quality'].mean()):.3f}")
met4.metric("Sampling freq", f"{int(freq_minutes)} min")

st.divider()

st.subheader("Feature importance & recommendations")

tab1, tab2, tab3 = st.tabs(["Global importance", "This prediction (local)", "Model diagnostics"])

with tab1:
    perm_path = ARTIFACTS_DIR / "permutation_importance.csv"
    if perm_path.exists():
        perm_df = pd.read_csv(perm_path)
        topk = perm_df.head(20).sort_values("importance_mean", ascending=True)
        fig = px.bar(
            topk,
            x="importance_mean",
            y="feature",
            orientation="h",
            title="Permutation importance (top 20)",
            template="plotly_white",
        )
        fig.update_traces(marker_color="#0ea5e9")
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No saved permutation importance found. Train the model first to generate it.")

with tab2:
    left, right = st.columns([1.35, 1.0])
    # Local explanation via linear contributions (after scaling it's not exact in raw feature space),
    # so we approximate using raw * coef for actionable ranking.
    clf = model.named_steps["clf"]
    coefs = clf.coef_.ravel().astype(float)
    coef_map = dict(zip(feature_names, coefs))

    contrib = pd.DataFrame(
        {
            "feature": feature_names,
            "value": [float(row[f]) for f in feature_names],
            "coef": [float(coef_map[f]) for f in feature_names],
        }
    )
    contrib["raw_contribution"] = contrib["value"] * contrib["coef"]
    contrib = contrib.sort_values("raw_contribution")

    with left:
        st.write("Drivers (most negative contributions to **high** sleep quality):")
        st.dataframe(contrib.head(14), use_container_width=True, hide_index=True)

        if show_raw_features:
            with st.expander("Raw feature row"):
                st.dataframe(pd.DataFrame([row[feature_names].to_dict()]), use_container_width=True, hide_index=True)

    with right:
        st.write("Recommendations")
        recs = _recommendations(row, coef_map)
        if not recs:
            st.success("No strong negative drivers detected. Keep your routine consistent.")
        else:
            for r in recs:
                st.markdown(f"- {r}")

        st.write("Quick checks")
        st.metric("Night steps", f"{float(row.get('steps_night', 0.0)):.0f}")
        st.metric("Evening intensity", f"{float(row.get('intensity_evening', 0.0)):.3f}")
        st.metric("Night HR mean", f"{float(row.get('hr_night_mean', 0.0)):.1f}")

with tab3:
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.json(metrics)
    else:
        st.info("No metrics found. Train the model first.")

if show_data_preview:
    st.divider()
    st.subheader("Dataset preview")
    st.dataframe(modeling_df.head(50), use_container_width=True, hide_index=True)

