from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from joblib import load
from sklearn.inspection import permutation_importance

from src.data_generation import SyntheticConfig, generate_synthetic_wearable_timeseries
from src.features import aggregate_daily_features


ARTIFACTS_DIR = Path("artifacts")
DATA_RAW_DIR = Path("data/raw")


def _ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (Path("data/raw")).mkdir(parents=True, exist_ok=True)
    (Path("data/processed")).mkdir(parents=True, exist_ok=True)


def _load_or_generate_data(n_users: int, days_per_user: int, seed: int, freq_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _load_model_and_schema() -> tuple[object | None, list[str] | None]:
    model_path = ARTIFACTS_DIR / "model.joblib"
    schema_path = ARTIFACTS_DIR / "feature_schema.json"
    if not model_path.exists() or not schema_path.exists():
        return None, None
    model = load(model_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, list(schema["feature_names"])


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


st.set_page_config(page_title="Wearable Sleep Quality Insights", layout="wide")
st.title("Wearable Prediction & Insights System")

_ensure_dirs()

with st.sidebar:
    st.header("Data")
    n_users = st.slider("Synthetic users", 50, 600, 200, 50)
    days_per_user = st.slider("Days per user", 5, 30, 10, 1)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=7, step=1)
    freq_minutes = st.selectbox("Sampling frequency (minutes)", [1, 5, 10, 15], index=1)
    st.caption("If `data/raw/*.parquet` exists, those files are used.")

model, feature_names = _load_model_and_schema()

ts_df, labels_df = _load_or_generate_data(int(n_users), int(days_per_user), int(seed), int(freq_minutes))
feat_df = aggregate_daily_features(ts_df)
modeling_df = feat_df.merge(labels_df, on=["user_id", "date"], how="inner")

colA, colB, colC = st.columns([1.2, 1.2, 1.6])

with colA:
    st.subheader("Select a user/day")
    user_id = st.selectbox("User", sorted(modeling_df["user_id"].unique().tolist()))
    dates = modeling_df.loc[modeling_df["user_id"] == user_id, "date"].sort_values().unique().tolist()
    date = st.selectbox("Date", dates)

row = modeling_df[(modeling_df["user_id"] == user_id) & (modeling_df["date"] == date)].iloc[0]
target = int(row["sleep_quality"])

with colB:
    st.subheader("Prediction")
    if model is None or feature_names is None:
        st.warning("Model artifacts not found. Run `python -m src.train --generate-data` to train and save them.")
        st.stop()

    X_row = pd.DataFrame([row[feature_names].to_dict()])
    proba = float(model.predict_proba(X_row)[0, 1])
    pred = int(proba >= 0.5)

    st.metric("P(high sleep quality)", f"{proba:.3f}")
    st.write(f"**Predicted**: {'High' if pred == 1 else 'Low'}")
    st.write(f"**Actual**: {'High' if target == 1 else 'Low'}")

with colC:
    st.subheader("Time-series view (selected day)")
    day_ts = ts_df[(ts_df["user_id"] == user_id) & (pd.to_datetime(ts_df["timestamp"]).dt.normalize() == pd.to_datetime(date))]
    day_ts = day_ts.sort_values("timestamp")
    fig = px.line(day_ts, x="timestamp", y="heart_rate", title="Heart rate")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.area(day_ts, x="timestamp", y="activity_intensity", title="Activity intensity")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("Feature importance & recommendations")

tab1, tab2, tab3 = st.tabs(["Global (saved)", "Local (this row)", "Model diagnostics"])

with tab1:
    perm_path = ARTIFACTS_DIR / "permutation_importance.csv"
    if perm_path.exists():
        perm_df = pd.read_csv(perm_path)
        topk = perm_df.head(20)
        fig = px.bar(topk[::-1], x="importance_mean", y="feature", orientation="h", title="Permutation importance (top 20)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No saved permutation importance found. Train the model first to generate it.")

with tab2:
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

    st.write("Lowest (most negative) estimated contributions to high sleep quality:")
    st.dataframe(contrib.head(12), use_container_width=True)

    st.write("Recommendations:")
    recs = _recommendations(row, coef_map)
    if not recs:
        st.write("- Keep doing what you're doing; no strong negative drivers detected.")
    else:
        for r in recs:
            st.write(f"- {r}")

with tab3:
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.json(metrics)
    else:
        st.info("No metrics found. Train the model first.")

