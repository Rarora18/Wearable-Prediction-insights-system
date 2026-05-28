from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.data_generation import SyntheticConfig, generate_synthetic_wearable_timeseries
from src.features import aggregate_daily_features
from src.modeling import save_artifacts, train_evaluate


def _ensure_dirs() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sleep-quality classifier from wearable data.")
    parser.add_argument("--generate-data", action="store_true", help="Generate a synthetic dataset.")
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--days-per-user", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--freq-minutes", type=int, default=5)

    args = parser.parse_args()
    _ensure_dirs()

    ts_path = Path("data/raw/wearable_timeseries.parquet")
    labels_path = Path("data/raw/daily_labels.parquet")

    if args.generate_data or (not ts_path.exists()) or (not labels_path.exists()):
        ts_df, labels_df = generate_synthetic_wearable_timeseries(
            n_users=args.n_users,
            days_per_user=args.days_per_user,
            cfg=SyntheticConfig(seed=args.seed, freq_minutes=args.freq_minutes),
        )
        ts_df.to_parquet(ts_path, index=False)
        labels_df.to_parquet(labels_path, index=False)
    else:
        ts_df = pd.read_parquet(ts_path)
        labels_df = pd.read_parquet(labels_path)

    feat_df = aggregate_daily_features(ts_df)
    modeling_df = feat_df.merge(labels_df, on=["user_id", "date"], how="inner")
    modeling_df.to_parquet("data/processed/daily_features.parquet", index=False)

    result = train_evaluate(modeling_df, target_col="sleep_quality", group_col="user_id", id_cols=["user_id", "date"])

    save_artifacts(
        result,
        model_path="artifacts/model.joblib",
        feature_schema_path="artifacts/feature_schema.json",
        metrics_path="artifacts/metrics.json",
        perm_importance_path="artifacts/permutation_importance.csv",
    )

    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()

