from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


@dataclass(frozen=True)
class TrainResult:
    pipeline: Pipeline
    feature_names: list[str]
    metrics: dict[str, Any]
    perm_importance: pd.DataFrame


def build_pipeline(feature_cols: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=None,
        class_weight="balanced",
        solver="lbfgs",
    )

    return Pipeline(steps=[("pre", preprocessor), ("clf", clf)])


def train_evaluate(
    df: pd.DataFrame,
    *,
    target_col: str = "sleep_quality",
    group_col: str = "user_id",
    id_cols: list[str] | None = None,
    random_state: int = 7,
) -> TrainResult:
    """
    Trains a classifier and evaluates ROC-AUC with a user-group split to reduce leakage.
    """
    id_cols = id_cols or ["user_id", "date"]

    y = df[target_col].astype(int).to_numpy()
    groups = df[group_col].to_numpy()

    feature_cols = [c for c in df.columns if c not in set(id_cols + [target_col])]
    X = df[feature_cols]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    (train_idx, test_idx) = next(splitter.split(X, y, groups=groups))

    pipe = build_pipeline(feature_cols)
    pipe.fit(X.iloc[train_idx], y[train_idx])

    proba_test = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    auc = float(roc_auc_score(y[test_idx], proba_test))

    # Permutation importance on test for interpretability
    perm = permutation_importance(
        pipe,
        X.iloc[test_idx],
        y[test_idx],
        scoring="roc_auc",
        n_repeats=10,
        random_state=random_state,
        n_jobs=None,
    )
    perm_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, kind="mergesort")

    # Model coefficient importances (after scaling). Keep aligned with feature_cols.
    coef = pipe.named_steps["clf"].coef_.ravel().astype(float)
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coef}).sort_values("coef", ascending=False)

    metrics: dict[str, Any] = {
        "roc_auc": auc,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "n_users": int(df[group_col].nunique()),
        "positive_rate": float(np.mean(y)),
        "top_positive_coef": coef_df.head(10).to_dict(orient="records"),
        "top_negative_coef": coef_df.tail(10).to_dict(orient="records"),
    }

    return TrainResult(pipeline=pipe, feature_names=feature_cols, metrics=metrics, perm_importance=perm_df)


def save_artifacts(
    result: TrainResult,
    *,
    model_path: str,
    feature_schema_path: str,
    metrics_path: str,
    perm_importance_path: str,
) -> None:
    dump(result.pipeline, model_path)

    import json

    with open(feature_schema_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": result.feature_names}, f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result.metrics, f, indent=2)

    result.perm_importance.to_csv(perm_importance_path, index=False)

