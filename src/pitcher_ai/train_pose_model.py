from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from pitcher_ai.config import PROCESSED_DIR, REPORTS_DIR
from pitcher_ai.pitch_types import pitch_type_to_class
from pitcher_ai.train_baseline import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from pitcher_ai.util import console, ensure_dirs

POSE_REPORT_PATH = REPORTS_DIR / "pose_comparison_report.txt"
POSE_METRICS_PATH = REPORTS_DIR / "pose_comparison_metrics.json"

_KEY_COLS = {"game_pk", "at_bat_number", "pitch_number", "pitcher", "pitch_type"}


def _build_rf_pipeline(cat_cols: list[str], num_cols: list[str], pose_cols: list[str]) -> Pipeline:
    from sklearn.compose import ColumnTransformer

    transformers = []
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            cat_cols,
        ))
    all_num = num_cols + pose_cols
    if all_num:
        transformers.append((
            "num",
            SimpleImputer(strategy="median"),
            all_num,
        ))

    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )),
    ])


def _cv_report(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, Any]:
    preds = cross_val_predict(pipeline, X, y, cv=cv)
    acc = accuracy_score(y, preds)
    labels = sorted(y.unique().tolist())
    report = classification_report(y, preds, labels=labels, zero_division=0)
    y_class = y.map(pitch_type_to_class)
    pred_class = pd.Series(preds, index=y.index).map(pitch_type_to_class)
    class_acc = accuracy_score(y_class, pred_class)

    # Per-fold accuracy for std
    fold_accs = []
    for train_idx, test_idx in cv.split(X, y):
        clone = _clone_pipeline(pipeline)
        clone.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_accs.append(accuracy_score(y.iloc[test_idx], clone.predict(X.iloc[test_idx])))

    console.print(f"\n[bold]{name}[/bold]")
    console.print(f"  pitch-type accuracy : {acc:.3f}  (fold std {np.std(fold_accs):.3f})")
    console.print(f"  pitch-class accuracy: {class_acc:.3f}")
    console.print(report)

    return {
        "name": name,
        "pitch_type_accuracy": float(acc),
        "pitch_type_accuracy_fold_std": float(np.std(fold_accs)),
        "pitch_class_accuracy": float(class_acc),
        "classification_report": report,
        "n_samples": int(len(y)),
        "labels": labels,
    }


def _clone_pipeline(pipeline: Pipeline) -> Pipeline:
    from sklearn.base import clone
    return clone(pipeline)


def train_pose_comparison(
    pose_path: Path = PROCESSED_DIR / "pose_features.parquet",
    context_path: Path = PROCESSED_DIR / "pitch_context_features.parquet",
    n_folds: int = 5,
    min_class_samples: int = 5,
) -> dict[str, Any]:
    ensure_dirs([REPORTS_DIR])

    # --- load and join ---
    pose_df = pd.read_parquet(pose_path)
    ctx_df = pd.read_parquet(context_path)

    merged = pose_df.merge(
        ctx_df,
        on=["game_pk", "at_bat_number", "pitch_number"],
        how="inner",
        suffixes=("", "_ctx"),
    )
    # Use pitch_type from pose_df (the _ctx duplicate is dropped by the suffix strategy)
    merged["pitch_type"] = merged["pitch_type"].astype(str)

    # Drop pitch types with too few samples for CV
    counts = merged["pitch_type"].value_counts()
    keep = counts[counts >= min_class_samples].index
    dropped = counts[counts < min_class_samples].index.tolist()
    if dropped:
        console.print(f"Dropping pitch types with <{min_class_samples} samples: {dropped}")
    merged = merged[merged["pitch_type"].isin(keep)].copy()

    console.print(f"Dataset: {len(merged)} rows, {merged['pitch_type'].nunique()} pitch types")
    console.print(merged["pitch_type"].value_counts().to_string())

    y = merged["pitch_type"]
    pose_cols = sorted(c for c in merged.columns if c not in _KEY_COLS and c not in
                       set(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + ["game_date", "pitch_type_ctx",
                                                                     "pitch_name", "pitch_class",
                                                                     "pitch_type_name", "p_throws"]))
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in merged.columns]
    num_cols = [c for c in NUMERIC_COLUMNS if c in merged.columns]

    console.print(f"\nFeature counts — context cat: {len(cat_cols)}, context num: {len(num_cols)}, pose: {len(pose_cols)}")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = []

    # 1. Context-only (same feature set as baseline, retrained on these 144 rows)
    pipe_ctx = _build_rf_pipeline(cat_cols, num_cols, [])
    results.append(_cv_report("Context-only (RF, same 144 rows)", pipe_ctx, merged[cat_cols + num_cols], y, cv))

    # 2. Pose-only
    pipe_pose = _build_rf_pipeline([], [], pose_cols)
    results.append(_cv_report("Pose-only (RF)", pipe_pose, merged[pose_cols], y, cv))

    # 3. Context + Pose
    all_feat_cols = cat_cols + num_cols + pose_cols
    pipe_both = _build_rf_pipeline(cat_cols, num_cols, pose_cols)
    results.append(_cv_report("Context + Pose (RF)", pipe_both, merged[all_feat_cols], y, cv))

    # --- write outputs ---
    lines = ["# Pose vs Context Comparison", f"", f"N={len(merged)} pitches, {n_folds}-fold stratified CV", ""]
    for r in results:
        lines += [
            f"## {r['name']}",
            f"Pitch-type accuracy : {r['pitch_type_accuracy']:.3f}  ±{r['pitch_type_accuracy_fold_std']:.3f}",
            f"Pitch-class accuracy: {r['pitch_class_accuracy']:.3f}",
            "",
            r["classification_report"],
        ]

    POSE_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    metrics = {
        "n_samples": int(len(merged)),
        "n_folds": n_folds,
        "pitch_types": sorted(y.unique().tolist()),
        "models": results,
    }
    POSE_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    console.print(f"\nWrote report -> {POSE_REPORT_PATH}")
    console.print(f"Wrote metrics -> {POSE_METRICS_PATH}")
    return metrics
