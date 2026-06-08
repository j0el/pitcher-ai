from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pitcher_ai.config import (
    CLASS_CONFUSION_MATRIX_PATH,
    CLASS_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PITCH_TYPE_KEY_PATH,
    REPORT_PATH,
    REPORTS_DIR,
    SUMMARY_PATH,
)
from pitcher_ai.features import TARGET
from pitcher_ai.pitch_types import (
    PITCH_CLASS,
    PITCH_CLASS_ORDER,
    PITCH_TYPE_NAME,
    pitch_type_key_rows,
    pitch_type_to_class,
    pitch_type_to_name,
)
from pitcher_ai.util import console, ensure_dirs, write_json

CATEGORICAL_COLUMNS = [
    "pitcher",
    "batter",
    "stand",
    "p_throws",
    "inning_topbot",
    "home_team",
    "away_team",
    "prev_pitch_type",
    "count",
    "base_state",
]

NUMERIC_COLUMNS = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "bat_score",
    "fld_score",
    "pitch_number",
    "on_1b",
    "on_2b",
    "on_3b",
    "prev_release_speed",
    "prev_balls",
    "prev_strikes",
    "score_diff",
]


def train_context_model(
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    test_days: int = 14,
    max_rows: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    ensure_dirs([model_path.parent, REPORTS_DIR])

    console.print(f"Reading features from {features_path}")
    df = pd.read_parquet(features_path)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state).sort_values(
            [col for col in ["game_date", "game_pk", "at_bat_number", "pitch_number"] if col in df.columns]
        )

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype("string")
    df[PITCH_CLASS] = df[TARGET].map(pitch_type_to_class).astype("string")
    df[PITCH_TYPE_NAME] = df[TARGET].map(pitch_type_to_name).astype("string")

    if "game_date" not in df.columns:
        raise ValueError("features need game_date for time-based train/test split")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])
    split_date = df["game_date"].max() - pd.Timedelta(days=test_days)

    train_df = df[df["game_date"] <= split_date].copy()
    test_df = df[df["game_date"] > split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("time split produced empty train or test set; use a wider date range or lower test_days")

    cat_cols = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    num_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]

    for col in cat_cols:
        train_df[col] = train_df[col].astype("string").fillna("UNK")
        test_df[col] = test_df[col].astype("string").fillna("UNK")
    for col in num_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    X_train = train_df[cat_cols + num_cols]
    y_train = train_df[TARGET].astype("string")
    X_test = test_df[cat_cols + num_cols]
    y_test = test_df[TARGET].astype("string")

    y_train_class = train_df[PITCH_CLASS].astype("string")
    y_test_class = test_df[PITCH_CLASS].astype("string")

    overall_baseline = majority_baseline_accuracy(y_train, y_test)
    pitcher_count_baseline = pitcher_count_baseline_accuracy(train_df, test_df, target_col=TARGET)
    overall_class_baseline = majority_baseline_accuracy(y_train_class, y_test_class)
    pitcher_count_class_baseline = pitcher_count_baseline_accuracy(train_df, test_df, target_col=PITCH_CLASS)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-5,
                    penalty="elasticnet",
                    l1_ratio=0.05,
                    max_iter=1000,
                    tol=1e-3,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    console.print(f"Training on {len(train_df):,} rows; testing on {len(test_df):,} rows")
    model.fit(X_train, y_train)
    preds = pd.Series(model.predict(X_test), index=y_test.index, name="predicted_pitch_type")
    accuracy = accuracy_score(y_test, preds)

    labels = sorted(y_train.unique().tolist())
    report = classification_report(y_test, preds, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    pred_classes = preds.map(pitch_type_to_class).astype("string")
    class_labels = [label for label in PITCH_CLASS_ORDER if label in set(y_train_class) | set(y_test_class) | set(pred_classes)]
    class_accuracy = accuracy_score(y_test_class, pred_classes)
    class_report = classification_report(y_test_class, pred_classes, labels=class_labels, zero_division=0)
    class_cm = confusion_matrix(y_test_class, pred_classes, labels=class_labels)
    class_cm_df = pd.DataFrame(class_cm, index=class_labels, columns=class_labels)

    joblib.dump(
        {
            "model": model,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "target": TARGET,
            "class_target": PITCH_CLASS,
            "split_date": str(split_date.date()),
        },
        model_path,
    )

    REPORT_PATH.write_text(report, encoding="utf-8")
    CLASS_REPORT_PATH.write_text(class_report, encoding="utf-8")
    cm_df.to_csv(CONFUSION_MATRIX_PATH)
    class_cm_df.to_csv(CLASS_CONFUSION_MATRIX_PATH)
    pd.DataFrame(pitch_type_key_rows()).to_csv(PITCH_TYPE_KEY_PATH, index=False)

    metrics = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "split_date": str(split_date.date()),
        "target_classes": labels,
        "pitch_classes": class_labels,
        "overall_majority_baseline_accuracy": overall_baseline,
        "pitcher_count_baseline_accuracy": pitcher_count_baseline,
        "context_model_accuracy": float(accuracy),
        "overall_majority_baseline_pitch_class_accuracy": overall_class_baseline,
        "pitcher_count_baseline_pitch_class_accuracy": pitcher_count_class_baseline,
        "context_model_pitch_class_accuracy": float(class_accuracy),
        "features_used": {
            "categorical": cat_cols,
            "numeric": num_cols,
        },
    }
    write_json(METRICS_PATH, metrics)
    write_summary(SUMMARY_PATH, metrics)

    console.print(f"Pitch-type overall majority baseline accuracy: {overall_baseline:.3f}")
    console.print(f"Pitch-type pitcher/count baseline accuracy: {pitcher_count_baseline:.3f}")
    console.print(f"Pitch-type context model accuracy: {accuracy:.3f}")
    console.print(f"Pitch-class context model accuracy: {class_accuracy:.3f}")
    console.print(f"Wrote model to {model_path}")
    console.print(f"Wrote metrics to {METRICS_PATH}")
    console.print(f"Wrote pitch-type confusion matrix to {CONFUSION_MATRIX_PATH}")
    console.print(f"Wrote pitch-class confusion matrix to {CLASS_CONFUSION_MATRIX_PATH}")
    return metrics


def majority_baseline_accuracy(y_train: pd.Series, y_test: pd.Series) -> float:
    majority = y_train.value_counts().idxmax()
    preds = pd.Series([majority] * len(y_test), index=y_test.index)
    return float(accuracy_score(y_test, preds))


def pitcher_count_baseline_accuracy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET,
) -> float:
    required = {"pitcher", "count", target_col}
    if not required <= set(train_df.columns):
        return float("nan")

    overall = train_df[target_col].value_counts().idxmax()

    by_pitcher_count: dict[tuple[str, str], str] = {}
    for key, group in train_df.groupby(["pitcher", "count"], dropna=False):
        by_pitcher_count[(str(key[0]), str(key[1]))] = group[target_col].value_counts().idxmax()

    by_pitcher: dict[str, str] = {}
    for key, group in train_df.groupby("pitcher", dropna=False):
        by_pitcher[str(key)] = group[target_col].value_counts().idxmax()

    preds = []
    for _, row in test_df.iterrows():
        pitcher = str(row.get("pitcher"))
        count = str(row.get("count"))
        preds.append(by_pitcher_count.get((pitcher, count), by_pitcher.get(pitcher, overall)))

    return float(accuracy_score(test_df[target_col].astype("string"), preds))


def summarize_pitch_mix(features_path: Path = FEATURES_PATH, top_n: int = 20) -> pd.DataFrame:
    df = pd.read_parquet(features_path, columns=["pitcher", TARGET])
    df[PITCH_CLASS] = df[TARGET].map(pitch_type_to_class)
    rows = []
    for pitcher, group in df.groupby("pitcher"):
        total = len(group)
        if total == 0:
            continue
        counts = Counter(group[TARGET])
        class_counts = Counter(group[PITCH_CLASS])
        common = counts.most_common(1)[0]
        common_class = class_counts.most_common(1)[0]
        rows.append(
            {
                "pitcher": pitcher,
                "pitches": total,
                "most_common_pitch": common[0],
                "most_common_share": common[1] / total,
                "most_common_class": common_class[0],
                "most_common_class_share": common_class[1] / total,
                "pitch_mix": json.dumps(dict(counts.most_common(10)), sort_keys=True),
                "pitch_class_mix": json.dumps(dict(class_counts.most_common(4)), sort_keys=True),
            }
        )
    out = pd.DataFrame(rows).sort_values("pitches", ascending=False).head(top_n)
    return out


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Pitcher AI baseline summary",
        "",
        f"Rows total: {metrics['rows_total']:,}",
        f"Rows train: {metrics['rows_train']:,}",
        f"Rows test: {metrics['rows_test']:,}",
        f"Split date: {metrics['split_date']}",
        "",
        "## Pitch-type accuracy",
        "",
        f"Overall majority baseline: {metrics['overall_majority_baseline_accuracy']:.3f}",
        f"Pitcher/count baseline: {metrics['pitcher_count_baseline_accuracy']:.3f}",
        f"Context model: {metrics['context_model_accuracy']:.3f}",
        "",
        "## Pitch-class accuracy",
        "",
        f"Overall majority baseline: {metrics['overall_majority_baseline_pitch_class_accuracy']:.3f}",
        f"Pitcher/count baseline: {metrics['pitcher_count_baseline_pitch_class_accuracy']:.3f}",
        f"Context model mapped to class: {metrics['context_model_pitch_class_accuracy']:.3f}",
        "",
        "## Pitch classes",
        "",
        "Fastball: FF, SI, FC, FT, FA",
        "Breaking: SL, ST, CU, KC, SV",
        "Offspeed: CH, FS, FO",
        "Other/Rare: KN, EP, SC, GY plus administrative/non-pitch feed codes if present",
        "",
        "## Interpretation",
        "",
        "The context model uses only pre-pitch information. It intentionally excludes current-pitch release speed, spin, movement, plate location, and batted-ball fields.",
        "",
        "The pitch-class confusion matrix maps predicted pitch types into broader pitch families, so FF predicted as SI is wrong at the official pitch-type level but still correct at the pitch-class level.",
        "",
        "The next major experiment is to add pre-release video or pose features and test whether they improve over this context-only baseline.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
