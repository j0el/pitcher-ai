from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from pitcher_ai.config import (
    CLASS_CONFUSION_MATRIX_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURES_PATH,
    METRICS_PATH,
    PITCH_TYPE_KEY_PATH,
    REPORTS_DIR,
    SUMMARY_PATH,
)
from pitcher_ai.features import TARGET
from pitcher_ai.pitch_types import PITCH_CLASS, PITCH_CLASS_ORDER, pitch_type_key_rows, pitch_type_to_class
from pitcher_ai.util import console, ensure_dirs

ACCURACY_CHART_PATH = REPORTS_DIR / "accuracy_comparison.png"
CONFUSION_TYPE_CHART_PATH = REPORTS_DIR / "confusion_matrix_pitch_type.png"
CONFUSION_TYPE_WHEN_PREDICTED_CHART_PATH = REPORTS_DIR / "confusion_matrix_pitch_type_when_predicted.png"
CONFUSION_CLASS_CHART_PATH = REPORTS_DIR / "confusion_matrix_pitch_class.png"
CONFUSION_CLASS_WHEN_PREDICTED_CHART_PATH = REPORTS_DIR / "confusion_matrix_pitch_class_when_predicted.png"
PITCH_DIST_CHART_PATH = REPORTS_DIR / "pitch_type_distribution.png"
PITCH_CLASS_DIST_CHART_PATH = REPORTS_DIR / "pitch_class_distribution.png"
PITCH_MIX_CHART_PATH = REPORTS_DIR / "pitch_mix_top_pitchers.png"
INDEX_PATH = REPORTS_DIR / "index.html"


def visualize_results(
    metrics_path: Path = METRICS_PATH,
    confusion_matrix_path: Path = CONFUSION_MATRIX_PATH,
    class_confusion_matrix_path: Path = CLASS_CONFUSION_MATRIX_PATH,
    features_path: Path = FEATURES_PATH,
    output_dir: Path = REPORTS_DIR,
    top_n_pitchers: int = 20,
    max_confusion_classes: int = 12,
) -> dict[str, str]:
    ensure_dirs([output_dir])

    metrics = read_metrics(metrics_path)
    generated: dict[str, str] = {}
    notes: list[str] = []

    if metrics:
        generated["accuracy_chart"] = str(
            plot_accuracy_comparison(metrics, output_dir / ACCURACY_CHART_PATH.name)
        )
    else:
        note = f"Metrics file not found: {metrics_path}"
        notes.append(note)
        console.print(note)

    if confusion_matrix_path.exists():
        generated["pitch_type_confusion_matrix_chart"] = str(
            plot_confusion_matrix(
                confusion_matrix_path,
                output_dir / CONFUSION_TYPE_CHART_PATH.name,
                max_classes=max_confusion_classes,
                title="Pitch-type confusion matrix",
                x_label="Predicted pitch type",
                y_label="Actual pitch type",
                colorbar_label="Share of actual pitch type",
            )
        )
        generated["pitch_type_when_predicted_chart"] = str(
            plot_when_predicted_matrix(
                confusion_matrix_path,
                output_dir / CONFUSION_TYPE_WHEN_PREDICTED_CHART_PATH.name,
                max_classes=max_confusion_classes,
                title="When the model predicts a pitch type",
                x_label="Actual pitch type",
                y_label="Predicted pitch type",
                colorbar_label="Share of predictions",
            )
        )
    else:
        note = f"Pitch-type confusion matrix file not found: {confusion_matrix_path}"
        notes.append(note)
        console.print(note)

    if not class_confusion_matrix_path.exists() and confusion_matrix_path.exists():
        class_confusion_matrix_path = build_class_confusion_from_type_confusion(
            confusion_matrix_path=confusion_matrix_path,
            output_path=class_confusion_matrix_path,
        )
        notes.append(
            "Created pitch-class confusion matrix by aggregating the official pitch-type confusion matrix."
        )

    if class_confusion_matrix_path.exists():
        generated["pitch_class_confusion_matrix_chart"] = str(
            plot_confusion_matrix(
                class_confusion_matrix_path,
                output_dir / CONFUSION_CLASS_CHART_PATH.name,
                max_classes=10,
                title="Pitch-class confusion matrix",
                x_label="Predicted pitch class",
                y_label="Actual pitch class",
                colorbar_label="Share of actual pitch class",
            )
        )
        generated["pitch_class_when_predicted_chart"] = str(
            plot_when_predicted_matrix(
                class_confusion_matrix_path,
                output_dir / CONFUSION_CLASS_WHEN_PREDICTED_CHART_PATH.name,
                max_classes=10,
                title="When the model predicts a pitch class",
                x_label="Actual pitch class",
                y_label="Predicted pitch class",
                colorbar_label="Share of predictions",
            )
        )
    else:
        note = f"Pitch-class confusion matrix file not found: {class_confusion_matrix_path}"
        notes.append(note)
        console.print(note)

    if features_path.exists():
        generated["pitch_type_distribution_chart"] = str(
            plot_pitch_distribution(features_path, output_dir / PITCH_DIST_CHART_PATH.name)
        )
        generated["pitch_class_distribution_chart"] = str(
            plot_pitch_class_distribution(features_path, output_dir / PITCH_CLASS_DIST_CHART_PATH.name)
        )
        generated["pitch_mix_chart"] = str(
            plot_pitcher_pitch_mix(
                features_path,
                output_dir / PITCH_MIX_CHART_PATH.name,
                top_n=top_n_pitchers,
            )
        )
    else:
        note = f"Feature file not found: {features_path}"
        notes.append(note)
        console.print(note)

    key_path = output_dir / PITCH_TYPE_KEY_PATH.name
    pd.DataFrame(pitch_type_key_rows()).to_csv(key_path, index=False)
    generated["pitch_type_key_csv"] = str(key_path)

    generated["index_html"] = str(
        write_html_index(
            output_dir / INDEX_PATH.name,
            metrics=metrics,
            generated=generated,
            summary_path=SUMMARY_PATH,
            pitch_type_key_path=key_path,
            notes=notes,
        )
    )

    console.print("Wrote visual report files:")
    for label, path in generated.items():
        console.print(f"  {label}: {path}")

    return generated


def read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def plot_accuracy_comparison(metrics: dict[str, Any], output_path: Path) -> Path:
    rows = []
    candidates = [
        ("Type majority", metrics.get("overall_majority_baseline_accuracy")),
        ("Type pitcher/count", metrics.get("pitcher_count_baseline_accuracy")),
        ("Type context model", metrics.get("context_model_accuracy")),
        ("Class majority", metrics.get("overall_majority_baseline_pitch_class_accuracy")),
        ("Class pitcher/count", metrics.get("pitcher_count_baseline_pitch_class_accuracy")),
        ("Class from type model", metrics.get("context_model_pitch_class_accuracy")),
    ]
    for label, value in candidates:
        if value is not None:
            try:
                rows.append((label, float(value)))
            except (TypeError, ValueError):
                continue

    if not rows:
        rows = [("No metrics", 0.0)]

    df = pd.DataFrame(rows, columns=["model", "accuracy"])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(df["model"], df["accuracy"])
    ax.set_ylim(0, max(1.0, float(df["accuracy"].max()) * 1.15))
    ax.set_ylabel("Accuracy")
    ax.set_title("Pitch-type and pitch-class prediction accuracy")
    ax.tick_params(axis="x", rotation=20)
    for idx, row in df.iterrows():
        ax.text(idx, row["accuracy"] + 0.015, f"{row['accuracy']:.1%}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def build_class_confusion_from_type_confusion(confusion_matrix_path: Path, output_path: Path) -> Path:
    """Aggregate the official pitch-type confusion matrix into four pitch classes.

    This lets the class matrix appear even when the model was trained before the
    class-aware train_baseline.py existed. It sums all official-code cells into
    Fastball, Breaking, Offspeed, and Other/Rare buckets.
    """
    type_cm = pd.read_csv(confusion_matrix_path, index_col=0)
    ordered_classes = [
        pitch_class
        for pitch_class in PITCH_CLASS_ORDER
        if any(pitch_type_to_class(code) == pitch_class for code in list(type_cm.index) + list(type_cm.columns))
    ]
    if not ordered_classes:
        ordered_classes = PITCH_CLASS_ORDER

    class_cm = pd.DataFrame(0, index=ordered_classes, columns=ordered_classes, dtype=int)
    for actual_type in type_cm.index:
        actual_class = pitch_type_to_class(actual_type)
        if actual_class not in class_cm.index:
            class_cm.loc[actual_class] = 0
        for predicted_type in type_cm.columns:
            predicted_class = pitch_type_to_class(predicted_type)
            if predicted_class not in class_cm.columns:
                class_cm[predicted_class] = 0
            class_cm.loc[actual_class, predicted_class] += int(type_cm.loc[actual_type, predicted_type])

    class_cm = class_cm.reindex(index=PITCH_CLASS_ORDER, columns=PITCH_CLASS_ORDER, fill_value=0)
    class_cm = class_cm.loc[(class_cm.sum(axis=1) > 0), (class_cm.sum(axis=0) > 0)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_cm.to_csv(output_path)
    console.print(f"Wrote derived pitch-class confusion matrix to {output_path}")
    return output_path


def plot_confusion_matrix(
    input_path: Path,
    output_path: Path,
    max_classes: int = 12,
    title: str = "Normalized confusion matrix",
    x_label: str = "Predicted",
    y_label: str = "Actual",
    colorbar_label: str = "Share of actual class",
) -> Path:
    cm = pd.read_csv(input_path, index_col=0)
    if len(cm) > max_classes:
        totals = cm.sum(axis=1).sort_values(ascending=False).head(max_classes).index.tolist()
        cm = cm.loc[totals, totals]

    row_sums = cm.sum(axis=1).replace(0, 1)
    cm_pct = cm.div(row_sums, axis=0)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(cm_pct.values, aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    ax.set_xticks(range(len(cm_pct.columns)))
    ax.set_yticks(range(len(cm_pct.index)))
    ax.set_xticklabels(cm_pct.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_pct.index)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for i in range(len(cm_pct.index)):
        for j in range(len(cm_pct.columns)):
            value = cm_pct.iloc[i, j]
            if value >= 0.10:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path



def plot_when_predicted_matrix(
    input_path: Path,
    output_path: Path,
    max_classes: int = 12,
    title: str = "When the model predicts",
    x_label: str = "Actual",
    y_label: str = "Predicted",
    colorbar_label: str = "Share of predictions",
) -> Path:
    """Plot a prediction-normalized confusion matrix.

    The standard confusion matrix in this report is row-normalized by actual
    class: "when the actual pitch was X, what did the model predict?"

    This view answers the opposite question Joel asked: "when the model
    predicts X, what actually happened?"  It transposes the raw confusion
    matrix so rows are predicted classes and columns are actual classes, then
    normalizes each prediction row to 100%.
    """
    cm = pd.read_csv(input_path, index_col=0)

    if len(cm) > max_classes:
        actual_support = cm.sum(axis=1)
        predicted_support = cm.sum(axis=0)
        totals = actual_support.add(predicted_support, fill_value=0).sort_values(ascending=False)
        keep = totals.head(max_classes).index.tolist()
        keep = [label for label in keep if label in cm.index and label in cm.columns]
        cm = cm.loc[keep, keep]

    predicted_by_actual = cm.T
    row_sums = predicted_by_actual.sum(axis=1).replace(0, 1)
    cm_pct = predicted_by_actual.div(row_sums, axis=0)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(cm_pct.values, aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    ax.set_xticks(range(len(cm_pct.columns)))
    ax.set_yticks(range(len(cm_pct.index)))
    ax.set_xticklabels(cm_pct.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_pct.index)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for i in range(len(cm_pct.index)):
        for j in range(len(cm_pct.columns)):
            value = cm_pct.iloc[i, j]
            if value >= 0.10:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_pitch_distribution(features_path: Path, output_path: Path) -> Path:
    df = pd.read_parquet(features_path, columns=[TARGET])
    counts = df[TARGET].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(counts) * 0.35)))
    ax.barh(counts.index.astype(str), counts.values)
    ax.set_xlabel("Pitch count")
    ax.set_ylabel("Pitch type")
    ax.set_title("Official pitch-type distribution in feature data")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_pitch_class_distribution(features_path: Path, output_path: Path) -> Path:
    df = read_existing_parquet_columns(features_path, [TARGET, PITCH_CLASS])
    if PITCH_CLASS not in df.columns:
        df[PITCH_CLASS] = df[TARGET].map(pitch_type_to_class)
    counts = df[PITCH_CLASS].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(counts.index.astype(str), counts.values)
    ax.set_xlabel("Pitch count")
    ax.set_ylabel("Pitch class")
    ax.set_title("Pitch-class distribution in feature data")
    for i, value in enumerate(counts.values):
        ax.text(value + max(counts.values) * 0.01, i, f"{value:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).schema_arrow.names


def read_existing_parquet_columns(path: Path, desired_columns: list[str]) -> pd.DataFrame:
    existing = set(parquet_columns(path))
    columns = [col for col in desired_columns if col in existing]
    if not columns:
        return pd.DataFrame()
    return pd.read_parquet(path, columns=columns)


def plot_pitcher_pitch_mix(features_path: Path, output_path: Path, top_n: int = 20) -> Path:
    df = read_existing_parquet_columns(features_path, ["player_name", "pitcher", TARGET])
    if df.empty or "pitcher" not in df.columns or TARGET not in df.columns:
        raise ValueError(
            f"Could not build pitch-mix chart because {features_path} lacks pitcher and {TARGET} columns"
        )

    name_col = "player_name" if "player_name" in df.columns else "pitcher"
    rows = []

    group_cols = ["pitcher"]
    if name_col != "pitcher":
        group_cols.append(name_col)

    for pitcher_key, group in df.groupby(group_cols, dropna=False):
        if isinstance(pitcher_key, tuple):
            pitcher_id = pitcher_key[0]
            pitcher_name = pitcher_key[1] if len(pitcher_key) > 1 else pitcher_key[0]
        else:
            pitcher_id = pitcher_key
            pitcher_name = pitcher_key

        top_pitch = group[TARGET].value_counts().idxmax()
        top_share = float((group[TARGET] == top_pitch).mean())
        rows.append(
            {
                "pitcher": pitcher_id,
                "pitcher_name": str(pitcher_name),
                "pitches": len(group),
                "top_pitch": str(top_pitch),
                "top_pitch_share": top_share,
            }
        )

    mix = pd.DataFrame(rows).sort_values("pitches", ascending=False).head(top_n)
    mix = mix.sort_values("top_pitch_share", ascending=True)
    labels = mix.apply(
        lambda row: f"Pitcher {row['pitcher_name']}  ({row['top_pitch']})",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(mix) * 0.4)))
    ax.barh(labels, mix["top_pitch_share"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Most-used pitch share")
    ax.set_ylabel("Pitcher and pitch")
    ax.set_title(f"Most common pitch share for top {len(mix)} pitchers by sample size")
    for i, value in enumerate(mix["top_pitch_share"]):
        ax.text(value + 0.01, i, f"{value:.0%}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_html_index(
    path: Path,
    metrics: dict[str, Any],
    generated: dict[str, str],
    summary_path: Path,
    pitch_type_key_path: Path,
    notes: list[str] | None = None,
) -> Path:
    notes = notes or []
    summary_html = ""
    if summary_path.exists():
        summary_text = html.escape(summary_path.read_text(encoding="utf-8"))
        summary_html = f"<pre>{summary_text}</pre>"

    metric_rows = ""
    for key in [
        "rows_total",
        "rows_train",
        "rows_test",
        "split_date",
        "overall_majority_baseline_accuracy",
        "pitcher_count_baseline_accuracy",
        "context_model_accuracy",
        "overall_majority_baseline_pitch_class_accuracy",
        "pitcher_count_baseline_pitch_class_accuracy",
        "context_model_pitch_class_accuracy",
    ]:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float) and "accuracy" in key:
                value = f"{value:.1%}"
            metric_rows += f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>\n"

    if not metric_rows:
        metric_rows = "<tr><td colspan='2'>No metrics file found.</td></tr>"

    ordered_images = [
        ("accuracy_chart", "Accuracy comparison"),
        ("pitch_type_confusion_matrix_chart", "Pitch Type Confusion Matrix: Actual → Predicted"),
        ("pitch_type_when_predicted_chart", "Pitch Type Matrix: When the Model Predicts…"),
        ("pitch_class_confusion_matrix_chart", "Pitch Class Confusion Matrix: Actual → Predicted"),
        ("pitch_class_when_predicted_chart", "Pitch Class Matrix: When the Model Predicts…"),
        ("pitch_type_distribution_chart", "Pitch Type Distribution"),
        ("pitch_class_distribution_chart", "Pitch Class Distribution"),
        ("pitch_mix_chart", "Pitcher Pitch Mix"),
    ]

    image_blocks = []
    for key, title in ordered_images:
        file_path = generated.get(key)
        if not file_path:
            image_blocks.append(
                f"<h2>{html.escape(title)}</h2><p class='missing'>Not generated. Check the report-generation notes below.</p>"
            )
            continue
        rel = Path(file_path).name
        image_blocks.append(f"<h2>{html.escape(title)}</h2><img src='{html.escape(rel)}' alt='{html.escape(title)}'>")

    notes_html = ""
    if notes:
        notes_html = "<h2>Report-generation notes</h2><ul>" + "".join(
            f"<li>{html.escape(note)}</li>" for note in notes
        ) + "</ul>"

    generated_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{html.escape(str(Path(file_path).name))}</td></tr>"
        for label, file_path in generated.items()
    )

    pitch_key_html = ""
    if pitch_type_key_path.exists():
        key_df = pd.read_csv(pitch_type_key_path)
        rows = []
        for _, row in key_df.iterrows():
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(row['pitch_type']))}</td>"
                f"<td>{html.escape(str(row['pitch_name']))}</td>"
                f"<td>{html.escape(str(row['pitch_class']))}</td>"
                "</tr>"
            )
        pitch_key_html = """
  <h2>Pitch Type Key</h2>
  <table>
    <tr><th>Code</th><th>Official pitch name</th><th>Pitch class</th></tr>
    {rows}
  </table>
""".format(rows="\n".join(rows))

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pitcher AI Results</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin: 1rem 0 2rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
    pre {{ white-space: pre-wrap; background: #f8f8f8; padding: 16px; border-radius: 8px; }}
    .missing {{ color: #8a4b00; background: #fff6e6; border: 1px solid #ffd999; padding: 10px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>Pitcher AI Results</h1>
  <p>This report summarizes the context-only baseline. It does not use video yet.</p>
  <p>The standard confusion matrices are read row by row as <strong>Actual → Predicted</strong>. The “When the Model Predicts…” matrices are read row by row as <strong>Predicted → Actual</strong>, which answers questions like: when the model predicts Fastball, what actually happened?</p>
  <h2>Metrics</h2>
  <table>{metric_rows}</table>
  {''.join(image_blocks)}
  {pitch_key_html}
  <h2>Generated Files</h2>
  <table><tr><th>Item</th><th>File</th></tr>{generated_rows}</table>
  {notes_html}
  <h2>Summary</h2>
  {summary_html}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")
    return path
