from __future__ import annotations

from pathlib import Path

import typer

from pitcher_ai.features import build_context_features
from pitcher_ai.statcast import ingest_statcast
from pitcher_ai.train_baseline import summarize_pitch_mix, train_context_model
from pitcher_ai.pitch_types import pitch_type_key_rows
from pitcher_ai.visualize import visualize_results
from pitcher_ai.util import console

app = typer.Typer(help="Pitcher AI research pipeline")


@app.command()
def ingest(
    start_date: str = typer.Option(..., help="Start date as YYYY-MM-DD"),
    end_date: str = typer.Option(..., help="End date as YYYY-MM-DD"),
    chunk_days: int = typer.Option(7, help="Days per Statcast fetch chunk"),
    overwrite: bool = typer.Option(False, help="Re-fetch chunks that already exist"),
) -> None:
    ingest_statcast(
        start_date=start_date,
        end_date=end_date,
        chunk_days=chunk_days,
        overwrite=overwrite,
        combine=True,
    )


@app.command("build-features")
def build_features(
    raw_glob: str = typer.Option("data/raw/statcast/*.parquet", help="Input Statcast parquet glob"),
    output_path: Path = typer.Option(Path("data/processed/pitch_context_features.parquet")),
    min_pitch_type_count: int = typer.Option(100, help="Drop pitch types with fewer rows"),
) -> None:
    build_context_features(
        raw_glob=raw_glob,
        output_path=output_path,
        min_pitch_type_count=min_pitch_type_count,
    )


@app.command("train-baseline")
def train_baseline(
    features_path: Path = typer.Option(Path("data/processed/pitch_context_features.parquet")),
    model_path: Path = typer.Option(Path("models/context_model.joblib")),
    test_days: int = typer.Option(14, help="Use last N days as test set"),
    max_rows: int | None = typer.Option(None, help="Optional sample size for fast experiments"),
) -> None:
    train_context_model(
        features_path=features_path,
        model_path=model_path,
        test_days=test_days,
        max_rows=max_rows,
    )


@app.command("pitch-mix")
def pitch_mix(
    features_path: Path = typer.Option(Path("data/processed/pitch_context_features.parquet")),
    top_n: int = typer.Option(20),
) -> None:
    table = summarize_pitch_mix(features_path=features_path, top_n=top_n)
    console.print(table.to_string(index=False))


@app.command("pitch-type-key")
def pitch_type_key() -> None:
    import pandas as pd

    table = pd.DataFrame(pitch_type_key_rows())
    console.print(table.to_string(index=False))


@app.command("visualize-results")
def visualize_results_command(
    metrics_path: Path = typer.Option(Path("reports/context_model_metrics.json")),
    confusion_matrix_path: Path = typer.Option(Path("reports/context_model_confusion_matrix.csv")),
    class_confusion_matrix_path: Path = typer.Option(Path("reports/context_model_pitch_class_confusion_matrix.csv")),
    features_path: Path = typer.Option(Path("data/processed/pitch_context_features.parquet")),
    output_dir: Path = typer.Option(Path("reports")),
    top_n_pitchers: int = typer.Option(20),
    max_confusion_classes: int = typer.Option(12),
) -> None:
    visualize_results(
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        class_confusion_matrix_path=class_confusion_matrix_path,
        features_path=features_path,
        output_dir=output_dir,
        top_n_pitchers=top_n_pitchers,
        max_confusion_classes=max_confusion_classes,
    )


@app.command("fetch-videos")
def fetch_videos_command(
    features_path: Path = typer.Option(Path("data/processed/pitch_context_features.parquet")),
    output_dir: Path = typer.Option(Path("data/interim/video_clips")),
    manifest_path: Path = typer.Option(Path("data/interim/video_manifest.csv")),
    max_per_type: int = typer.Option(20, help="Max clips to download per pitch type"),
    pitch_types: str = typer.Option("", help="Comma-separated pitch types to include, e.g. FF,SL (empty=all)"),
    pitcher_id: int | None = typer.Option(None, help="Filter to a single pitcher MLBAM ID"),
    request_delay: float = typer.Option(0.5, help="Seconds to wait between HTTP requests"),
    overwrite: bool = typer.Option(False, help="Re-download clips that already exist"),
) -> None:
    """Download pitch video clips from Baseball Savant, tagged by pitch type."""
    from pitcher_ai.video_fetch import fetch_videos

    types_list = [t.strip() for t in pitch_types.split(",") if t.strip()] or None
    fetch_videos(
        features_path=features_path,
        output_dir=output_dir,
        manifest_path=manifest_path,
        max_per_type=max_per_type,
        pitch_types=types_list,
        pitcher_id=pitcher_id,
        request_delay=request_delay,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()
