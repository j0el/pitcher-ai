# Pitcher AI

A Python and uv starter project for predicting MLB pitch type from pre-pitch context first, then adding pre-release video and pose features later.

The first milestone is deliberately not computer vision. It builds a Statcast-only baseline so we can later ask the important question:

Does pre-release video improve prediction beyond count, pitcher, batter, and pitch history?

## Local project path

Recommended location on Joel's Mac:

```bash
/Users/jberman/Projects/pitcher-ai
```

## Install

Unzip the project so the folder is exactly:

```bash
/Users/jberman/Projects/pitcher-ai
```

Then run:

```bash
cd "/Users/jberman/Projects/pitcher-ai"
uv sync
```

## First test run

Start with a small date range:

```bash
cd "/Users/jberman/Projects/pitcher-ai"
uv run pitcher-ai ingest --start-date "2024-04-01" --end-date "2024-04-07"
uv run pitcher-ai build-features --min-pitch-type-count 25
uv run pitcher-ai train-baseline --test-days 2
```

For a larger run:

```bash
cd "/Users/jberman/Projects/pitcher-ai"
uv run pitcher-ai ingest --start-date "2024-04-01" --end-date "2024-04-30" --chunk-days 7
uv run pitcher-ai build-features --min-pitch-type-count 100
uv run pitcher-ai train-baseline --test-days 7
```

## Outputs

```text
data/raw/statcast/                  cached Statcast chunks
data/processed/statcast_all.parquet combined raw pitch rows
data/processed/pitch_context_features.parquet pre-pitch training rows
models/context_model.joblib         trained baseline model
reports/context_model_metrics.json  accuracy and metadata
reports/context_model_report.txt    per-class classification report
reports/context_model_confusion_matrix.csv confusion matrix
reports/baseline_summary.md         short summary
```

## What the baseline uses

The baseline uses only information that should be known before the pitch:

```text
pitcher
batter
batter side
pitcher handedness
balls
strikes
outs
inning
score
base state
home and away team
previous pitch type in the at-bat
previous pitch speed in the at-bat
previous count
```

It intentionally excludes current-pitch leakage fields such as:

```text
release_speed
release_spin_rate
movement
plate_x
plate_z
zone
description
events
launch_speed
launch_angle
estimated_ba_using_speedangle
woba_value
```

Previous-pitch speed is allowed because that happened before the next pitch.

## Useful commands

Show basic pitch mix by pitcher:

```bash
cd "/Users/jberman/Projects/pitcher-ai"
uv run pitcher-ai pitch-mix --top-n 25
```

Run a faster training sample:

```bash
cd "/Users/jberman/Projects/pitcher-ai"
uv run pitcher-ai train-baseline --test-days 7 --max-rows 50000
```

Re-fetch a date range:

```bash
cd "/Users/jberman/Projects/pitcher-ai"
uv run pitcher-ai ingest --start-date "2024-04-01" --end-date "2024-04-07" --overwrite
```

## Suggested next milestones

1. Add a stronger context model after the baseline works.
2. Add pitcher-season pitch mix features without leaking future pitches.
3. Build a video manifest keyed by `game_pk`, `at_bat_number`, and `pitch_number`.
4. Trim clips to a pre-release window only.
5. Extract pose features.
6. Compare context-only, pose-only, and context-plus-pose models.

## Notes

This project is meant for research, player-development, and scouting-style analysis. It should not be used for in-game electronic sign stealing or real-time communication to hitters.
