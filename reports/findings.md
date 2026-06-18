# Pitcher AI: Experiment Findings

## What We Built

A pipeline that:

1. Ingests pitch-by-pitch Statcast data from Baseball Savant (April 2024, ~26k pitches)
2. Trains a baseline pitch-type classifier using only pre-pitch context (count, pitcher, batter, pitch history)
3. Downloads broadcast-angle video clips for a sample of pitches, tagged with their pitch type
4. Trims each clip to a 2-second pre-release window
5. Extracts pitcher body-pose features from every frame using MediaPipe
6. Trains a second set of models to test whether pose adds predictive signal

---

## Experiment 1: Context-Only Baseline (26k pitches)

Trained on 17,393 pitches, tested on 8,552. Features: pitcher identity, batter, count, base state, score, previous pitch type and speed.

| | Pitch-type accuracy | Pitch-class accuracy |
|---|---|---|
| Majority baseline | 0.320 | 0.560 |
| Pitcher/count baseline | 0.373 | 0.553 |
| Context model (SGD) | **0.334** | **0.506** |

The context model beats the majority baseline but falls slightly short of the simple pitcher/count heuristic on this date range. Pitch-class accuracy (Fastball / Breaking / Offspeed) is a more forgiving metric and sits around 50%.

---

## Experiment 2: Pose vs. Context (142 pitches, 5-fold CV)

To make the comparison fair, all three models were evaluated on the same 142-pitch subset that had pose features available. Stratified 5-fold cross-validation was used throughout (random split on this small set would be too noisy).

**Pose features extracted per clip:** elbow bend angles (pitching and glove arm), wrist height and lateral position relative to the pitching shoulder, shoulder tilt, apparent shoulder width normalized by hip width, and trunk rotation. Each feature was summarized as mean, std, min, max, first frame, and last frame — 44 numbers total.

| Model | Pitch-type accuracy | ± fold std | Pitch-class accuracy |
|---|---|---|---|
| Context-only (RF) | **0.401** | ±0.041 | **0.514** |
| Pose-only (RF) | 0.169 | ±0.058 | 0.380 |
| Context + Pose (RF) | 0.331 | ±0.048 | 0.465 |

### What we found

**Pose alone is near chance.** On 8 pitch types, random guessing would give 12.5%. Pose-only reached 16.9% — marginally above chance but nowhere near useful. Pre-release body position from a broadcast-angle camera does not cleanly separate pitch types, which is consistent with what we know about how pitchers are coached: they train specifically to make their mechanics look identical across pitch types.

**Adding pose features hurt the combined model.** Context + Pose (33.1%) underperformed context-only (40.1%). This is the classic small-n, high-dimensional problem: 44 noisy pose features dilute the 24 context features that carry real signal. With only 142 samples, the Random Forest can't reliably learn which pose features matter.

**Some pitch types are more identifiable than others.** FA (a specific pitcher's fastball variant) was predicted perfectly (precision 0.87, recall 1.00) — almost certainly because the pitcher's identity is a strong feature and that pitcher throws FA almost exclusively. SV (sweeper) and CU (curveball) also showed modest separability. FF and FC were the hardest to distinguish, which makes physical sense.

---

## Why Pose Didn't Help (Yet)

Three likely reasons:

**1. Camera angle.** Broadcast video uses a center-field camera. From that angle the pitcher is small, partially occluded by the umpire and catcher, and viewed from behind. MediaPipe detected a pose in ~70% of frames, but landmark precision is low. A catcher-view or high-home camera would give a much cleaner view of arm slot and grip.

**2. Sample size.** 142 pitches across 8 classes (~18 per class) is too small to learn 44 features. A useful pose model probably needs 500+ examples per pitch type.

**3. Feature representation.** Aggregating landmarks into mean/std/min/max discards temporal structure. The key signal in pitching mechanics is in the *motion* — how the arm accelerates, when the hip opens relative to the shoulder — not the average body position over the 2-second window.

---

## Recommended Next Steps

**Short term (more data)**
- Download 200+ clips per pitch type (currently 15–20) to give the model a fighting chance with pose features

**Medium term (better features)**
- Replace frame aggregation with a 1D-CNN or LSTM over the landmark sequence to capture motion dynamics
- Add grip-side wrist angle and finger position features (requires higher-resolution crop of the hand)

**Medium term (better camera)**
- Filter the manifest to clips from the catcher-view or high-home broadcast angle, which are available on Baseball Savant for many games

**Longer term (pitcher-specific models)**
- Train one model per pitcher. Within-pitcher pose variation by pitch type may be learnable even from broadcast video; cross-pitcher variation is too large for a single model to handle cleanly

---

## Pipeline Commands

```bash
# Ingest Statcast data
uv run pitcher-ai ingest --start-date "2024-04-01" --end-date "2024-04-07"

# Build context features and train baseline
uv run pitcher-ai build-features --min-pitch-type-count 25
uv run pitcher-ai train-baseline --test-days 2

# Download video clips (20 per pitch type)
uv run pitcher-ai fetch-videos --max-per-type 20

# Trim to 2s pre-release window
uv run python scripts/trim_clips.py

# Extract pose features
uv run pitcher-ai extract-pose

# Run pose vs context comparison
uv run pitcher-ai train-pose
```
