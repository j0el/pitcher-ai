from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from pitcher_ai.util import console

# MediaPipe Tasks API
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    RunningMode,
)
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark as LM

MODEL_PATH = Path("models/pose_landmarker_lite.task")

# Landmark indices we care about
_L_SHOULDER = LM.LEFT_SHOULDER.value
_R_SHOULDER = LM.RIGHT_SHOULDER.value
_L_ELBOW = LM.LEFT_ELBOW.value
_R_ELBOW = LM.RIGHT_ELBOW.value
_L_WRIST = LM.LEFT_WRIST.value
_R_WRIST = LM.RIGHT_WRIST.value
_L_HIP = LM.LEFT_HIP.value
_R_HIP = LM.RIGHT_HIP.value


def _pt(lms: list, idx: int) -> np.ndarray:
    lm = lms[idx]
    return np.array([lm.x, lm.y, lm.z])


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by rays b->a and b->c, in degrees."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


class FrameFeatures(NamedTuple):
    # Elbow bend angles (degrees; 180 = straight arm)
    elbow_angle_pitch: float
    elbow_angle_glove: float
    # Pitching wrist position relative to pitching shoulder (normalized coords)
    wrist_above_shoulder_pitch: float   # positive = wrist above shoulder
    wrist_lateral_pitch: float          # signed: positive = toward plate side
    # Shoulder tilt: pitching shoulder y minus glove shoulder y (positive = pitch side higher)
    shoulder_tilt: float
    # Apparent shoulder width (proxy for rotation toward plate; shrinks as pitcher turns)
    shoulder_width_norm: float          # normalized by hip width to remove scale
    # Hip-shoulder separation: angle of shoulder line vs hip line (degrees)
    trunk_rotation: float


def _frame_features(lms: list, pitch_arm: str) -> FrameFeatures:
    """Compute biomechanical features from a single frame's landmarks."""
    ls = _pt(lms, _L_SHOULDER)
    rs = _pt(lms, _R_SHOULDER)
    le = _pt(lms, _L_ELBOW)
    re = _pt(lms, _R_ELBOW)
    lw = _pt(lms, _L_WRIST)
    rw = _pt(lms, _R_WRIST)
    lh = _pt(lms, _L_HIP)
    rh = _pt(lms, _R_HIP)

    # Pitching arm vs glove arm (MediaPipe: LEFT == pitcher's left, i.e., opposite of screen)
    # For a right-handed pitcher, the pitching arm is the right arm.
    if pitch_arm == "R":
        pa_s, pa_e, pa_w = rs, re, rw   # pitching
        ga_s, ga_e, ga_w = ls, le, lw   # glove
    else:
        pa_s, pa_e, pa_w = ls, le, lw
        ga_s, ga_e, ga_w = rs, re, rw

    elbow_pitch = _angle(pa_s, pa_e, pa_w)
    elbow_glove = _angle(ga_s, ga_e, ga_w)

    # Wrist relative to pitching shoulder (y: smaller = higher in image coords)
    wrist_above = pa_s[1] - pa_w[1]          # positive = wrist is above shoulder
    wrist_lateral = (pa_w[0] - pa_s[0]) * (1 if pitch_arm == "R" else -1)

    shoulder_tilt = (pa_s[1] - ga_s[1]) * (1 if pitch_arm == "R" else -1)

    hip_width = float(np.linalg.norm((rh - lh)[:2]) + 1e-9)
    sh_width = float(np.linalg.norm((rs - ls)[:2]))
    shoulder_width_norm = sh_width / hip_width

    # Trunk rotation: angle between shoulder vector and hip vector (both projected to XY)
    sh_vec = (rs - ls)[:2]
    hi_vec = (rh - lh)[:2]
    cos = np.dot(sh_vec, hi_vec) / (np.linalg.norm(sh_vec) * np.linalg.norm(hi_vec) + 1e-9)
    trunk_rot = math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))

    return FrameFeatures(
        elbow_angle_pitch=elbow_pitch,
        elbow_angle_glove=elbow_glove,
        wrist_above_shoulder_pitch=float(wrist_above),
        wrist_lateral_pitch=float(wrist_lateral),
        shoulder_tilt=float(shoulder_tilt),
        shoulder_width_norm=float(shoulder_width_norm),
        trunk_rotation=float(trunk_rot),
    )


def _aggregate(frames: list[FrameFeatures]) -> dict[str, float]:
    """Collapse per-frame features into a fixed-length clip feature vector."""
    arr = np.array(frames, dtype=float)   # (N, 7)
    fields = FrameFeatures._fields
    out: dict[str, float] = {"pose_frames_detected": float(len(frames))}
    for i, name in enumerate(fields):
        col = arr[:, i]
        out[f"{name}_mean"] = float(col.mean())
        out[f"{name}_std"] = float(col.std())
        out[f"{name}_min"] = float(col.min())
        out[f"{name}_max"] = float(col.max())
        # First and last frame capture stance start vs. release approach
        out[f"{name}_first"] = float(col[0])
        out[f"{name}_last"] = float(col[-1])
    return out


def extract_clip_features(
    clip_path: Path,
    pitch_arm: str,
    model_path: Path = MODEL_PATH,
    min_confidence: float = 0.3,
) -> dict[str, float] | None:
    """Run MediaPipe Pose on every frame of clip_path and return aggregated features.

    Returns None if fewer than 3 frames have a detected pose.
    """
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_features: list[FrameFeatures] = []

    with PoseLandmarker.create_from_options(options) as detector:
        for _ in range(total_frames):
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result: PoseLandmarkerResult = detector.detect(mp_img)
            if result.pose_landmarks:
                try:
                    ff = _frame_features(result.pose_landmarks[0], pitch_arm)
                    frame_features.append(ff)
                except Exception:
                    pass

    cap.release()

    if len(frame_features) < 3:
        return None
    return _aggregate(frame_features)


def extract_pose_features(
    manifest_path: Path,
    statcast_path: Path,
    output_path: Path,
    model_path: Path = MODEL_PATH,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Process all successfully downloaded clips in the manifest and write pose features."""
    if output_path.exists() and not overwrite:
        console.print(f"[dim]{output_path} exists, skipping. Use --overwrite to re-run.[/dim]")
        return pd.read_parquet(output_path)

    manifest = pd.read_csv(manifest_path)
    ok_clips = manifest[(manifest["status"] == "ok") & manifest["local_clip_path"].notna()].copy()
    console.print(f"Processing {len(ok_clips)} clips from manifest")

    # Pull pitcher handedness from Statcast data
    sc = pd.read_parquet(statcast_path, columns=["game_pk", "at_bat_number", "pitch_number", "p_throws"])
    ok_clips = ok_clips.merge(sc, on=["game_pk", "at_bat_number", "pitch_number"], how="left")

    records: list[dict] = []
    for _, row in ok_clips.iterrows():
        clip_path = Path(str(row["local_clip_path"]))
        if not clip_path.exists():
            console.print(f"[yellow]Missing: {clip_path.name}[/yellow]")
            continue

        pitch_arm = str(row.get("p_throws", "R"))
        console.print(f"  {clip_path.name} ({row['pitch_type']}, {pitch_arm})")

        feats = extract_clip_features(clip_path, pitch_arm, model_path=model_path)
        if feats is None:
            console.print(f"    [yellow]pose not detected[/yellow]")
            continue

        record = {
            "game_pk": row["game_pk"],
            "at_bat_number": row["at_bat_number"],
            "pitch_number": row["pitch_number"],
            "pitcher": row["pitcher"],
            "pitch_type": row["pitch_type"],
        }
        record.update(feats)
        records.append(record)

    if not records:
        console.print("[red]No pose features extracted.[/red]")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    console.print(f"Wrote {len(df)} rows -> {output_path}")
    return df
