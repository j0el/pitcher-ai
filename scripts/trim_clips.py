"""Trim downloaded pitch clips to a fixed pre-release window.

Usage:
    uv run python scripts/trim_clips.py

Reads data/interim/video_manifest.csv for successfully downloaded clips.
Writes trimmed clips to data/interim/video_clips_trimmed/{pitch_type}/.

The full broadcast clip typically runs 6-12 seconds.  We keep a configurable
window ending at the ball's release point.  Since we don't yet have per-frame
release detection, we use a heuristic: trim to the last PRE_RELEASE_SEC seconds
of each clip.  Once you have a release-frame detector this script can be updated
to anchor on that frame instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

PRE_RELEASE_SEC = 2.0   # seconds of footage to keep before release
OUTPUT_DIR = Path("data/interim/video_clips_trimmed")
MANIFEST = Path("data/interim/video_manifest.csv")


def get_duration(path: Path) -> float | None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def trim_clip(src: Path, dest: Path, duration_sec: float) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = get_duration(src)
    if total is None:
        print(f"  skip {src.name}: cannot read duration")
        return False
    start = max(0.0, total - duration_sec)
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", str(src),
         "-t", str(duration_sec), "-c", "copy", str(dest)],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  ffmpeg error for {src.name}: {result.stderr.decode()[:200]}")
        return False
    return True


def main() -> None:
    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}")
        sys.exit(1)

    df = pd.read_csv(MANIFEST)
    ok_rows = df[(df["status"] == "ok") & df["local_clip_path"].notna()]
    print(f"Trimming {len(ok_rows)} clips to {PRE_RELEASE_SEC}s pre-release window")

    success = 0
    for _, row in ok_rows.iterrows():
        src = Path(row["local_clip_path"])
        if not src.exists():
            print(f"  missing: {src}")
            continue
        pt = row["pitch_type"]
        dest = OUTPUT_DIR / pt / src.name
        if dest.exists():
            print(f"  cached: {dest.name}")
            success += 1
            continue
        print(f"  trimming {src.name} ({pt})")
        if trim_clip(src, dest, PRE_RELEASE_SEC):
            success += 1

    print(f"Done. {success}/{len(ok_rows)} clips trimmed -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
