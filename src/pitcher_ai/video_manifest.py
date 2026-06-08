from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class VideoClipCandidate:
    game_pk: int
    at_bat_number: int
    pitch_number: int
    pitcher: int
    batter: int
    pitch_type: str
    game_date: str
    local_clip_path: str | None = None
    source_url: str | None = None


def write_empty_manifest(path: Path = Path("data/interim/video_manifest.csv")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "pitcher",
        "batter",
        "pitch_type",
        "game_date",
        "local_clip_path",
        "source_url",
        "notes",
    ]
    pd.DataFrame(columns=columns).to_csv(path, index=False)
    return path
