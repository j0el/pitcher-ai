from __future__ import annotations

import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Iterator

import pandas as pd

from pitcher_ai.util import console


_STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
_SAVANT_SPORTY = "https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
_MP4_RE = re.compile(r'https://sporty-clips\.mlb\.com/[^"\'>\s]+\.mp4')

# (at_bat_number, pitch_number) -> play_id
PlayIdMap = dict[tuple[int, int], str]


def _get_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "pitcher-ai-research/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        import json
        return json.load(resp)


def build_play_id_map(game_pk: int) -> PlayIdMap:
    """Fetch MLB Stats API play-by-play and return a mapping of
    (at_bat_number, pitch_number) -> playId for all pitches in the game.

    MLB Stats API uses 0-based atBatIndex; Statcast uses 1-based at_bat_number.
    """
    url = f"{_STATSAPI_BASE}/game/{game_pk}/playByPlay"
    data = _get_json(url)
    mapping: PlayIdMap = {}
    for play in data.get("allPlays", []):
        ab_num = play["about"]["atBatIndex"] + 1  # convert to 1-based
        for ev in play.get("playEvents", []):
            if ev.get("isPitch") and ev.get("playId"):
                key = (ab_num, ev["pitchNumber"])
                mapping[key] = ev["playId"]
    return mapping


def resolve_mp4_url(play_id: str) -> str | None:
    """Fetch the Baseball Savant sporty-videos page and extract the direct .mp4 URL."""
    url = _SAVANT_SPORTY.format(play_id=play_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pitcher-ai-research/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        match = _MP4_RE.search(html)
        return match.group(0) if match else None
    except Exception:
        return None


def download_clip(mp4_url: str, dest: Path, chunk_size: int = 1 << 20) -> bool:
    """Stream-download an .mp4 to dest. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(mp4_url, headers={"User-Agent": "pitcher-ai-research/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as fh:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
        return True
    except Exception as exc:
        console.print(f"[yellow]Download failed for {dest.name}: {exc}[/yellow]")
        if dest.exists():
            dest.unlink()
        return False


def iter_pitch_sample(
    features_path: Path,
    max_per_type: int = 20,
    pitch_types: list[str] | None = None,
    pitcher_id: int | None = None,
    seed: int = 42,
) -> Iterator[pd.Series]:
    """Yield up to max_per_type rows per pitch_type from the features parquet."""
    df = pd.read_parquet(features_path, columns=[
        "game_pk", "at_bat_number", "pitch_number",
        "pitch_type", "pitcher", "batter", "game_date",
    ])
    if pitcher_id is not None:
        df = df[df["pitcher"] == pitcher_id]
    if pitch_types:
        df = df[df["pitch_type"].isin(pitch_types)]

    groups = df.groupby("pitch_type")
    for _, grp in groups:
        sample = grp.sample(n=min(max_per_type, len(grp)), random_state=seed)
        for _, row in sample.iterrows():
            yield row


def fetch_videos(
    features_path: Path,
    output_dir: Path,
    manifest_path: Path,
    max_per_type: int = 20,
    pitch_types: list[str] | None = None,
    pitcher_id: int | None = None,
    request_delay: float = 0.5,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Main entry point: download pitch clips and write/update the manifest CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create existing manifest
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = pd.DataFrame(columns=[
            "game_pk", "at_bat_number", "pitch_number",
            "pitcher", "batter", "pitch_type", "game_date",
            "play_id", "source_url", "local_clip_path", "status",
        ])

    # Cache play_id maps by game to avoid redundant API calls
    play_id_cache: dict[int, PlayIdMap] = {}
    rows: list[dict] = []

    for row in iter_pitch_sample(features_path, max_per_type, pitch_types, pitcher_id):
        game_pk = int(row["game_pk"])
        ab = int(row["at_bat_number"])
        pn = int(row["pitch_number"])
        pt = str(row["pitch_type"])

        clip_name = f"{game_pk}_{ab}_{pn}_{pt}.mp4"
        clip_path = output_dir / pt / clip_name

        # Skip if already downloaded and manifest exists
        if not overwrite and clip_path.exists():
            console.print(f"[dim]skip {clip_name} (exists)[/dim]")
            rows.append({**row.to_dict(), "play_id": None, "source_url": None,
                         "local_clip_path": str(clip_path), "status": "cached"})
            continue

        # Fetch play_id map for the game (cached)
        if game_pk not in play_id_cache:
            console.print(f"Fetching play-by-play for game {game_pk}…")
            try:
                play_id_cache[game_pk] = build_play_id_map(game_pk)
            except Exception as exc:
                console.print(f"[red]Failed to fetch game {game_pk}: {exc}[/red]")
                play_id_cache[game_pk] = {}
            time.sleep(request_delay)

        play_id = play_id_cache[game_pk].get((ab, pn))
        if not play_id:
            console.print(f"[yellow]No playId for {game_pk} ab={ab} p={pn}[/yellow]")
            rows.append({**row.to_dict(), "play_id": None, "source_url": None,
                         "local_clip_path": None, "status": "no_play_id"})
            continue

        # Resolve .mp4 URL from Baseball Savant
        mp4_url = resolve_mp4_url(play_id)
        time.sleep(request_delay)
        if not mp4_url:
            console.print(f"[yellow]No mp4 URL for playId {play_id}[/yellow]")
            rows.append({**row.to_dict(), "play_id": play_id, "source_url": None,
                         "local_clip_path": None, "status": "no_mp4_url"})
            continue

        # Download the clip
        console.print(f"Downloading {pt} pitch {clip_name}…")
        ok = download_clip(mp4_url, clip_path)
        time.sleep(request_delay)

        status = "ok" if ok else "download_failed"
        rows.append({**row.to_dict(), "play_id": play_id, "source_url": mp4_url,
                     "local_clip_path": str(clip_path) if ok else None, "status": status})

    new_df = pd.DataFrame(rows)
    if not new_df.empty:
        manifest = pd.concat([manifest, new_df], ignore_index=True).drop_duplicates(
            subset=["game_pk", "at_bat_number", "pitch_number"]
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    console.print(f"Manifest written to {manifest_path} ({len(manifest)} rows)")
    return manifest
