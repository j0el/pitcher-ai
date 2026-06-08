from __future__ import annotations

from pathlib import Path

import pandas as pd

from pitcher_ai.config import PROCESSED_DIR, STATCAST_RAW_DIR
from pitcher_ai.util import console, date_chunks, ensure_dirs, parse_date


def _safe_filename(start_date: str, end_date: str) -> str:
    return f"statcast_{start_date}_to_{end_date}.parquet"


def ingest_statcast(
    start_date: str,
    end_date: str,
    chunk_days: int = 7,
    overwrite: bool = False,
    combine: bool = True,
) -> list[Path]:
    """Fetch pitch-level Statcast rows from Baseball Savant using pybaseball.

    Baseball Savant can be slow or fail on large ranges, so this function fetches
    a date range in smaller chunks and caches each chunk as Parquet.
    """
    from pybaseball import cache, statcast

    ensure_dirs([STATCAST_RAW_DIR, PROCESSED_DIR])
    cache.enable()

    start = parse_date(start_date)
    end = parse_date(end_date)
    written: list[Path] = []

    for chunk_start, chunk_end in date_chunks(start, end, chunk_days):
        chunk_start_s = chunk_start.isoformat()
        chunk_end_s = chunk_end.isoformat()
        out_path = STATCAST_RAW_DIR / _safe_filename(chunk_start_s, chunk_end_s)

        if out_path.exists() and not overwrite:
            console.print(f"Using existing {out_path}")
            written.append(out_path)
            continue

        console.print(f"Fetching Statcast rows {chunk_start_s} to {chunk_end_s}")
        df = statcast(start_dt=chunk_start_s, end_dt=chunk_end_s)

        if df is None or df.empty:
            console.print(f"No rows returned for {chunk_start_s} to {chunk_end_s}")
            continue

        df = _normalize_dataframe(df)
        df.to_parquet(out_path, index=False)
        console.print(f"Wrote {len(df):,} rows to {out_path}")
        written.append(out_path)

    if combine and written:
        combined_path = PROCESSED_DIR / "statcast_all.parquet"
        frames = [pd.read_parquet(path) for path in sorted(written)]
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=[col for col in ["game_pk", "at_bat_number", "pitch_number"] if col in combined.columns]
        )
        combined.to_parquet(combined_path, index=False)
        console.print(f"Wrote combined file with {len(combined):,} rows to {combined_path}")

    return written


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["game_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date.astype("string")
    return df
