"""
Day 2: load the offline Spotify tracks dataset, clean it, and take a working
sample.

Source: a public mirror of the "Spotify Tracks Dataset" (114k tracks, 20
columns, audio features included) — no Kaggle account/API key needed.
See data/DATASET_SOURCE.md for provenance.

Run:
    python prepare_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
FULL_CSV = DATA_DIR / "spotify_tracks_full.csv"
SAMPLE_CSV = DATA_DIR / "sample_tracks.csv"

AUDIO_FEATURE_COLS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

SAMPLE_SIZE = 8000


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(FULL_CSV)

    # drop the dataset's own unnamed index column if present
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    before = len(df)
    df = df.dropna(subset=["track_name", "artists"] + AUDIO_FEATURE_COLS)
    df = df.drop_duplicates(subset=["track_name", "artists"])
    after = len(df)
    print(f"Rows: {before} -> {after} after dropping missing/duplicate rows")

    # Spotify's Web API playlist calls need a real track id — keep only rows
    # that have one, so every sampled track can later be added to a playlist.
    df = df[df["track_id"].notna() & (df["track_id"].str.len() > 0)]

    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample across genres so the working set isn't dominated by whichever
    genre happens to have the most rows.

    Deliberately avoids groupby(...).apply(...): as of pandas 3.0 that drops
    the grouping column from each group by default, which silently produced
    a sample where 99% of rows had a blank track_genre. Building an index
    list by hand and using .loc sidesteps that entirely.
    """
    genre_groups = df.groupby("track_genre").groups  # {genre: index labels}
    per_genre = max(1, n // len(genre_groups))

    rng = pd.Series(dtype="int64")  # unused, kept for clarity of random_state below
    selected_idx: list = []
    for genre, idx in genre_groups.items():
        idx = pd.Index(idx)
        take = min(len(idx), per_genre)
        selected_idx.extend(idx.to_series().sample(take, random_state=42).tolist())

    sampled = df.loc[selected_idx]

    if len(sampled) < n:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(min(len(remaining), n - len(sampled)), random_state=42)
        sampled = pd.concat([sampled, extra])

    return sampled.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle


def main() -> None:
    if not FULL_CSV.exists():
        raise SystemExit(
            f"{FULL_CSV} not found. Download it first (see data/DATASET_SOURCE.md)."
        )

    df = load_and_clean()
    sample = stratified_sample(df, SAMPLE_SIZE)

    keep_cols = ["track_id", "artists", "track_name", "track_genre"] + AUDIO_FEATURE_COLS
    sample = sample[keep_cols]
    sample.to_csv(SAMPLE_CSV, index=False)

    print(f"\nSample: {len(sample)} tracks -> {SAMPLE_CSV}")
    print(f"Genres represented: {sample['track_genre'].nunique()} / {df['track_genre'].nunique()}")
    print(f"Unique artists in sample: {sample['artists'].nunique()}")
    print("\nAudio feature ranges (sanity check):")
    print(sample[AUDIO_FEATURE_COLS].describe().loc[["min", "max", "mean"]].round(3))


if __name__ == "__main__":
    main()
