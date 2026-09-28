"""
Assembles the shared track format (see ../docs/track-format.md) for every
track in data/sample_tracks.csv, using whatever lyrics are already cached
(from lyrics.py's cache) and the dataset's own audio features.

Tracks with no cached lyrics yet still get built — just with a null lyric
embedding, so they aren't dropped, and can be re-processed later once more
lyrics are fetched (fetch_all_lyrics.py does the fetching).

Run:
    python build.py
Output:
    data/tracks_built.parquet  — one row per track, embeddings included
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from embeddings import (
    AUDIO_FEATURE_COLS,
    embed_lyrics,
    fit_audio_scaler,
    normalize_audio_features,
)
from lyrics import CACHE_PATH

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_CSV = DATA_DIR / "sample_tracks.csv"
OUTPUT_PATH = DATA_DIR / "tracks_built.parquet"


def load_lyrics_cache() -> dict[tuple[str, str], str | None]:
    """Returns {(artist_lower, title_lower): lyrics_or_None} for everything
    already fetched by lyrics.py. Empty dict if nothing's been fetched yet."""
    if not CACHE_PATH.exists():
        return {}
    conn = sqlite3.connect(CACHE_PATH)
    rows = conn.execute("SELECT artist, title, lyrics FROM lyrics_cache").fetchall()
    conn.close()
    return {(artist, title): lyrics for artist, title, lyrics in rows}


def build_track(row: pd.Series, lyrics_cache: dict, scaler) -> dict:
    """row: one row of sample_tracks.csv. Returns a dict in the shared
    track format."""
    primary_artist = str(row["artists"]).split(";")[0].strip()
    cache_key = (primary_artist.lower(), str(row["track_name"]).strip().lower())
    lyrics_text = lyrics_cache.get(cache_key)  # None if not fetched / not found

    raw_audio = {col: row[col] for col in AUDIO_FEATURE_COLS}
    normalized_audio = normalize_audio_features(raw_audio, scaler)

    lyric_vec = embed_lyrics(lyrics_text)  # None if no lyrics

    return {
        "spotify_id": row["track_id"],
        "title": row["track_name"],
        "artist": primary_artist,
        "genre": row["track_genre"],
        "lyrics_text": lyrics_text,
        "audio_features": raw_audio,
        "audio_features_normalized": normalized_audio,
        "lyric_embedding": lyric_vec,
        "has_lyrics": lyric_vec is not None,
    }


def main() -> None:
    df = pd.read_csv(SAMPLE_CSV)
    lyrics_cache = load_lyrics_cache()
    print(f"Loaded {len(df)} tracks, {len(lyrics_cache)} cached lyrics lookups")

    # Fit (or refit) the scaler on the FULL sample's raw audio features —
    # every track's normalized vector must come from the same scaler.
    scaler = fit_audio_scaler(df[AUDIO_FEATURE_COLS].to_numpy())

    built = [
        build_track(row, lyrics_cache, scaler)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building tracks")
    ]

    has_lyrics_count = sum(t["has_lyrics"] for t in built)
    print(f"\n{has_lyrics_count}/{len(built)} tracks have a lyric embedding "
          f"({has_lyrics_count / len(built):.1%})")

    out_df = pd.DataFrame(built)
    out_df.to_parquet(OUTPUT_PATH)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
