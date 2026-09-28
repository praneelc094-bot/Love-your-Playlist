"""
Fetches lyrics for every track in data/sample_tracks.csv (all 8,000, unless
you pass a smaller number) and caches them in data/lyrics_cache.sqlite.

This is the real data-collection pass (check_lyrics_coverage.py was just a
sanity check on 200). At ~0.5s/request that's roughly an hour for 8,000
tracks. Safe to stop (Ctrl+C) and re-run any time — already-cached tracks
are skipped instantly, so it resumes where it left off.

Run:
    python fetch_all_lyrics.py            # all 8000
    python fetch_all_lyrics.py 2000       # just the first 2000
"""

import sys
import time
from pathlib import Path

import pandas as pd

from lyrics import cache_stats, fetch_lyrics

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_CSV = DATA_DIR / "sample_tracks.csv"

N = int(sys.argv[1]) if len(sys.argv) > 1 else None


def main() -> None:
    df = pd.read_csv(SAMPLE_CSV)
    if N:
        df = df.head(N)

    start = time.time()
    found = 0
    skipped_cached = 0

    for i, row in df.iterrows():
        primary_artist = str(row["artists"]).split(";")[0].strip()
        before_stats = cache_stats()["total_lookups"]

        lyrics = fetch_lyrics(primary_artist, row["track_name"])

        after_stats = cache_stats()["total_lookups"]
        if after_stats == before_stats:
            skipped_cached += 1  # was already cached, no new lookup happened
        if lyrics:
            found += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            print(f"  ...{i + 1}/{len(df)} done | {found} found | "
                  f"{skipped_cached} already cached | {elapsed / 60:.1f} min elapsed")

    stats = cache_stats()
    print(f"\nDone. This run: {found} found out of {len(df)} tracks.")
    print(f"Cumulative cache: {stats}")


if __name__ == "__main__":
    main()
