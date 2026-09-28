"""
Day 3: measure real lyrics coverage on the sample dataset.

Run this on your own machine (not in a sandbox) since it needs to reach
lrclib.net. Fetches lyrics for the first N tracks of data/sample_tracks.csv,
using the cache in lyrics.py, and reports the coverage rate.

Run:
    python check_lyrics_coverage.py [n]
"""

import sys
from pathlib import Path

import pandas as pd

from lyrics import cache_stats, fetch_lyrics

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_CSV = DATA_DIR / "sample_tracks.csv"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def main() -> None:
    df = pd.read_csv(SAMPLE_CSV).head(N)
    found = 0
    not_found = []

    for i, row in df.iterrows():
        # dataset lists multiple artists as "A;B;C" — try the first artist
        primary_artist = str(row["artists"]).split(";")[0].strip()
        lyrics = fetch_lyrics(primary_artist, row["track_name"])
        if lyrics:
            found += 1
        else:
            not_found.append(f"{primary_artist} - {row['track_name']}")
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(df)} done, {found} found so far")

    print(f"\nChecked {len(df)} tracks: {found} found ({found / len(df):.1%} coverage)")
    print(f"Cumulative cache stats: {cache_stats()}")

    if not_found:
        print(f"\nFirst 10 not found:")
        for line in not_found[:10]:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
