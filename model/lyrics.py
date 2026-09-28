"""
Lyrics fetcher for the Love your Playlist model.

fetch_lyrics(artist, title) -> str | None

Uses LRCLIB (https://lrclib.net) — free, no API key. Caches results locally
in a SQLite database so the same track is never fetched twice.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CACHE_PATH = DATA_DIR / "lyrics_cache.sqlite"

LRCLIB_URL = "https://lrclib.net/api/get"
USER_AGENT = "LoveYourPlaylist/0.1 (personal project)"

# be polite to the free API
_MIN_INTERVAL_SECONDS = 0.5
_last_request_time = 0.0


def _init_cache() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lyrics_cache (
            artist TEXT NOT NULL,
            title TEXT NOT NULL,
            lyrics TEXT,           -- NULL means "looked up, not found"
            fetched_at REAL NOT NULL,
            PRIMARY KEY (artist, title)
        )
        """
    )
    conn.commit()
    return conn


def _cache_key(artist: str, title: str) -> tuple[str, str]:
    return artist.strip().lower(), title.strip().lower()


def _clean_lyrics(raw: str) -> str:
    """Strip LRC timestamps like [00:12.34] and bracketed section tags."""
    no_timestamps = re.sub(r"\[\d{2}:\d{2}(\.\d{2,3})?\]", "", raw)
    no_tags = re.sub(r"^\s*\[[A-Za-z0-9 /'-]+\]\s*$", "", no_timestamps, flags=re.MULTILINE)
    lines = [line.strip() for line in no_tags.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_INTERVAL_SECONDS:
        time.sleep(_MIN_INTERVAL_SECONDS - elapsed)
    _last_request_time = time.monotonic()


def fetch_lyrics(artist: str, title: str, *, use_cache: bool = True) -> str | None:
    """
    Returns the lyrics text for (artist, title), or None if not found.
    Cached in data/lyrics_cache.sqlite, keyed by lowercased (artist, title).
    """
    conn = _init_cache()
    key = _cache_key(artist, title)

    if use_cache:
        row = conn.execute(
            "SELECT lyrics FROM lyrics_cache WHERE artist = ? AND title = ?", key
        ).fetchone()
        if row is not None:
            conn.close()
            return row[0]  # may legitimately be None (a cached "not found")

    lyrics: str | None = None
    try:
        _rate_limit()
        resp = requests.get(
            LRCLIB_URL,
            params={"artist_name": artist, "track_name": title},
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            raw = data.get("plainLyrics") or data.get("syncedLyrics")
            if raw:
                lyrics = _clean_lyrics(raw)
    except requests.RequestException:
        lyrics = None  # network hiccup — don't cache a false "not found"; just return None
        conn.close()
        return None

    conn.execute(
        "INSERT OR REPLACE INTO lyrics_cache (artist, title, lyrics, fetched_at) VALUES (?, ?, ?, ?)",
        (key[0], key[1], lyrics, time.time()),
    )
    conn.commit()
    conn.close()
    return lyrics


def cache_stats() -> dict:
    conn = _init_cache()
    total = conn.execute("SELECT COUNT(*) FROM lyrics_cache").fetchone()[0]
    found = conn.execute(
        "SELECT COUNT(*) FROM lyrics_cache WHERE lyrics IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return {"total_lookups": total, "found": found, "coverage": found / total if total else 0.0}


if __name__ == "__main__":
    test_tracks = [
        ("Coldplay", "Yellow"),
        ("Arijit Singh", "Tum Hi Ho"),
        ("Taylor Swift", "Cardigan"),
        ("A. R. Rahman", "Jai Ho"),
    ]
    for artist, title in test_tracks:
        result = fetch_lyrics(artist, title)
        status = f"{len(result)} chars" if result else "NOT FOUND"
        print(f"{artist} - {title}: {status}")
    print(cache_stats())
