"""
Search-based Candidate Track Fetcher for Person B.

Addresses Spotify API Changes (2026):
- Spotify removed the Recommendations & Related Artists endpoints for new applications.
- Spotify caps /v1/search queries at 10 results per request in Development Mode.

Solution:
- Builds a candidate pool by executing diverse, targeted search queries across:
    1. User's top artists (e.g., 'artist:"Coldplay"')
    2. User's top genres (e.g., 'genre:"indie pop"', 'genre:"rock"')
    3. Mood / Vibe keyword variations (e.g., 'chill indie 2024', 'synthwave pop')
    4. Related / popular queries with genre + year filters
- Deduplicates tracks against each other and filters out tracks already in the user's library.
- Converts all candidates into the shared track format.
"""

from __future__ import annotations

import random
import re
from typing import Any

from app.library import spotify_track_to_shared_format

MOOD_PRESETS: dict[str, list[str]] = {
    "Current Taste (Default)": [],
    "Chill & Cozy": ["chill indie", "acoustic bedroom pop", "lofi sunset", "calm vibes"],
    "High Energy Workout": ["high energy pop", "workout motivation", "fast electronic", "hype hip hop"],
    "Deep Focus & Late Night": ["ambient electronic", "deep focus study", "synthwave night", "downtempo"],
    "Upbeat Party": ["dance party hits", "upbeat disco pop", "club bangers", "groove funk"],
    "Sad & Melancholic": ["sad acoustic", "heartbreak indie", "melancholy ballad", "rainy day vibes"],
    "Late Night Drive": ["synthwave drive", "night aesthetic indie", "after hours r&b", "indie roadtrip"],
}


def clean_query_term(term: str) -> str:
    """Cleans artist names or genres for safe use in Spotify search query syntax."""
    cleaned = re.sub(r'["\:\(\)\[\]]', "", term).strip()
    return cleaned


def build_search_queries(
    top_artists: list[dict[str, Any]],
    top_genres: list[tuple[str, int]],
    mood: str | None = None,
    max_queries: int = 25,
) -> list[tuple[str, str]]:
    """Generates a list of (search_query, genre_hint) tuples."""
    queries: list[tuple[str, str]] = []

    # 1. Top Artist Queries
    for artist in top_artists[:8]:
        name = artist.get("name")
        if name:
            clean_name = clean_query_term(name)
            genres = artist.get("genres", [])
            primary_genre = genres[0] if genres else ""
            queries.append((f'artist:"{clean_name}"', primary_genre))

    # 2. Top Genre Queries with Year Windows
    years = ["year:2020-2026", "year:2015-2020", "year:2010-2015"]
    for genre, _ in top_genres[:6]:
        clean_genre = clean_query_term(genre)
        if clean_genre:
            queries.append((f'genre:"{clean_genre}"', clean_genre))
            y = random.choice(years)
            queries.append((f'genre:"{clean_genre}" {y}', clean_genre))

    # 3. Mood-based queries if selected
    if mood and mood in MOOD_PRESETS and MOOD_PRESETS[mood]:
        for kw in MOOD_PRESETS[mood]:
            queries.append((kw, mood))

    random.shuffle(queries)
    return queries[:max_queries]


def fetch_candidate_tracks_from_spotify(
    sp,
    queries: list[tuple[str, str]],
    *,
    limit_per_query: int = 10,
) -> list[tuple[dict[str, Any], str]]:
    """Executes Spotify search queries and collects candidate raw track objects with genre hints."""
    candidates_raw: list[tuple[dict[str, Any], str]] = []
    seen_ids: set[str] = set()

    for q_str, genre_hint in queries:
        try:
            results = sp.search(q=q_str, type="track", limit=min(10, limit_per_query))
            tracks_obj = results.get("tracks", {})
            items = tracks_obj.get("items", [])
            for item in items:
                tid = item.get("id")
                if tid and tid not in seen_ids:
                    seen_ids.add(tid)
                    candidates_raw.append((item, genre_hint))
        except Exception:
            continue

    return candidates_raw


def generate_candidate_pool(
    sp,
    user_tracks: list[dict[str, Any]],
    top_artists: list[dict[str, Any]],
    top_genres: list[tuple[str, int]],
    target_pool_size: int = 100,
    mood_preset: str = "Current Taste (Default)",
    lyrics_lookup: bool = True,
    scaler: Any = None,
) -> list[dict[str, Any]]:
    """
    Generates a full, de-duplicated candidate track pool converted to the shared track format.
    Excludes tracks already present in the user's top tracks / library.
    """
    user_spotify_ids = {
        t.get("spotify_id") for t in user_tracks if t.get("spotify_id")
    }
    user_artist_titles = {
        (str(t.get("artist", "")).strip().lower(), str(t.get("title", "")).strip().lower())
        for t in user_tracks
    }

    needed_queries = max(15, min(40, (target_pool_size // 5) + 10))
    queries = build_search_queries(
        top_artists, top_genres, mood=mood_preset, max_queries=needed_queries
    )

    raw_candidates = fetch_candidate_tracks_from_spotify(
        sp, queries, limit_per_query=10
    )

    candidate_pool: list[dict[str, Any]] = []
    seen_pool_keys: set[tuple[str, str]] = set()

    for raw_track, genre_hint in raw_candidates:
        tid = raw_track.get("id")
        title = raw_track.get("name", "")
        artists = raw_track.get("artists", [])
        primary_artist = artists[0].get("name", "") if artists else ""
        key = (primary_artist.strip().lower(), title.strip().lower())

        if tid in user_spotify_ids or key in user_artist_titles or key in seen_pool_keys:
            continue

        seen_pool_keys.add(key)

        shared_track = spotify_track_to_shared_format(
            raw_track,
            lyrics_lookup=lyrics_lookup,
            genre_hint=genre_hint,
            scaler=scaler,
        )
        candidate_pool.append(shared_track)

        if len(candidate_pool) >= target_pool_size:
            break

    return candidate_pool
