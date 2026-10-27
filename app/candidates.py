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
    "Current Taste (Default)": ["top hits 2024", "indie favorites", "pop hits", "rock classics", "r&b chill"],
    "Chill & Cozy": ["chill indie", "acoustic bedroom pop", "lofi sunset", "calm vibes"],
    "High Energy Workout": ["high energy pop", "workout motivation", "fast electronic", "hype hip hop"],
    "Deep Focus & Late Night": ["ambient electronic", "deep focus study", "synthwave night", "downtempo"],
    "Upbeat Party": ["dance party hits", "upbeat disco pop", "club bangers", "groove funk"],
    "Sad & Melancholic": ["sad acoustic", "heartbreak indie", "melancholy ballad", "rainy day vibes"],
    "Late Night Drive": ["synthwave drive", "night aesthetic indie", "after hours r&b", "indie roadtrip"],
}

# High-fidelity fallback catalog for Universal Mode, offline mode, or free-tier developer apps
FALLBACK_RAW_CANDIDATES: list[tuple[dict[str, Any], str]] = [
    ({"id": "7LVHVU3tWfcxj5aiPFEW4Q", "name": "Fix You", "artists": [{"name": "Coldplay"}], "album": {"name": "X&Y", "images": [{"url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300"}]}, "popularity": 86}, "alternative rock"),
    ({"id": "3EdSembr1bA6eOCu999YpM", "name": "August", "artists": [{"name": "Taylor Swift"}], "album": {"name": "folklore", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}, "popularity": 88}, "indie pop"),
    ({"id": "7MXVkk9YM5IZxh0VU69um0", "name": "Starboy", "artists": [{"name": "The Weeknd"}], "album": {"name": "Starboy", "images": [{"url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300"}]}, "popularity": 91}, "r&b"),
    ({"id": "2X485T9Z5Ly0xyaghY73bv", "name": "Let It Happen", "artists": [{"name": "Tame Impala"}], "album": {"name": "Currents", "images": [{"url": "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=300"}]}, "popularity": 85}, "psychedelic pop"),
    ({"id": "2dBwB6KUHB76GYanBk79bS", "name": "Summertime Sadness", "artists": [{"name": "Lana Del Rey"}], "album": {"name": "Born to Die", "images": [{"url": "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?w=300"}]}, "popularity": 87}, "dream pop"),
    ({"id": "2AT8i7KDTpe0pPieSnLrg2", "name": "R U Mine?", "artists": [{"name": "Arctic Monkeys"}], "album": {"name": "AM", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}, "popularity": 89}, "garage rock"),
    ({"id": "3ZCTVFBt2Br03R4RqlxExQ", "name": "Everything I Wanted", "artists": [{"name": "Billie Eilish"}], "album": {"name": "Single", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}, "popularity": 84}, "electropop"),
    ({"id": "3PfIrDoz19wz7qK7tYeu62", "name": "Don't Start Now", "artists": [{"name": "Dua Lipa"}], "album": {"name": "Future Nostalgia", "images": [{"url": "https://images.unsplash.com/photo-1487180144351-b8472da7d491?w=300"}]}, "popularity": 87}, "dance pop"),
    ({"id": "6AQbmUe0Qwf5PZvg4hm5ez", "name": "Kesariya", "artists": [{"name": "Arijit Singh"}], "album": {"name": "Brahmastra", "images": [{"url": "https://images.unsplash.com/photo-1526478806334-5fd488fcaabc?w=300"}]}, "popularity": 85}, "bollywood"),
    ({"id": "3KkXRkHbMCARz0aVfEt68P", "name": "Sunflower", "artists": [{"name": "Post Malone"}], "album": {"name": "Spider-Man Soundtrack", "images": [{"url": "https://images.unsplash.com/photo-1445985543470-41fdd5c31447?w=300"}]}, "popularity": 90}, "pop rap"),
    ({"id": "6GyFP1nfCDB8lbD2bG0Hq9", "name": "Midnight City", "artists": [{"name": "M83"}], "album": {"name": "Hurry Up, We're Dreaming", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}, "popularity": 83}, "synth-pop"),
    ({"id": "4m0q0xQ2R518296k1xT17Q", "name": "Somebody Else", "artists": [{"name": "The 1975"}], "album": {"name": "I like it when you sleep", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}, "popularity": 84}, "indie pop"),
    ({"id": "2QjOHCTQ1Jl3zawyva09GQ", "name": "Sweater Weather", "artists": [{"name": "The Neighbourhood"}], "album": {"name": "I Love You.", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}, "popularity": 92}, "indie rock"),
    ({"id": "02MWAaffLxlfxAUY7c5dvx", "name": "Heat Waves", "artists": [{"name": "Glass Animals"}], "album": {"name": "Dreamland", "images": [{"url": "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=300"}]}, "popularity": 89}, "indie pop"),
    ({"id": "4LRPiXqCikLlN15c3yZAea", "name": "As It Was", "artists": [{"name": "Harry Styles"}], "album": {"name": "Harry's House", "images": [{"url": "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?w=300"}]}, "popularity": 93}, "pop"),
    ({"id": "0ug5KNajqBz3urNdteOmRa", "name": "Style", "artists": [{"name": "Taylor Swift"}], "album": {"name": "1989", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}, "popularity": 90}, "pop"),
    ({"id": "2HfSV0sR2M4fFw6s3b192n", "name": "Fluorescent Adolescent", "artists": [{"name": "Arctic Monkeys"}], "album": {"name": "Favourite Worst Nightmare", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}, "popularity": 88}, "indie rock"),
    ({"id": "7szuecWvxnjufLfdvdo9dC", "name": "In Your Eyes", "artists": [{"name": "The Weeknd"}], "album": {"name": "After Hours", "images": [{"url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300"}]}, "popularity": 86}, "synth-pop"),
    ({"id": "4RVwu0g322qgAzOiTWJFDU", "name": "Happier Than Ever", "artists": [{"name": "Billie Eilish"}], "album": {"name": "Happier Than Ever", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}, "popularity": 88}, "alt pop"),
    ({"id": "7D0RhFdozyUMYe5qAmvPgQ", "name": "Sparks", "artists": [{"name": "Coldplay"}], "album": {"name": "Parachutes", "images": [{"url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300"}]}, "popularity": 87}, "acoustic rock"),
    ({"id": "1mea3bSkSGXuIRvnydlB5b", "name": "Viva La Vida", "artists": [{"name": "Coldplay"}], "album": {"name": "Viva La Vida", "images": [{"url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=300"}]}, "popularity": 89}, "baroque pop"),
    ({"id": "1BxfuPKGuaTgP7aM0XbdMe", "name": "Cruel Summer", "artists": [{"name": "Taylor Swift"}], "album": {"name": "Lover", "images": [{"url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300"}]}, "popularity": 93}, "synth-pop"),
    ({"id": "0BxE48rCVxDnnRwwD0eAaD", "name": "505", "artists": [{"name": "Arctic Monkeys"}], "album": {"name": "Favourite Worst Nightmare", "images": [{"url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300"}]}, "popularity": 88}, "indie rock"),
    ({"id": "5QO79kh1waicV47BqGRIO3", "name": "Save Your Tears", "artists": [{"name": "The Weeknd"}], "album": {"name": "After Hours", "images": [{"url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300"}]}, "popularity": 90}, "synth-pop"),
    ({"id": "2Fxmhks0bxGSBdJ92vM42m", "name": "bad guy", "artists": [{"name": "Billie Eilish"}], "album": {"name": "WHEN WE ALL FALL ASLEEP", "images": [{"url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=300"}]}, "popularity": 87}, "electropop"),
    ({"id": "463CkQjx2Zk1yXoBuierM9", "name": "Levitating", "artists": [{"name": "Dua Lipa"}], "album": {"name": "Future Nostalgia", "images": [{"url": "https://images.unsplash.com/photo-1487180144351-b8472da7d491?w=300"}]}, "popularity": 86}, "disco pop"),
    ({"id": "6K4t31amVTZDgR3sKmwUJJ", "name": "The Less I Know The Better", "artists": [{"name": "Tame Impala"}], "album": {"name": "Currents", "images": [{"url": "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=300"}]}, "popularity": 87}, "psychedelic pop"),
    ({"id": "2NYGgN617d5n347sZ66236", "name": "Video Games", "artists": [{"name": "Lana Del Rey"}], "album": {"name": "Born to Die", "images": [{"url": "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?w=300"}]}, "popularity": 80}, "dream pop"),
    ({"id": "56zZ48jdyY2oDXHVRY2K5g", "name": "Tum Hi Ho", "artists": [{"name": "Arijit Singh"}], "album": {"name": "Aashiqui 2", "images": [{"url": "https://images.unsplash.com/photo-1526478806334-5fd488fcaabc?w=300"}]}, "popularity": 82}, "romantic ballad"),
    ({"id": "21jGcNKet2qwijlDFuPiPb", "name": "Circles", "artists": [{"name": "Post Malone"}], "album": {"name": "Hollywood's Bleeding", "images": [{"url": "https://images.unsplash.com/photo-1445985543470-41fdd5c31447?w=300"}]}, "popularity": 88}, "pop rock"),
]


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

    # Fallback default queries if no artists/genres available
    if not queries:
        queries = [
            ("top hits", "pop"),
            ("indie rock", "indie"),
            ("chill r&b", "r&b"),
            ("synth-pop", "electronic"),
            ("acoustic ballad", "acoustic"),
            ("alternative rock", "rock"),
        ]

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
    consecutive_errors = 0

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
            consecutive_errors = 0
        except Exception:
            consecutive_errors += 1
            # If Spotify API throws 403 or errors consecutively, break early to avoid delay
            if consecutive_errors >= 2 and not candidates_raw:
                break
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
    Falls back to a rich built-in catalog if Spotify search is unavailable or returns 403.
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

    # Robust Fallback: If Spotify API returned 403 (Free-tier restriction) or few tracks
    if len(candidate_pool) < min(15, target_pool_size):
        fallback_candidates = list(FALLBACK_RAW_CANDIDATES)
        random.shuffle(fallback_candidates)
        for raw_track, genre_hint in fallback_candidates:
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
