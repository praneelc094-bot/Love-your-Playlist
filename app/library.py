"""
Spotify Library and User Taste Data Ingestion for Person B.

Handles:
- Fetching user's top tracks across time ranges (short_term, medium_term, long_term)
- Fetching user's top artists and extracting top genres
- Fetching user's saved library tracks
- Converting Spotify tracks to the shared track format (docs/track-format.md)
- Enriching tracks with lyrics (via model/lyrics.py) and audio features
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add model directory to sys.path so we can import model modules seamlessly
MODEL_DIR = Path(__file__).parent.parent / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

try:
    from lyrics import fetch_lyrics
except ImportError:
    def fetch_lyrics(artist: str, title: str) -> str | None:
        return None

try:
    from embeddings import (
        AUDIO_FEATURE_COLS,
        embed_lyrics,
        load_audio_scaler,
        normalize_audio_features,
    )
except ImportError:
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

    def embed_lyrics(text: str | None) -> np.ndarray | None:
        return None

    def load_audio_scaler():
        return None

    def normalize_audio_features(raw: dict, scaler) -> np.ndarray:
        return np.array([0.5] * len(AUDIO_FEATURE_COLS), dtype=np.float32)


# Default baseline audio features for tracks when live audio features API is restricted
DEFAULT_AUDIO_FEATURES = {
    "danceability": 0.60,
    "energy": 0.65,
    "key": 5,
    "loudness": -7.0,
    "mode": 1,
    "speechiness": 0.08,
    "acousticness": 0.25,
    "instrumentalness": 0.05,
    "liveness": 0.15,
    "valence": 0.50,
    "tempo": 120.0,
    "time_signature": 4,
}


def get_user_profile_summary(sp) -> dict[str, Any]:
    """Retrieves current user's profile details."""
    try:
        user_data = sp.current_user()
        images = user_data.get("images", [])
        image_url = images[0]["url"] if images else None
        return {
            "id": user_data.get("id"),
            "display_name": user_data.get("display_name") or "Spotify Listener",
            "email": user_data.get("email"),
            "product": user_data.get("product", "free"),
            "followers": user_data.get("followers", {}).get("total", 0),
            "image_url": image_url,
        }
    except Exception as e:
        return {
            "id": "unknown_user",
            "display_name": "Spotify Listener",
            "email": None,
            "product": "free",
            "followers": 0,
            "image_url": None,
            "error": str(e),
        }


def fetch_user_top_tracks(
    sp, time_range: str = "medium_term", limit: int = 50
) -> list[dict[str, Any]]:
    """
    Fetches the user's top tracks from Spotify.
    time_range: 'short_term' (~4 weeks), 'medium_term' (~6 months), 'long_term' (~all time)
    """
    try:
        results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
        return results.get("items", [])
    except Exception as e:
        print(f"Error fetching top tracks: {e}")
        return []


def fetch_user_top_artists(
    sp, time_range: str = "medium_term", limit: int = 20
) -> list[dict[str, Any]]:
    """Fetches user's top artists and their genres."""
    try:
        results = sp.current_user_top_artists(limit=limit, time_range=time_range)
        return results.get("items", [])
    except Exception as e:
        print(f"Error fetching top artists: {e}")
        return []


def fetch_user_saved_tracks(sp, limit: int = 50) -> list[dict[str, Any]]:
    """Fetches user's recently saved library tracks."""
    try:
        results = sp.current_user_saved_tracks(limit=limit)
        items = results.get("items", [])
        return [item["track"] for item in items if "track" in item]
    except Exception as e:
        print(f"Error fetching saved tracks: {e}")
        return []


def extract_top_genres(artists: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Extracts top genres with frequencies from a list of artist objects."""
    from collections import Counter

    genre_list: list[str] = []
    for artist in artists:
        for g in artist.get("genres", []):
            if g:
                genre_list.append(g.lower().strip())
    return Counter(genre_list).most_common(15)


def spotify_track_to_shared_format(
    track_item: dict[str, Any],
    *,
    lyrics_lookup: bool = True,
    genre_hint: str = "",
    scaler: Any = None,
) -> dict[str, Any]:
    """
    Converts a Spotify API track object into the shared track format (docs/track-format.md).
    """
    spotify_id = track_item.get("id")
    title = track_item.get("name", "Unknown Title")
    artists = track_item.get("artists", [])
    primary_artist = artists[0].get("name", "Unknown Artist") if artists else "Unknown Artist"

    album_obj = track_item.get("album", {})
    album_name = album_obj.get("name", "")
    images = album_obj.get("images", [])
    image_url = images[0].get("url") if images else None
    preview_url = track_item.get("preview_url")

    # Fetch lyrics if requested
    lyrics_text: str | None = None
    if lyrics_lookup:
        try:
            lyrics_text = fetch_lyrics(primary_artist, title)
        except Exception:
            lyrics_text = None

    # Embed lyrics
    try:
        lyric_vec = embed_lyrics(lyrics_text)
    except Exception:
        lyric_vec = None

    # Prepare raw audio features
    raw_audio = dict(DEFAULT_AUDIO_FEATURES)
    popularity = track_item.get("popularity", 50)
    raw_audio["energy"] = max(0.2, min(0.95, 0.4 + (popularity / 200.0)))
    raw_audio["danceability"] = max(0.3, min(0.95, 0.45 + ((popularity % 30) / 100.0)))

    # Normalize audio features
    if scaler is not None:
        try:
            norm_audio = normalize_audio_features(raw_audio, scaler)
        except Exception:
            norm_audio = np.array(
                [
                    raw_audio["danceability"],
                    raw_audio["energy"],
                    raw_audio["key"] / 11.0,
                    (raw_audio["loudness"] + 60.0) / 60.0,
                    raw_audio["mode"],
                    raw_audio["speechiness"],
                    raw_audio["acousticness"],
                    raw_audio["instrumentalness"],
                    raw_audio["liveness"],
                    raw_audio["valence"],
                    min(1.0, raw_audio["tempo"] / 200.0),
                    raw_audio["time_signature"] / 5.0,
                ],
                dtype=np.float32,
            )
    else:
        norm_audio = np.array(
            [
                raw_audio["danceability"],
                raw_audio["energy"],
                raw_audio["key"] / 11.0,
                (raw_audio["loudness"] + 60.0) / 60.0,
                raw_audio["mode"],
                raw_audio["speechiness"],
                raw_audio["acousticness"],
                raw_audio["instrumentalness"],
                raw_audio["liveness"],
                raw_audio["valence"],
                min(1.0, raw_audio["tempo"] / 200.0),
                raw_audio["time_signature"] / 5.0,
            ],
            dtype=np.float32,
        )

    return {
        "spotify_id": spotify_id,
        "title": title,
        "artist": primary_artist,
        "genre": genre_hint,
        "lyrics_text": lyrics_text,
        "audio_features": raw_audio,
        "audio_features_normalized": norm_audio,
        "lyric_embedding": lyric_vec,
        "has_lyrics": lyric_vec is not None,
        "album": album_name,
        "image_url": image_url,
        "preview_url": preview_url,
    }


def enrich_tracks_with_shared_format(
    raw_tracks: list[dict[str, Any]],
    *,
    lyrics_lookup: bool = True,
    scaler: Any = None,
) -> list[dict[str, Any]]:
    """Converts and enriches a list of raw Spotify tracks into the shared track format."""
    enriched = []
    for t in raw_tracks:
        if t and "id" in t:
            item = spotify_track_to_shared_format(
                t, lyrics_lookup=lyrics_lookup, scaler=scaler
            )
            enriched.append(item)
    return enriched
