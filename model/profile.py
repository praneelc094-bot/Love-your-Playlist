"""
Taste profile builder for the Love your Playlist model.

build_profile(user_tracks: list[dict]) -> dict

Builds a user's music taste profile from their top tracks / library.
Combines:
  - Average lyric semantic embedding (theme / topic / mood in words)
  - Preferred audio-feature distribution (mean & std across tempo, energy, valence, etc.)
  - Top genres and artist affinity distribution
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from embeddings import AUDIO_FEATURE_COLS


def build_profile(user_tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Builds a taste profile from a list of user tracks in the shared track format.
    
    Parameters:
        user_tracks: list of dicts with keys:
            - 'spotify_id': str
            - 'title': str
            - 'artist': str
            - 'genre': str (optional)
            - 'lyrics_text': str | None
            - 'audio_features': dict (raw features)
            - 'audio_features_normalized': np.ndarray (12-dim vector)
            - 'lyric_embedding': np.ndarray | None (384-dim vector)
            - 'has_lyrics': bool
            
    Returns:
        A profile dict containing:
            - 'lyric_embedding': np.ndarray | None (normalized mean lyric vector)
            - 'audio_features_mean': np.ndarray (12-dim mean audio vector)
            - 'audio_features_std': np.ndarray (12-dim std audio vector)
            - 'raw_audio_averages': dict[str, float]
            - 'top_genres': list[tuple[str, int]]
            - 'top_artists': list[tuple[str, int]]
            - 'track_count': int
            - 'has_lyrics_count': int
            - 'has_lyrics_ratio': float
    """
    if not user_tracks:
        raise ValueError("user_tracks cannot be empty to build a taste profile.")

    n_tracks = len(user_tracks)

    # 1. Aggregate Lyric Embeddings
    lyric_vectors: list[np.ndarray] = []
    lyric_weights: list[float] = []

    for idx, track in enumerate(user_tracks):
        emb = track.get("lyric_embedding")
        if emb is not None and isinstance(emb, (np.ndarray, list)) and len(emb) > 0:
            vec = np.asarray(emb, dtype=np.float32)
            # Higher weight for top-ranked songs
            rank_weight = 1.0 - 0.5 * (idx / max(1, n_tracks - 1))
            lyric_vectors.append(vec)
            lyric_weights.append(rank_weight)

    if lyric_vectors:
        weights_arr = np.array(lyric_weights, dtype=np.float32)[:, np.newaxis]
        weighted_sum = np.sum(np.array(lyric_vectors) * weights_arr, axis=0)
        norm = np.linalg.norm(weighted_sum)
        mean_lyric_embedding = (weighted_sum / norm) if norm > 0 else weighted_sum
    else:
        mean_lyric_embedding = None

    # 2. Aggregate Audio Features
    norm_audio_list: list[np.ndarray] = []
    raw_audio_accum: dict[str, list[float]] = {col: [] for col in AUDIO_FEATURE_COLS}

    for track in user_tracks:
        norm_feat = track.get("audio_features_normalized")
        if norm_feat is not None and isinstance(norm_feat, (np.ndarray, list)):
            norm_audio_list.append(np.asarray(norm_feat, dtype=np.float32))

        raw_feat = track.get("audio_features")
        if isinstance(raw_feat, dict):
            for col in AUDIO_FEATURE_COLS:
                if col in raw_feat and raw_feat[col] is not None:
                    raw_audio_accum[col].append(float(raw_feat[col]))

    if norm_audio_list:
        audio_matrix = np.array(norm_audio_list)
        audio_mean = np.mean(audio_matrix, axis=0).astype(np.float32)
        audio_std = np.std(audio_matrix, axis=0).astype(np.float32)
        audio_std = np.where(audio_std < 1e-4, 1.0, audio_std)
    else:
        audio_mean = np.zeros(len(AUDIO_FEATURE_COLS), dtype=np.float32)
        audio_std = np.ones(len(AUDIO_FEATURE_COLS), dtype=np.float32)

    raw_averages = {
        col: float(np.mean(vals)) if vals else 0.0
        for col, vals in raw_audio_accum.items()
    }

    # 3. Artist and Genre Distributions
    genres: list[str] = []
    artists: list[str] = []
    for track in user_tracks:
        g = track.get("genre")
        if g and str(g).strip().lower() not in ["", "none", "unknown"]:
            genres.append(str(g).strip().lower())
        a = track.get("artist")
        if a and str(a).strip():
            artists.append(str(a).strip())

    top_genres = Counter(genres).most_common(10)
    top_artists = Counter(artists).most_common(10)

    has_lyrics_count = len(lyric_vectors)
    has_lyrics_ratio = has_lyrics_count / n_tracks if n_tracks > 0 else 0.0

    return {
        "lyric_embedding": mean_lyric_embedding,
        "audio_features_mean": audio_mean,
        "audio_features_std": audio_std,
        "raw_audio_averages": raw_averages,
        "top_genres": top_genres,
        "top_artists": top_artists,
        "track_count": n_tracks,
        "has_lyrics_count": has_lyrics_count,
        "has_lyrics_ratio": has_lyrics_ratio,
    }
